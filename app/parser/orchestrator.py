"""
Top-level orchestrator: coordinates the full parsing pipeline.

  parse_resume_file(path)  →  ResumeSchema dict

Steps:
  1. Layout-aware document parse  (Docling → PyMuPDF → fallbacks)
  2. Semantic section detection
  3. Deterministic contact extraction (regex)
  4. Rule-based section extraction  (primary — no LLM)
     └── LLM fallback per section  (only if rules return empty/low-confidence)
  5. Python-level experience calculation (no LLM math)
  6. Education normalization
  7. Skill merge (rule-based + text-matched)
  8. Pre-ML validation
  9. Schema assembly and normalization

Exposes extract_from_document() separately so tests can inject a
pre-built ResumeDocument without touching the file system.
"""
from __future__ import annotations

import re

from app.parser.layout import ResumeDocument, parse_resume_document
from app.parser.sections import detect_sections, section_to_text
from app.parser.llm import (
    extract_contact_info,
    extract_experience_section,
    extract_education_section,
    extract_skills_section,
    extract_certifications_section,
    extract_basic_info_llm,
)
from app.parser.rules import (
    extract_experience_rules,
    extract_education_rules,
    extract_skills_rules,
    extract_certifications_rules,
)
from app.parser.experience import calculate_total_experience, separate_internships
from app.parser.education import best_education_record, deduplicate_education_records
from app.parser.skills import merge_skills
from app.services.skill_matcher import extract_skills_exact
from app.utils.validation import validate_and_clean_jobs
from app.utils.schema import validate_schema

# Confidence threshold below which rule-based results are discarded in favour of LLM
_RULES_EXP_CONF_THRESHOLD  = 0.55   # at least 1 job with company+date
_RULES_EDU_CONF_THRESHOLD  = 0.70   # clear degree pattern found
_RULES_SKILL_CONF_THRESHOLD = 0.65  # at least a few skills found


def parse_resume_file(file_path: str) -> dict:
    """
    Full pipeline: file path → validated ResumeSchema dict.
    This is the single entry point for the pipeline.py adapter.
    """
    resume_doc = parse_resume_document(file_path)
    return extract_from_document(resume_doc)


def extract_from_document(resume_doc: ResumeDocument) -> dict:
    """
    Run the full extraction pipeline on an already-parsed ResumeDocument.
    Separated from parse_resume_file() to allow unit testing without I/O.
    """
    sections = detect_sections(resume_doc)
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Contact info — deterministic regex, no LLM
    # ------------------------------------------------------------------
    # Build header text: "header" section + any explicit contact section
    header_text = section_to_text(sections.get("header", []))
    for extra_sec in ("contact", "summary"):
        extra_text = section_to_text(sections.get(extra_sec, []))
        if extra_text:
            header_text = header_text + "\n" + extra_text

    contact = extract_contact_info(header_text, blocks=resume_doc.blocks)

    # Fallback: some parsers (e.g. Docling) place sidebar content — including
    # the candidate name and contact details — AFTER the main body, so they
    # land in the last detected section rather than the header.  Scanning the
    # full raw text catches these cases.
    if not contact.get("name") or not contact.get("email"):
        full_contact = extract_contact_info(resume_doc.raw_text, blocks=resume_doc.blocks)
        if not contact.get("name"):
            contact["name"] = full_contact.get("name", "")
        if not contact.get("email"):
            contact["email"] = full_contact.get("email", "")
        if not contact.get("phone"):
            contact["phone"] = full_contact.get("phone", "")
        if not contact.get("linkedin"):
            contact["linkedin"] = full_contact.get("linkedin", "")

    # ------------------------------------------------------------------
    # Step 2: Experience — rules first, LLM fallback
    # ------------------------------------------------------------------
    exp_text = section_to_text(sections.get("experience", []))

    rules_exp = extract_experience_rules(exp_text) if exp_text else {"jobs": [], "confidence": 0.0}
    used_llm_exp = False

    if rules_exp["confidence"] >= _RULES_EXP_CONF_THRESHOLD and rules_exp["jobs"]:
        exp_result = rules_exp
    else:
        # Rules found nothing useful — fall back to LLM
        exp_result = extract_experience_section(exp_text) if exp_text else {"jobs": []}
        used_llm_exp = True
        if "error" in exp_result:
            warnings.append(f"Experience extraction error (LLM): {exp_result['error']}")

    # ------------------------------------------------------------------
    # Orphan date patching — Docling sidebar pattern
    # ------------------------------------------------------------------
    def _patch_orphan_dates(jobs: list[dict]) -> None:
        undated = [j for j in jobs if not j.get("start_date")]
        if not undated:
            return
        from app.parser.rules import _DATE_RANGE_RE, _parse_date_range
        non_exp_text = "\n".join(
            section_to_text(blocks)
            for key, blocks in sections.items()
            if key not in ("experience", "education")
        )
        found_dates: list[tuple[str, str, int]] = []  # (start, end, position_in_text)
        seen_date_keys: set[tuple[str, str]] = set()
        for m in _DATE_RANGE_RE.finditer(non_exp_text):
            start, end = _parse_date_range(m.group(0))
            if not start:
                continue
            is_bare_year_start = bool(re.fullmatch(r"\d{4}", start.strip()))
            is_bare_year_end   = bool(re.fullmatch(r"\d{4}", end.strip()))
            if is_bare_year_start and is_bare_year_end:
                continue   # looks like an education date span, skip
            key = (start.lower().strip(), end.lower().strip())
            if key in seen_date_keys:
                continue
            seen_date_keys.add(key)
            found_dates.append((start.strip(), end.strip(), m.start()))

        if not found_dates:
            return

        # Proximity matching: for each undated job, find the date closest
        # to the job's company/designation text in the raw document.
        raw_text = resume_doc.raw_text.lower()
        used_dates: set[int] = set()
        for job in undated:
            job_text = f"{job.get('company', '')} {job.get('designation', '')}".lower().strip()
            if not job_text:
                continue
            job_pos = raw_text.find(job_text)
            if job_pos < 0:
                # Try just the company name
                company_lower = (job.get("company") or "").lower().strip()
                if company_lower:
                    job_pos = raw_text.find(company_lower)
            if job_pos < 0:
                continue

            # Find the closest unused date by position distance
            best_idx = -1
            best_dist = float("inf")
            for i, (start, end, date_pos) in enumerate(found_dates):
                if i in used_dates:
                    continue
                dist = abs(date_pos - job_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                job["start_date"] = found_dates[best_idx][0]
                job["end_date"]   = found_dates[best_idx][1]
                used_dates.add(best_idx)

    raw_jobs = exp_result.get("jobs", [])
    _patch_orphan_dates(raw_jobs)
    
    valid_jobs, job_warnings = validate_and_clean_jobs(raw_jobs)
    warnings.extend(job_warnings)

    # If no valid jobs were found (e.g. section detector missed heading or rules failed),
    # run LLM experience extraction on full raw document text as a complete safety net
    if not valid_jobs:
        llm_exp = extract_experience_section(resume_doc.raw_text)
        if not llm_exp.get("error"):
            raw_jobs2 = llm_exp.get("jobs", [])
            _patch_orphan_dates(raw_jobs2)
            valid_jobs2, job_warnings2 = validate_and_clean_jobs(raw_jobs2)
            if valid_jobs2:
                valid_jobs = valid_jobs2
                warnings.extend(job_warnings2)

    # Calculate total experience in Python — LLM never does this
    total_years = calculate_total_experience(valid_jobs)
    _full_time, internships = separate_internships(valid_jobs)

    # ------------------------------------------------------------------
    # Step 3: Education — rules first, LLM fallback
    # ------------------------------------------------------------------
    edu_text = section_to_text(sections.get("education", []))

    rules_edu = extract_education_rules(edu_text) if edu_text else {"education": [], "confidence": 0.0}

    if rules_edu["confidence"] >= _RULES_EDU_CONF_THRESHOLD and rules_edu["education"]:
        edu_result = rules_edu
    else:
        edu_result = extract_education_section(edu_text) if edu_text else {"education": []}
        if "error" in edu_result:
            warnings.append(f"Education extraction error (LLM): {edu_result['error']}")

    education_list = deduplicate_education_records(edu_result.get("education", []))

    # ------------------------------------------------------------------
    # Step 4: Skills — rules first, LLM fallback
    # ------------------------------------------------------------------
    skills_text = section_to_text(sections.get("skills", []))

    rules_skills = extract_skills_rules(skills_text) if skills_text else {"skills": [], "confidence": 0.0}

    if rules_skills["confidence"] >= _RULES_SKILL_CONF_THRESHOLD and rules_skills["skills"]:
        section_skills = rules_skills["skills"]
    else:
        llm_skills_result = (
            extract_skills_section(skills_text) if skills_text else {"skills": [], "confidence": 0.0}
        )
        if "error" in llm_skills_result:
            warnings.append(f"Skills extraction error (LLM): {llm_skills_result['error']}")
        section_skills = llm_skills_result.get("skills", [])

    # Always merge with full-text dictionary scan
    text_skills   = extract_skills_exact(resume_doc.raw_text)
    merged_skills = merge_skills(section_skills, text_skills)

    # ------------------------------------------------------------------
    # Step 5: Certifications — rules first, LLM fallback
    # ------------------------------------------------------------------
    cert_text = section_to_text(sections.get("certifications", []))

    rules_cert = extract_certifications_rules(cert_text) if cert_text else {"certifications": [], "confidence": 0.0}

    if rules_cert["certifications"]:
        certifications = rules_cert["certifications"]
    else:
        cert_result = (
            extract_certifications_section(cert_text) if cert_text else {"certifications": []}
        )
        certifications = cert_result.get("certifications", [])

    # ------------------------------------------------------------------
    # Step 6: Projects (raw text blocks — no LLM, preserves descriptions)
    # ------------------------------------------------------------------
    project_blocks = sections.get("projects", [])
    projects = [
        {"description": b.text}
        for b in project_blocks
        if b.text.strip()
    ]

    # ------------------------------------------------------------------
    # Step 7: Languages
    # ------------------------------------------------------------------
    lang_text = section_to_text(sections.get("languages", []))
    languages = (
        [line.strip() for line in lang_text.splitlines() if line.strip()]
        if lang_text else []
    )

    # ------------------------------------------------------------------
    # Step 8: Fallback — if section detection found nothing useful
    # ------------------------------------------------------------------
    has_key_sections = exp_text or edu_text or skills_text

    if not has_key_sections:
        warnings.append(
            "Section detection found no key sections — "
            "falling back to full-text LLM extraction."
        )
        basic = extract_basic_info_llm(resume_doc.raw_text)

        if not contact.get("name"):
            contact["name"] = basic.get("name", "")
        if not contact.get("email"):
            contact["email"] = basic.get("email", "")

        if not merged_skills:
            merged_skills = basic.get("skills", [])

        if not education_list:
            deg = basic.get("education_degree", "")
            if deg and deg != "Unknown":
                education_list = [{
                    "institution": "",
                    "degree":      deg,
                    "field":       "",
                    "start_date":  "",
                    "end_date":    "",
                    "gpa":         "",
                    "confidence":  0.4,
                }]

        if not certifications:
            certifications = basic.get("certifications", [])

        extraction_method = "Hybrid_Fallback_Ollama"
    else:
        extraction_method = "Rules_Primary_LLM_Fallback"

    # ------------------------------------------------------------------
    # Step 9: Assemble and validate canonical schema
    # ------------------------------------------------------------------
    raw_schema = {
        "name":  contact.get("name", ""),
        "email": contact.get("email", ""),
        "phone": contact.get("phone", ""),
        "experience": {
            "total_years": total_years,
            "jobs":        valid_jobs,
        },
        "education":     education_list,
        "skills":        merged_skills,
        "projects":      projects,
        "certifications": certifications,
        "languages":     languages,
        "parsing_metadata": {
            "parser_used":       resume_doc.parser_used,
            "extraction_method": extraction_method,
            "warnings":          warnings,
            "sections_detected": list(sections.keys()),
        },
    }

    return validate_schema(raw_schema)
