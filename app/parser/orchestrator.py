"""
Top-level orchestrator: coordinates the full parsing pipeline.

  parse_resume_file(path)  →  ResumeSchema dict

Steps:
  1. Layout-aware document parse  (Docling → PyMuPDF → fallbacks)
  2. Semantic section detection
  3. Deterministic contact extraction (regex)
  4. LLM extraction — one focused call per section
  5. Python-level experience calculation (no LLM math)
  6. Education normalization
  7. Skill merge (LLM + text-matched)
  8. Pre-ML validation
  9. Schema assembly and normalization

Exposes extract_from_document() separately so tests can inject a
pre-built ResumeDocument without touching the file system.
"""
from __future__ import annotations

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
from app.parser.experience import calculate_total_experience, separate_internships
from app.parser.education import best_education_record
from app.parser.skills import merge_skills
from app.services.skill_matcher import extract_skills_exact
from app.utils.validation import validate_and_clean_jobs
from app.utils.schema import validate_schema


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

    contact = extract_contact_info(header_text)

    # ------------------------------------------------------------------
    # Step 2: Experience — LLM extraction of section only
    # ------------------------------------------------------------------
    exp_text    = section_to_text(sections.get("experience", []))
    exp_result  = extract_experience_section(exp_text) if exp_text else {"jobs": []}

    if "error" in exp_result:
        warnings.append(f"Experience extraction error: {exp_result['error']}")

    raw_jobs = exp_result.get("jobs", [])
    valid_jobs, job_warnings = validate_and_clean_jobs(raw_jobs)
    warnings.extend(job_warnings)

    # Calculate total experience in Python — LLM never does this
    total_years = calculate_total_experience(valid_jobs)
    _full_time, internships = separate_internships(valid_jobs)

    # ------------------------------------------------------------------
    # Step 3: Education — LLM extraction of section only
    # ------------------------------------------------------------------
    edu_text    = section_to_text(sections.get("education", []))
    edu_result  = extract_education_section(edu_text) if edu_text else {"education": []}

    if "error" in edu_result:
        warnings.append(f"Education extraction error: {edu_result['error']}")

    education_list = edu_result.get("education", [])

    # ------------------------------------------------------------------
    # Step 4: Skills — LLM + text-based merge
    # ------------------------------------------------------------------
    skills_text      = section_to_text(sections.get("skills", []))
    llm_skills_result = (
        extract_skills_section(skills_text) if skills_text else {"skills": [], "confidence": 0.0}
    )

    if "error" in llm_skills_result:
        warnings.append(f"Skills extraction error: {llm_skills_result['error']}")

    llm_skills  = llm_skills_result.get("skills", [])
    text_skills = extract_skills_exact(resume_doc.raw_text)   # dictionary-based
    merged_skills = merge_skills(llm_skills, text_skills)

    # ------------------------------------------------------------------
    # Step 5: Certifications
    # ------------------------------------------------------------------
    cert_text   = section_to_text(sections.get("certifications", []))
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
        extraction_method = "Hybrid_Docling_Ollama"

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
