"""
Main pipeline entry point.

Routing:
  is_file=True  → New layout-aware parsing path (Docling → section LLMs → Python calc)
  is_file=False → Legacy path: raw text passed directly, falls back to full-resume LLM

The new parsing path converts ResumeSchema → legacy llm_extracted dict via
_to_legacy_format() so that build_feature_vector(), evaluate_candidate_ml(),
and assemble_final_report() are called with exactly the same dict shape as before.
None of those modules are modified.
"""
from __future__ import annotations

from app.parser.layout import parse_resume_document
from app.parser.orchestrator import extract_from_document
from app.parser.education import best_education_record
from app.services.text_extraction import safe_extract_text_from_file
from app.services.llm_extractor import extract_structured_data_llm
from app.services.feature_builder import build_feature_vector
from app.services.ml_classifier import evaluate_candidate_ml
from app.services.report_generator import assemble_final_report


def _to_legacy_format(resume_schema: dict) -> dict:
    """
    Convert the new ResumeSchema into the legacy llm_extracted dict
    that build_feature_vector() expects (unchanged interface).

    Mapped fields:
      candidate_name         ← schema["name"]
      total_years_experience ← schema["experience"]["total_years"]
      skills                 ← schema["skills"]
      education_degree       ← best_education_record(schema["education"])[0]
      education_level        ← best_education_record(schema["education"])[1]
      certifications         ← schema["certifications"]
      extraction_method      ← schema["parsing_metadata"]["extraction_method"]
    """
    exp = resume_schema.get("experience", {})
    edu_list = resume_schema.get("education", [])
    edu_degree, edu_level = best_education_record(edu_list)

    name = (resume_schema.get("name") or "").strip() or "Unknown Candidate"

    return {
        "candidate_name":         name,
        "total_years_experience": exp.get("total_years"),   # float | None
        "skills":                 resume_schema.get("skills", []),
        "education_degree":       edu_degree,
        "education_level":        edu_level,
        "certifications":         resume_schema.get("certifications", []),
        "extraction_method":      resume_schema.get("parsing_metadata", {}).get(
                                      "extraction_method", "Unknown"
                                  ),
    }


def screen_candidate(
    resume_file_or_text: str,
    jd_file_or_text: str,
    is_file: bool = True,
) -> dict:
    """
    Main execution pipeline for screening a candidate against a job description.

    When is_file=True (default):
      1. Parses resume via layout-aware Docling parser
      2. Detects sections, runs per-section LLM extraction
      3. Calculates experience in Python (no LLM math)
      4. Validates and normalises into ResumeSchema
      5. Converts to legacy llm_extracted dict (adapter)

    When is_file=False:
      Falls back to original full-resume LLM extraction on raw text.

    Either path then continues with:
      6. Build 10-feature vector
      7. RandomForest prediction + SHAP explanations
      8. Assemble final report
    """
    if is_file:
        # ── New layout-aware parsing path ──────────────────────────────
        resume_doc    = parse_resume_document(resume_file_or_text)
        resume_schema = extract_from_document(resume_doc)

        # Use the raw document text for semantic similarity computation
        resume_text_for_features = resume_doc.raw_text

        jd_text, jd_err = safe_extract_text_from_file(jd_file_or_text)
        if jd_err:
            # JD was passed as literal text instead of a file path — use it directly
            jd_text = jd_file_or_text

        llm_extracted = _to_legacy_format(resume_schema)

    else:
        # ── Legacy path: raw text (no file to parse with Docling) ──────
        resume_text_for_features = resume_file_or_text
        jd_text = jd_file_or_text
        llm_extracted = extract_structured_data_llm(resume_text_for_features)
        resume_schema = None

    # ── Steps shared by both paths ─────────────────────────────────────

    # Step 1: Build 10-Feature Vector (UNTOUCHED)
    feature_data = build_feature_vector(
        resume_text_for_features, jd_text, llm_extracted
    )

    # Step 2: RandomForest Prediction & SHAP Explanations (UNTOUCHED)
    ml_eval = evaluate_candidate_ml(feature_data)

    # Step 3: Assemble Final Report (UNTOUCHED)
    final_report = assemble_final_report(feature_data, ml_eval)

    # Attach parsing metadata for observability (additive — no existing keys changed)
    if resume_schema is not None:
        final_report["parsing_metadata"] = resume_schema.get("parsing_metadata", {})
        final_report["parsed_resume"] = {
            "name":      resume_schema.get("name"),
            "email":     resume_schema.get("email"),
            "phone":     resume_schema.get("phone"),
            "jobs":      resume_schema.get("experience", {}).get("jobs", []),
            "education": resume_schema.get("education", []),
        }

    return final_report
