import re
import pandas as pd
import numpy as np
from app.config import (
    EXPERIENCE_UNKNOWN_SENTINEL,
    OVERQUALIFICATION_RATIO_THRESHOLD,
    HIGH_EXPERIENCE_REQUIREMENT_THRESHOLD,
    FEATURES
)
from app.services.skill_matcher import (
    extract_skills_exact,
    combine_and_normalize_skills,
    get_matched_skills,
    get_missing_skills,
    skill_match_score,
    identify_domain,
    compute_semantic_similarity
)

def extract_jd_experience(jd_text: str):
    """Extract explicit required experience years from Job Description."""
    text = str(jd_text).lower()
    patterns = [
        re.compile(r'experience\s*:?\s*(\d+)\+?\s*years'),
        re.compile(r'experience\s*:?\s*(\d+)\+?\s*year'),
        re.compile(r'(\d+)\+?\s*years'),
        re.compile(r'(\d+)\+?\s*year'),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return np.nan

def build_feature_vector(resume_text_raw: str, jd_text_raw: str, llm_extracted: dict) -> dict:
    """
    Transforms LLM extraction output + raw texts into the exact
    10 numeric features required by the Random Forest model.
    """
    # Skill Extraction
    text_resume_skills = extract_skills_exact(resume_text_raw)
    jd_skills = extract_skills_exact(jd_text_raw)

    llm_skills = llm_extracted.get("skills", [])
    resume_skills = combine_and_normalize_skills(text_resume_skills, llm_skills)

    matched_skills = get_matched_skills(resume_skills, jd_skills)
    missing_skills = get_missing_skills(resume_skills, jd_skills)
    sk_score = skill_match_score(resume_skills, jd_skills)

    # Experience Extraction & Math
    resume_exp = llm_extracted.get("total_years_experience")
    jd_exp = extract_jd_experience(jd_text_raw)

    resume_exp_mentioned = int(resume_exp is not None and not pd.isna(resume_exp))
    jd_exp_mentioned = int(not pd.isna(jd_exp))

    if resume_exp_mentioned and jd_exp_mentioned and jd_exp > 0:
        exp_score = round(min(resume_exp / jd_exp, 1.0), 2)
        exp_ratio = round(resume_exp / jd_exp, 2)
        overqualified = bool(exp_ratio >= OVERQUALIFICATION_RATIO_THRESHOLD)
    else:
        exp_score = EXPERIENCE_UNKNOWN_SENTINEL
        exp_ratio = EXPERIENCE_UNKNOWN_SENTINEL
        overqualified = False

    unstated_exp_senior = int(
        not resume_exp_mentioned and not pd.isna(jd_exp) and jd_exp >= HIGH_EXPERIENCE_REQUIREMENT_THRESHOLD
    )

    # Domain Match
    resume_domain = identify_domain(resume_skills)
    jd_domain = identify_domain(jd_skills)
    domain_match = int(resume_domain == jd_domain and resume_domain != "Unknown")

    # Semantic Similarity
    sim_score = compute_semantic_similarity(resume_text_raw, jd_text_raw)

    # Features dict for ML
    row_features = {
        "skill_match_score": sk_score,
        "matched_skill_count": len(matched_skills),
        "missing_skill_count": len(missing_skills),
        "experience_match_score": exp_score,
        "experience_ratio": exp_ratio,
        "resume_experience_mentioned": resume_exp_mentioned,
        "jd_experience_mentioned": jd_exp_mentioned,
        "unstated_experience_for_senior_role": unstated_exp_senior,
        "semantic_similarity_score": sim_score,
        "domain_match": domain_match
    }

    return {
        "features": row_features,
        "candidate_name": llm_extracted.get("candidate_name", "Unknown Candidate"),
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_experience": resume_exp,
        "jd_experience": jd_exp,
        "is_overqualified": overqualified,
        "resume_domain": resume_domain,
        "jd_domain": jd_domain,
        "education_degree": llm_extracted.get("education_degree", "Unknown"),
        "education_level": llm_extracted.get("education_level", "Unknown"),
        "certifications": llm_extracted.get("certifications", []),
        "extraction_method": llm_extracted.get("extraction_method", "Unknown")
    }
