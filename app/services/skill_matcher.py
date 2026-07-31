import json
import re
import numpy as np
from rapidfuzz import fuzz, process as rf_process
from app.config import DATA_DIR

# Load skill DB files
with open(DATA_DIR / "skills_db.json", "r", encoding="utf-8") as f:
    _skills_data = json.load(f)

SKILLS_DB = _skills_data["skills_db"]
SKILL_ALIASES = _skills_data["skill_aliases"]
IMPLIED_SKILLS = _skills_data["implied_skills"]

with open(DATA_DIR / "domain_skills.json", "r", encoding="utf-8") as f:
    DOMAIN_SKILLS = json.load(f)

_SYMBOL_SAFE_REPLACEMENTS = {
    "c++": "cplusplus",
    "c#": "csharp",
    "node.js": "nodejs",
    "fp&a": "fpna",
    "m&a": "mergers and acquisitions",
    ".net": "dotnet",
    "asp.net": "aspdotnet",
}

_SKILL_PATTERNS = [(skill, re.compile(r'\b' + re.escape(skill) + r'\b')) for skill in SKILLS_DB]

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    for term, safe in _SYMBOL_SAFE_REPLACEMENTS.items():
        text = text.replace(term, safe)
    text = text.replace("&", " and ").replace("-", " ")
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\S+|linkedin\.com/\S+', ' ', text)
    text = re.sub(r'\+?\d[\d\s\-\(\)]{8,}\d', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text: str) -> str:
    text = str(text).lower()
    for phrase, replacement in SKILL_ALIASES.items():
        text = text.replace(phrase, replacement)
    return text

def extract_skills_exact(text: str) -> list:
    text_norm = normalize_text(clean_text(text))
    found = [skill for skill, pattern in _SKILL_PATTERNS if pattern.search(text_norm)]
    found = _apply_implied_skills(set(found))
    return sorted(found)

def _apply_implied_skills(skills: set) -> set:
    result = set(skills)
    for skill in list(skills):
        for implied in IMPLIED_SKILLS.get(skill, []):
            result.add(implied)
    return result

def combine_and_normalize_skills(text_extracted_skills: list, llm_extracted_skills: list) -> list:
    """Combines exact dictionary matching with LLM-detected skills."""
    combined = set(text_extracted_skills)
    for s in llm_extracted_skills:
        clean_s = normalize_text(clean_text(s))
        if clean_s in SKILLS_DB:
            combined.add(clean_s)
        elif len(clean_s) >= 2:
            combined.add(clean_s)
    
    combined = _apply_implied_skills(combined)
    return sorted(combined)

def _normalize_skill_token(skill: str) -> str:
    s = skill.lower().strip()
    _PLURAL_MAP = {
        "llms": "llm",
        "rest api": "rest apis",
        "ai agents": "ai agent",
        "web apis": "web api",
        "apis": "api",
    }
    return _PLURAL_MAP.get(s, s)

def get_matched_skills(resume_skills: list, jd_skills: list) -> list:
    res_map = {_normalize_skill_token(s): s for s in resume_skills}
    jd_norm = {_normalize_skill_token(s) for s in jd_skills}
    matched_norm = set(res_map.keys()) & jd_norm
    return sorted({res_map[m] for m in matched_norm})

def get_missing_skills(resume_skills: list, jd_skills: list) -> list:
    res_norm = {_normalize_skill_token(s) for s in resume_skills}
    jd_map = {_normalize_skill_token(s): s for s in jd_skills}
    missing_norm = set(jd_map.keys()) - res_norm
    return sorted({jd_map[m] for m in missing_norm})

def skill_match_score(resume_skills: list, jd_skills: list) -> float:
    if not jd_skills:
        return 0.0
    res_norm = {_normalize_skill_token(s) for s in resume_skills}
    jd_norm = {_normalize_skill_token(s) for s in jd_skills}
    matched = len(res_norm & jd_norm)
    return round(matched / len(jd_norm), 2)

def identify_domain(skill_list: list) -> str:
    scores = {domain: len(set(skill_list) & set(skills))
              for domain, skills in DOMAIN_SKILLS.items()}
    best_domain = max(scores, key=scores.get)
    return "Unknown" if scores[best_domain] == 0 else best_domain

# Sentence-BERT Semantic Similarity (Lazy Loaded)
_sbert_model = None
_SBERT_FAILED = False

def compute_semantic_similarity(resume_text: str, jd_text: str) -> float:
    global _sbert_model, _SBERT_FAILED
    from sklearn.metrics.pairwise import cosine_similarity

    if _SBERT_FAILED:
        return _tfidf_similarity(resume_text, jd_text)

    try:
        if _sbert_model is None:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        embeddings = _sbert_model.encode([resume_text, jd_text])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(float(score), 4)
    except Exception:
        _SBERT_FAILED = True
        return _tfidf_similarity(resume_text, jd_text)

def _tfidf_similarity(text1: str, text2: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vec.fit_transform([clean_text(text1), clean_text(text2)])
        score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    except Exception:
        score = 0.0
    return round(float(score), 4)
