"""
Education normalization.

Maps raw degree strings (as returned by the LLM extractor) into the
(degree_name, education_level) tuple expected by the existing
feature_builder pipeline.

Education level hierarchy (highest wins when multiple records exist):
  Doctorate > Master > Bachelor > Diploma > Unknown
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Regex patterns for each education level
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, list[str]]] = [
    ("Doctorate", [
        r"\bph\s*\.?\s*d\b", r"\bdoctorate\b", r"\bdoctor of\b",
        r"\bd\s*\.?\s*sc\b", r"\bd\s*\.?\s*litt\b", r"\bedd\b", r"\bd\s*\.?\s*phil\b",
    ]),
    ("Master", [
        r"\bm\s*\.?\s*tech\b", r"\bm\s*\.?\s*e\b", r"\bm\s*\.?\s*sc\b", r"\bm\s*\.?\s*s\b",
        r"\bmba\b", r"\bm\s*\.?\s*a\b", r"\bm\s*\.?\s*com\b", r"\bm\s*\.?\s*ca\b",
        r"\bmaster\b", r"\bpost.?graduate\b", r"\bpg\b",
        r"\bm\s*\.?\s*eng\b", r"\bpgd\b", r"\bpgdm\b", r"\bm\s*\.?\s*phil\b",
        r"\bmaster of\b",
    ]),
    ("Bachelor", [
        r"\bb\s*\.?\s*tech\b", r"\bb\s*\.?\s*e\b", r"\bb\s*\.?\s*sc\b", r"\bb\s*\.?\s*a\b",
        r"\bbba\b", r"\bb\s*\.?\s*com\b", r"\bb\s*\.?\s*ca\b", r"\bb\s*\.?\s*arch\b",
        r"\bb\s*\.?\s*pharm\b", r"\bbachelor\b", r"\bundergraduate\b",
        r"\bb\s*\.?\s*eng\b", r"\bbs\b", r"\bba\b", r"\bbachelor of\b",
        r"\bbachelors?\b",
    ]),
    ("Diploma", [
        r"\bdiploma\b", r"\bpolytechnic\b", r"\bpoly\b",
        r"\bassociate\b", r"\bcertificate\b", r"\badvanced diploma\b",
        r"\bssc\b", r"\bhsc\b", r"\binter\b",
    ]),
]

# Pre-compile patterns
_COMPILED_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    (level, [re.compile(pat) for pat in pats])
    for level, pats in _PATTERNS
]

_LEVEL_ORDER: dict[str, int] = {
    "Doctorate": 4,
    "Master": 3,
    "Bachelor": 2,
    "Diploma": 1,
    "Unknown": 0,
}


def normalize_degree(raw_degree: str) -> tuple[str, str]:
    """
    Map a raw degree string → (degree_name, education_level).

    Returns ("Unknown", "Unknown") if no pattern matches.

    Examples:
      "B.Tech in Computer Science"              → ("B.Tech", "Bachelor")
      "Master of Business Administration"        → ("Master of Business Administration", "Master")
      "PhD in Machine Learning"                  → ("PhD", "Doctorate")
      "Diploma in Electrical Engineering"        → ("Diploma", "Diploma")
    """
    if not raw_degree:
        return "Unknown", "Unknown"

    raw_lower = raw_degree.lower().strip()

    for level, compiled in _COMPILED_PATTERNS:
        for pat in compiled:
            if pat.search(raw_lower):
                degree_name = _extract_degree_name(raw_degree)
                return degree_name, level

    return raw_degree.strip()[:60], "Unknown"


def _extract_degree_name(raw: str) -> str:
    """
    Extract a concise degree name from a longer string.
    Stops at "in", "of", "with", "from", "at", or punctuation.
    Falls back to the first 60 characters of the raw string.
    """
    m = re.match(
        r"^([^,;(\n]+?)(?:\s+(?:in|of|with|from|at)\b|[,;(]|$)",
        raw,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return raw[:60].strip()


def deduplicate_education_records(education_list: list[dict]) -> list[dict]:
    """
    Deduplicate and clean education records.
    Filter out records that represent teaching assistant roles or course context
    and merge duplicates by (normalized_degree, institution).
    """
    if not education_list:
        return []

    seen = {}
    cleaned_records = []

    for item in education_list:
        degree = (item.get("degree") or "").strip()
        field_name = (item.get("field") or "").strip()
        institution = (item.get("institution") or "").strip()

        combined = f"{degree} {field_name}".lower()
        if "teaching assistant" in combined or "assisted students" in combined or "evaluating" in combined:
            continue

        degree_norm, _level = normalize_degree(degree)
        inst_norm = re.sub(r"[^a-z0-9]", "", institution.lower())
        key = (degree_norm.lower(), inst_norm)

        if key in seen:
            existing = seen[key]
            if not existing.get("gpa") and item.get("gpa"):
                existing["gpa"] = item["gpa"]
            if not existing.get("start_date") and item.get("start_date"):
                existing["start_date"] = item["start_date"]
            if not existing.get("end_date") and item.get("end_date"):
                existing["end_date"] = item["end_date"]
        else:
            seen[key] = dict(item)
            cleaned_records.append(seen[key])

    return cleaned_records


def best_education_record(education_list: list[dict]) -> tuple[str, str]:
    """
    Given a list of education records, return the (degree_name, education_level)
    of the highest qualification found.
    """
    cleaned = deduplicate_education_records(education_list)
    if not cleaned:
        cleaned = education_list

    if not cleaned:
        return "Unknown", "Unknown"

    best_degree = "Unknown"
    best_level  = "Unknown"
    best_score  = -1

    for edu in cleaned:
        raw_deg = edu.get("degree", "")
        degree_name, level = normalize_degree(raw_deg)
        score = _LEVEL_ORDER.get(level, 0)
        if score > best_score:
            best_score  = score
            best_degree = degree_name
            best_level  = level

    return best_degree, best_level
