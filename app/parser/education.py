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
        r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctor of\b",
        r"\bd\.sc\b", r"\bd\.litt\b", r"\bedd\b", r"\bd\.phil\b",
    ]),
    ("Master", [
        r"\bm\.?tech\b", r"\bm\.?e\b", r"\bm\.?sc\b", r"\bm\.?s\b",
        r"\bmba\b", r"\bm\.?a\b", r"\bm\.?com\b", r"\bm\.?ca\b",
        r"\bmaster\b", r"\bpost.?graduate\b", r"\bpg\b",
        r"\bm\.?eng\b", r"\bpgd\b", r"\bpgdm\b", r"\bm\.?phil\b",
        r"\bmaster of\b",
    ]),
    ("Bachelor", [
        r"\bb\.?tech\b", r"\bb\.?e\b", r"\bb\.?sc\b", r"\bb\.?a\b",
        r"\bbba\b", r"\bb\.?com\b", r"\bb\.?ca\b", r"\bb\.?arch\b",
        r"\bb\.?pharm\b", r"\bbachelor\b", r"\bundergraduate\b",
        r"\bb\.?eng\b", r"\bbs\b", r"\bba\b", r"\bbachelor of\b",
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


def best_education_record(education_list: list[dict]) -> tuple[str, str]:
    """
    Given a list of education records, return the (degree_name, education_level)
    of the highest qualification found.

    This preserves compatibility with the existing feature_builder which expects
    'education_degree' and 'education_level' string fields.
    """
    if not education_list:
        return "Unknown", "Unknown"

    best_degree = "Unknown"
    best_level  = "Unknown"
    best_score  = -1

    for edu in education_list:
        raw_deg = edu.get("degree", "")
        degree_name, level = normalize_degree(raw_deg)
        score = _LEVEL_ORDER.get(level, 0)
        if score > best_score:
            best_score  = score
            best_degree = degree_name
            best_level  = level

    return best_degree, best_level
