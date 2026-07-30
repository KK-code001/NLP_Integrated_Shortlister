"""
Skill post-processing — merges and deduplicates skills from
LLM extraction and text/dictionary-based extraction.

Delegates to the existing skill_matcher so all normalization logic
(aliases, implied skills, symbol substitutions) is applied once.
"""
from __future__ import annotations

from app.services.skill_matcher import combine_and_normalize_skills


def merge_skills(llm_skills: list[str], text_skills: list[str]) -> list[str]:
    """
    Merge LLM-detected skills with text/dictionary-matched skills.

    Uses the existing combine_and_normalize_skills() from skill_matcher
    to apply implied skill expansion and alias normalization.

    Returns a sorted, deduplicated list of normalized skill strings.
    """
    return combine_and_normalize_skills(text_skills, llm_skills)


def deduplicate_skills(skills: list[str]) -> list[str]:
    """Remove exact duplicates while preserving list order."""
    seen: set[str] = set()
    result: list[str] = []
    for s in skills:
        s_clean = (s or "").strip().lower()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            result.append(s_clean)
    return result
