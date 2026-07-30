"""
Canonical ResumeSchema — every parsed resume is normalised into this shape
before being passed to the adapter layer and then feature engineering.
Downstream modules (feature_builder, ml_classifier, report_generator) are
never touched; they receive the same dict shape they always expected via the
adapter in pipeline.py.
"""
from __future__ import annotations

from typing import Any


def _default_schema() -> dict[str, Any]:
    return {
        "name": "",
        "email": "",
        "phone": "",
        "experience": {
            "total_years": None,
            "jobs": [],
        },
        "education": [],
        "skills": [],
        "projects": [],
        "certifications": [],
        "languages": [],
        "parsing_metadata": {
            "parser_used": "unknown",
            "extraction_method": "unknown",
            "warnings": [],
            "sections_detected": [],
        },
    }


def validate_schema(data: dict[str, Any]) -> dict[str, Any]:
    """
    Merge *data* into a default schema so that every downstream module
    can safely access any key without a KeyError.

    - Scalar string fields are stripped.
    - List fields are kept as-is if they are lists.
    - Nested dicts (experience, parsing_metadata) are merged shallowly.
    """
    schema = _default_schema()

    for key in ("name", "email", "phone"):
        val = data.get(key)
        if val and isinstance(val, str):
            schema[key] = val.strip()

    # Experience
    exp_in = data.get("experience", {})
    if isinstance(exp_in, dict):
        ty = exp_in.get("total_years")
        if ty is not None:
            try:
                schema["experience"]["total_years"] = float(ty)
            except (TypeError, ValueError):
                pass
        jobs = exp_in.get("jobs")
        if isinstance(jobs, list):
            schema["experience"]["jobs"] = jobs

    # Simple list fields
    for key in ("education", "skills", "projects", "certifications", "languages"):
        val = data.get(key)
        if isinstance(val, list):
            schema[key] = val

    # Metadata
    meta = data.get("parsing_metadata", {})
    if isinstance(meta, dict):
        schema["parsing_metadata"].update(meta)

    return schema
