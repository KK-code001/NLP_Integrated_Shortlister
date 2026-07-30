"""
Pre-feature-engineering validation for parsed job records.

Rejects clearly bad records (company = "2022", empty designation,
start > end) and normalises all "present" variants to "Present"
before they reach the ML pipeline.
"""
from __future__ import annotations

import re
from typing import Any

from app.utils.dates import parse_date_flexible, is_present

# A string that is only a 4-digit year — never a valid company name
_YEAR_ONLY_RE = re.compile(r"^\d{4}$")


def _looks_like_year(s: str) -> bool:
    return bool(_YEAR_ONLY_RE.match(s.strip()))


def normalize_end_date(raw: str) -> str:
    """Map all 'present' variants → 'Present'. Pass through everything else."""
    if not raw:
        return raw
    return "Present" if is_present(raw) else raw


def validate_job(job: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Validate and normalise a single job record.

    Returns (cleaned_job, warnings).
    If the record is rejected, cleaned_job is an empty dict and
    warnings contains the reason.
    """
    warnings: list[str] = []

    company     = (job.get("company") or "").strip()
    designation = (job.get("designation") or "").strip()
    start_raw   = (job.get("start_date") or "").strip()
    end_raw     = normalize_end_date((job.get("end_date") or "").strip())

    # Reject/Fix: company looks like a plain year or starts with dates
    # If the LLM accidentally includes the date range prefix (e.g. "2018-2019 Cloud Engineer"), strip it
    cleaned_company = re.sub(r'^(?:19|20)\d{2}\s*[\-–—\sto]*\s*(?:19|20)?\d{2}?\s*', '', company).strip()
    # Strip common designation words if they leak into company
    cleaned_company = re.sub(r'^(?:cloud engineer|software engineer|developer|engineer|professor|assistant professor)\s*,?\s*', '', cleaned_company, flags=re.IGNORECASE).strip()

    if _looks_like_year(cleaned_company) or not cleaned_company:
        warnings.append(
            f"Rejected job: company field '{company}' does not contain a valid company name."
        )
        return {}, warnings
    
    company = cleaned_company

    # Reject: missing company name
    if not company:
        warnings.append("Rejected job: company name is missing.")
        return {}, warnings

    # Reject: missing designation
    if not designation:
        warnings.append(
            f"Rejected job at '{company}': designation is empty."
        )
        return {}, warnings

    # Reject: start_date is after end_date (when both are present, non-"Present")
    if start_raw and end_raw and end_raw != "Present":
        start_dt = parse_date_flexible(start_raw)
        end_dt   = parse_date_flexible(end_raw)
        if start_dt and end_dt and start_dt > end_dt:
            warnings.append(
                f"Rejected job at '{company}': start_date '{start_raw}' "
                f"is after end_date '{end_raw}'."
            )
            return {}, warnings

    cleaned = {
        **job,
        "company":     company,
        "designation": designation,
        "start_date":  start_raw,
        "end_date":    end_raw,
    }
    return cleaned, warnings


def validate_and_clean_jobs(jobs: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Run validation on every job record and remove exact duplicates.

    Returns (valid_jobs, all_warnings).
    """
    all_warnings: list[str] = []
    valid_jobs: list[dict] = []
    seen: set[tuple] = set()

    for job in jobs:
        cleaned, warns = validate_job(job)
        all_warnings.extend(warns)
        if not cleaned:
            continue

        # Deduplicate by (company_lower, start_date)
        key = (
            cleaned.get("company", "").lower(),
            cleaned.get("start_date", ""),
        )
        if key in seen:
            all_warnings.append(
                f"Duplicate job entry removed: "
                f"{cleaned.get('company')} / {cleaned.get('start_date')}."
            )
            continue
        seen.add(key)
        valid_jobs.append(cleaned)

    return valid_jobs, all_warnings
