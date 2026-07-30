"""
Python-level experience calculation from structured job records.

The LLM is NOT involved here. All date parsing and arithmetic is deterministic.

Algorithm:
  1. Parse (start_date, end_date) for every job using the flexible date parser.
  2. Build a list of (start_datetime, end_datetime) intervals.
  3. Sort intervals by start date.
  4. Merge overlapping or adjacent intervals (handles concurrent jobs correctly).
  5. Sum total months across merged intervals.
  6. Return total years rounded to 1 decimal place.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.utils.dates import parse_date_flexible, months_between


def calculate_total_experience(jobs: list[dict]) -> Optional[float]:
    """
    Calculate total years of professional experience from a list of validated job records.

    Returns None if no valid date intervals can be extracted from the jobs list.
    Returns a float (years, rounded to 1 decimal) otherwise.

    Overlap handling: if two jobs overlap in time (e.g. a part-time role during
    full-time employment), the overlapping period is counted only once.
    """
    intervals: list[tuple[datetime, datetime]] = []

    for job in jobs:
        start_raw = (job.get("start_date") or "").strip()
        end_raw   = (job.get("end_date") or "").strip()

        start_dt = parse_date_flexible(start_raw) if start_raw else None
        end_dt   = parse_date_flexible(end_raw)  if end_raw   else datetime.now()

        # Treat missing end date as current
        if end_dt is None:
            end_dt = datetime.now()

        if start_dt is None:
            continue  # Cannot place this job on the timeline

        if end_dt < start_dt:
            continue  # Invalid interval — already caught by validation, skip anyway

        intervals.append((start_dt, end_dt))

    if not intervals:
        return None

    # Sort by start date ascending
    intervals.sort(key=lambda t: t[0])

    # Merge overlapping/adjacent intervals
    merged: list[tuple[datetime, datetime]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlapping — extend the current merged interval if needed
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Sum months across all merged intervals
    total_months = sum(months_between(s, e) for s, e in merged)

    if total_months <= 0:
        return None

    return round(total_months / 12.0, 1)


def separate_internships(jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split job records into (full_time_or_unknown, internships).

    Internship detection uses the employment_type field set by the LLM extractor.
    Jobs with no employment_type are classified as full-time.
    """
    full_time:   list[dict] = []
    internships: list[dict] = []

    for job in jobs:
        emp_type = (job.get("employment_type") or "").lower()
        if "intern" in emp_type:
            internships.append(job)
        else:
            full_time.append(job)

    return full_time, internships
