"""
Flexible date parsing utilities shared across the parser layer.
All date math (experience calculation) lives here — the LLM never touches dates.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRESENT_TOKENS: frozenset[str] = frozenset({
    "present", "current", "now", "tillnow", "tildate", "tilldate",
    "ongoing", "todate", "today", "running", "till now", "till date",
    "to date", "to present",
})

_MONTH_MAP: dict[str, int] = {
    "jan": 1,  "january": 1,
    "feb": 2,  "february": 2,
    "mar": 3,  "march": 3,
    "apr": 4,  "april": 4,
    "may": 5,
    "jun": 6,  "june": 6,
    "jul": 7,  "july": 7,
    "aug": 8,  "august": 8,
    "sep": 9,  "september": 9, "sept": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_present(s: str) -> bool:
    """Return True if the string represents 'current / ongoing'."""
    normalized = re.sub(r"\s+", " ", s.lower().strip())
    # Check exact match and no-space variant
    return normalized in _PRESENT_TOKENS or normalized.replace(" ", "") in {
        t.replace(" ", "") for t in _PRESENT_TOKENS
    }


def parse_date_flexible(s: str) -> Optional[datetime]:
    """
    Parse a human-readable date string into a datetime.

    Handles:
      - "Present" / "Current" / "Now" / "Till Date" → datetime.now()
      - "Jan 2021" / "January 2021"
      - "2021" (year only → Jan of that year)
      - "01/2021" / "2021/01"
      - "2021-01" / "01-2021"
      - "01-01-2021" / "2021-01-01" / "01/01/2021"

    Returns None if the string cannot be parsed.
    """
    if not s:
        return None

    s_clean = s.strip()
    
    # Check YYYY-MM / MM-YYYY / YYYY/MM / MM/YYYY before range splitting
    m = re.match(r"^(\d{4})[\-–—/](\d{1,2})$", s_clean)
    if m:
        month_val = int(m.group(2))
        if 1 <= month_val <= 12:
            return datetime(int(m.group(1)), month_val, 1)

    m = re.match(r"^(\d{1,2})[\-–—/](\d{4})$", s_clean)
    if m:
        month_val = int(m.group(1))
        if 1 <= month_val <= 12:
            return datetime(int(m.group(2)), month_val, 1)

    # If the input contains a range separator (e.g. "Jan 2026 - Present", "2018 - 2019"),
    # split it and only parse the first part.
    for sep in (" to ", "–", "—", " - ", "-"):
        if sep in s_clean:
            parts = s_clean.split(sep)
            if parts and parts[0].strip():
                s_clean = parts[0].strip()
                break

    if is_present(s_clean):
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    s_lower = s_clean.lower()

    # "Month Year" → "Jan 2021" or "January 2021"
    m = re.match(r"^([a-z]+)\.?\s+(\d{4})$", s_lower)
    if m:
        month_str, year_str = m.group(1), m.group(2)
        month_num = _MONTH_MAP.get(month_str) or _MONTH_MAP.get(month_str[:3])
        if month_num:
            return datetime(int(year_str), month_num, 1)

    # "Year Month" → "2021 Jan"
    m = re.match(r"^(\d{4})\s+([a-z]+)\.?$", s_lower)
    if m:
        year_str, month_str = m.group(1), m.group(2)
        month_num = _MONTH_MAP.get(month_str) or _MONTH_MAP.get(month_str[:3])
        if month_num:
            return datetime(int(year_str), month_num, 1)

    # "YYYY" year only
    m = re.match(r"^(\d{4})$", s_clean)
    if m:
        return datetime(int(m.group(1)), 1, 1)

    # "MM/YYYY"
    m = re.match(r"^(\d{1,2})/(\d{4})$", s_clean)
    if m:
        return datetime(int(m.group(2)), int(m.group(1)), 1)

    # "YYYY/MM"
    m = re.match(r"^(\d{4})/(\d{1,2})$", s_clean)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)

    # "YYYY-MM" (allows hyphens, en-dashes, em-dashes)
    m = re.match(r"^(\d{4})[\-–—](\d{1,2})$", s_clean)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)

    # "MM-YYYY" (allows hyphens, en-dashes, em-dashes)
    m = re.match(r"^(\d{1,2})[\-–—](\d{4})$", s_clean)
    if m:
        return datetime(int(m.group(2)), int(m.group(1)), 1)

    # Full date formats
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s_clean, fmt)
        except ValueError:
            pass

    return None


def months_between(d1: datetime, d2: datetime) -> int:
    """Return the positive number of whole months between two datetimes."""
    if d2 < d1:
        d1, d2 = d2, d1
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)
