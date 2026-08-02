"""
Rule-based (deterministic) section parsers.

These run BEFORE the LLM for every section and cover the majority of
well-structured resumes without any model calls.  The LLM is only
invoked when the rule-based pass returns empty or low-confidence results.

Public API
----------
extract_experience_rules(text)     -> {"jobs": [...], "confidence": float}
extract_education_rules(text)      -> {"education": [...], "confidence": float}
extract_skills_rules(text)         -> {"skills": [...], "confidence": float}
extract_certifications_rules(text) -> {"certifications": [...], "confidence": float}
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Shared patterns
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^[\-\*•·▪▸►‣⁃→✓]\s*")

_MONTH_PAT = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)"
)
_YEAR_PAT   = r"(?:19|20)\d{2}"
_PRESENT_PAT = r"(?:present|current|now|ongoing|till\s*(?:date|now))"

# Additional date patterns: MM/YYYY, Q1-Q4, academic seasons
_MM_YEAR_PAT = r"(?:\d{1,2}[/\-]\d{4}|\d{4}[/\-]\d{1,2})"
_QUARTER_PAT = r"(?:Q[1-4]\s*" + _YEAR_PAT + r")"
_SEASON_PAT = r"(?:(?:spring|summer|fall|autumn|winter)\s+" + _YEAR_PAT + r")"

# A single date token: any of the above patterns
_DATE_TOKEN = (
    rf"(?:{_MONTH_PAT}\s+{_YEAR_PAT}|{_YEAR_PAT}|{_MONTH_PAT}"
    rf"|{_MM_YEAR_PAT}|{_QUARTER_PAT}|{_SEASON_PAT})"
)

# Matches full date ranges: "January 2026 - Present", "Jan 2026 - Feb 2026",
# "2026 - Present", "06/2020 - 02/2023", "Q1 2024 - Q3 2024",
# "Summer 2023 - Fall 2023", "October 2025 - November 2025"
_DATE_RANGE_RE = re.compile(
    rf"{_DATE_TOKEN}"
    r"[\s,]*[\-–—/][\s,]*"
    rf"(?:{_PRESENT_PAT}|{_DATE_TOKEN})",
    re.IGNORECASE,
)

_DATE_SPLIT_RE = re.compile(r"\s*[\-–—]\s*|\s+to\s+", re.IGNORECASE)

_DEGREE_RE = re.compile(
    r"\b(b\.?\s*tech|b\.?\s*e\.?|b\.?\s*sc|b\.?\s*a\.?|b\.?\s*com|bba|bca|"
    r"m\.?\s*tech|m\.?\s*e\.?|m\.?\s*sc|mba|m\.?\s*a\.?|mca|"
    r"ph\.?\s*d\.?|doctorate|diploma|b\.?\s*arch|b\.?\s*pharm|"
    r"bachelor|master|associate)\b",
    re.IGNORECASE,
)

_GPA_RE = re.compile(
    # Matches: "CGPA: 8.82", "GPA: 3.9", "CGPA (latest):8.82", "CGPA (Sem 4): 8.82"
    r"(?:gpa|cgpa|cpi|sgpa|percentage)[\w\s()]*?:?\s*([0-9]+\.[0-9]+)",
    re.IGNORECASE,
)

# Known role keywords — a line containing any of these is likely a job title
_ROLE_KW = {
    # Engineering, Tech & Product
    "engineer", "developer", "programmer", "architect", "lead", "head", "cto", "ceo",
    "founder", "co-founder", "cofounder", "vp", "director", "admin", "administrator",
    "specialist", "technician", "support", "sysadmin", "devops",
    # Design & Creative
    "designer", "illustrator", "animator", "artist", "creative",
    # Business, Finance, Legal & HR
    "manager", "consultant", "analyst", "strategist", "marketer", "marketing",
    "sales", "accountant", "auditor", "article", "articleship", "lawyer",
    "counsel", "recruiter", "hr", "executive", "officer", "associate",
    "assistant", "coordinator", "advisor", "agent", "representative",
    # Education, Research & Academia
    "researcher", "scientist", "fellow", "trainee", "intern", "internship",
    "professor", "teacher", "tutor", "instructor", "scholar", "member",
}

# Location words that should never be treated as company names
_LOCATION_WORDS = {
    "jaipur", "rajasthan", "india", "delhi", "mumbai", "bangalore",
    "bengaluru", "hyderabad", "pune", "chennai", "kolkata", "noida",
    "gurugram", "remote", "hybrid", "onsite",
}

_CERT_KW_RE = re.compile(
    r"\b(certif(?:ied|icate|ication)|license[d]?|credential|nanodegree|"
    r"specialization|aws\s|azure\s|google\s|coursera|udemy|edx)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_bullet(line: str) -> bool:
    return bool(_BULLET_RE.match(line.strip()))


def _is_date_line(line: str) -> bool:
    return bool(_DATE_RANGE_RE.search(line.strip()))


def _strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line.strip()).strip()


def _parse_date_range(line: str) -> tuple[str, str]:
    """Split a date-range string into (start, end). Returns ('', '') on failure."""
    line = line.strip()
    parts = _DATE_SPLIT_RE.split(line, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # Single date with 'Present' keyword
    if re.search(r"\bpresent\b", line, re.IGNORECASE):
        yr = re.search(r"((?:19|20)\d{2})", line)
        return (yr.group(1) if yr else line), "Present"
    return line, ""


def _is_role_line(line: str) -> bool:
    """True if the line looks like a job title."""
    line = line.strip()
    if not line or _is_bullet(line) or _is_date_line(line):
        return False
    words = line.split()
    if len(words) > 12:
        return False
        
    line_lower = line.lower()
    
    # Explicitly reject common non-roles that are title-cased
    if any(x in line_lower for x in ["my webpage", "google scholar", "linkedin", "github"]):
        return False

    # Handle inline company names like "Cloud Engineer, Google" or "Data Scientist | Meta"
    primary_part = line
    if "," in line:
        primary_part = line.split(",")[0]
    elif "|" in line:
        primary_part = line.split("|")[0]
    elif "-" in line:
        primary_part = line.split("-")[0]
        
    # Direct keyword hit anywhere in line (e.g. "AIC-JKLU, Jaipur — Intern", "Cloud Engineer, Google")
    if any(kw in line_lower for kw in _ROLE_KW):
        return True
        
    # Short, title-cased, alphabetic — e.g. "Core Member", "Project Lead"
    cleaned = re.sub(r"[^a-zA-Z\s]", "", primary_part).strip()
    cwords = cleaned.split()
    if 2 <= len(cwords) <= 5 and cleaned.istitle() and not re.search(r"\d", primary_part):
        if not any(w.lower() in _LOCATION_WORDS for w in cwords):
            return True
    return False


def _is_company_candidate(line: str) -> bool:
    """True if the line could be a company / organisation name."""
    line = line.strip()
    if not line or _is_bullet(line) or _is_date_line(line) or _is_role_line(line):
        return False
    words = line.split()
    if len(words) > 6:
        return False
    # All-location words → skip
    if all(w.lower() in _LOCATION_WORDS for w in words):
        return False
    # Contains only commas/punctuation → skip (e.g. stray "," block from PDF)
    if re.fullmatch(r"[,.\s]+", line):
        return False
    return True


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------

def extract_experience_rules(section_text: str) -> dict:
    """
    Deterministic work-experience extractor.

    Strategy
    --------
    Scan lines top-to-bottom.  When a role-title line is detected:
      - look ahead (up to 4 lines) for a company and/or date range
      - collect subsequent bullet points as description
    Build a job record and advance the cursor past all consumed lines.

    Returns {"jobs": [...], "confidence": float}
    """
    if not section_text.strip():
        return {"jobs": [], "confidence": 0.0}

    lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
    jobs: list[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if not _is_role_line(line):
            i += 1
            continue

        designation = line
        company = ""
        start_date = ""
        end_date = ""
        
        # Check for inline company
        if "," in line:
            parts = line.split(",", 1)
            designation = parts[0].strip()
            company = parts[1].strip().rstrip("., ")
        elif "|" in line:
            parts = line.split("|", 1)
            designation = parts[0].strip()
            company = parts[1].strip().rstrip("., ")
            
        company_found = bool(company)
        descriptions: list[str] = []

        # --- Lookahead: company + date (within next 4 lines) ---------------
        j = i + 1
        date_found = False

        while j < len(lines) and j <= i + 4:
            nxt = lines[j]

            if _is_date_line(nxt) and not date_found:
                start_date, end_date = _parse_date_range(nxt)
                date_found = True
                j += 1
            elif _is_company_candidate(nxt) and not company_found:
                company = nxt
                company_found = True
                j += 1
            elif _is_bullet(nxt):
                break                       # bullets start → stop lookahead
            elif _is_role_line(nxt):
                break                       # next role starts
            else:
                j += 1

        # --- Collect bullet-point descriptions -----------------------------
        k = j
        while k < len(lines):
            nxt = lines[k]
            if _is_bullet(nxt):
                descriptions.append(_strip_bullet(nxt))
                k += 1
            elif _is_role_line(nxt) and nxt != designation:
                break
            elif _is_date_line(nxt) and not date_found:
                start_date, end_date = _parse_date_range(nxt)
                date_found = True
                k += 1
            else:
                # Non-bullet, non-role content — skip silently
                k += 1
                # Stop if we've drifted far without bullets
                if k > j + 3 and not descriptions:
                    break
        
        # Skip entries with zero corroborating evidence — these are usually
        # profile/summary lines (e.g. "AI/ML Engineer") that bled into exp section
        has_evidence = company or start_date or descriptions
        if designation and has_evidence:
            # Determine employment type from designation text
            emp_type = ""
            dl = designation.lower()
            if "intern" in dl:
                emp_type = "Internship"
            elif "full" in dl and "time" in dl:
                emp_type = "Full-time"
            elif "part" in dl and "time" in dl:
                emp_type = "Part-time"
            elif "contract" in dl or "freelance" in dl:
                emp_type = "Contract"

            confidence = 0.5
            if company:
                confidence += 0.2
            if start_date:
                confidence += 0.2
            if descriptions:
                confidence += 0.1

            jobs.append({
                "company":         company,
                "designation":     designation,
                "start_date":      start_date,
                "end_date":        end_date,
                "employment_type": emp_type,
                "description":     descriptions,
                "confidence":      round(min(confidence, 1.0), 2),
            })

        i = k   # advance past all consumed lines regardless

    avg_conf = round(sum(j["confidence"] for j in jobs) / len(jobs), 2) if jobs else 0.0
    return {"jobs": jobs, "confidence": avg_conf}


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------

def extract_education_rules(section_text: str) -> dict:
    """
    Deterministic education extractor.

    Scans for degree-pattern lines; collects institution, dates, and GPA
    from the surrounding context window (±3 lines).

    Returns {"education": [...], "confidence": float}
    """
    if not section_text.strip():
        return {"education": [], "confidence": 0.0}

    lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
    entries: list[dict] = []
    used_indices: set[int] = set()

    for i, line in enumerate(lines):
        if i in used_indices:
            continue
        if not _DEGREE_RE.search(line):
            continue

        degree = line
        institution = ""
        start_date = ""
        end_date = ""
        gpa = ""

        context_start = max(0, i - 2)
        context_end = min(len(lines), i + 4)
        consumed = {i}

        for ci in range(context_start, context_end):
            if ci == i:
                continue
            ctx = lines[ci]

            # GPA
            gpa_m = _GPA_RE.search(ctx)
            if gpa_m and not gpa:
                gpa = gpa_m.group(1)
                consumed.add(ci)
                # Institution may be embedded in the same line as GPA
                # e.g. "JK Lakshmipat University, Jaipur [Sem 4th] CGPA (latest):8.82."
                if not institution:
                    clean_inst = _GPA_RE.sub("", ctx).strip()
                    clean_inst = re.sub(r"\[.*?\]", "", clean_inst).strip()
                    clean_inst = re.sub(r"[^\x00-\x7F]+", "", clean_inst).strip()  # strip non-ASCII (°, etc.)
                    clean_inst = clean_inst.rstrip(",. ").strip()
                    if len(clean_inst.split()) >= 2:
                        institution = clean_inst
                continue

            # Date range
            if _is_date_line(ctx) and not start_date:
                start_date, end_date = _parse_date_range(ctx)
                consumed.add(ci)
                continue

            # Institution — non-degree, non-bullet, multi-word
            if (not institution
                    and not _DEGREE_RE.search(ctx)
                    and not _is_bullet(ctx)
                    and not _is_date_line(ctx)
                    and len(ctx.split()) >= 2
                    and not re.fullmatch(r"[,.\s]+", ctx)):
                # Strip embedded GPA/bracket noise from institution text
                # e.g. "JKLU, Jaipur [Sem 4th] CGPA (latest):8.82."
                clean_inst = re.sub(
                    r"(?:gpa|cgpa|cpi|sgpa)[\w\s()]*?:?\s*[0-9]+\.[0-9]+\.?",
                    "", ctx, flags=re.IGNORECASE
                ).strip()
                clean_inst = re.sub(r"\[.*?\]", "", clean_inst).strip()
                clean_inst = clean_inst.rstrip(",.").strip()
                if clean_inst:
                    institution = clean_inst
                consumed.add(ci)

        used_indices.update(consumed)
        entries.append({
            "institution": institution,
            "degree":      degree,
            "field":       "",
            "start_date":  start_date,
            "end_date":    end_date,
            "gpa":         gpa,
            "confidence":  0.85,
        })

    avg_conf = 0.85 if entries else 0.0
    return {"education": entries, "confidence": avg_conf}


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def extract_skills_rules(section_text: str) -> dict:
    """
    Deterministic skills extractor.

    Uses the dictionary-based `extract_skills_exact` as the primary source,
    then falls back to delimiter-splitting the raw text.

    Returns {"skills": [...], "confidence": float}
    """
    if not section_text.strip():
        return {"skills": [], "confidence": 0.0}

    # Primary: dictionary-based exact + fuzzy matching
    try:
        from app.services.skill_matcher import extract_skills_exact
        dict_skills = extract_skills_exact(section_text)
    except Exception:
        dict_skills = []

    if dict_skills:
        return {"skills": dict_skills, "confidence": 0.9}

    # Fallback: delimiter split
    raw_tokens = re.split(r"[,|•\n;/]+", section_text)
    skills: list[str] = []
    for tok in raw_tokens:
        cleaned = re.sub(r"[^a-zA-Z0-9\s\.\+#]", "", tok).strip().lower()
        if 2 <= len(cleaned) <= 40 and cleaned:
            skills.append(cleaned)

    unique = sorted(set(skills))
    return {
        "skills":     unique,
        "confidence": 0.65 if unique else 0.0,
    }


# ---------------------------------------------------------------------------
# Certifications
# ---------------------------------------------------------------------------

def extract_certifications_rules(section_text: str) -> dict:
    """
    Deterministic certifications extractor.

    Collects lines that either contain certification keywords or look like
    short, clean credential titles within a certifications section.

    Returns {"certifications": [...], "confidence": float}
    """
    if not section_text.strip():
        return {"certifications": [], "confidence": 0.0}

    lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
    certs: list[str] = []

    for line in lines:
        clean = _strip_bullet(line)
        if not clean or len(clean) < 5:
            continue
        words = clean.split()
        if len(words) > 15:
            continue
        if _CERT_KW_RE.search(clean) or (1 <= len(words) <= 12 and not _is_date_line(clean)):
            certs.append(clean)

    return {
        "certifications": certs,
        "confidence":     0.75 if certs else 0.0,
    }
