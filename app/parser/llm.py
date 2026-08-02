"""
Section-specific LLM extractors via Ollama.

Each function receives ONLY the text of one section (not the full resume)
and returns a structured JSON dict.

Extraction rules enforced in every prompt:
  - Return JSON only — no markdown, no explanation.
  - Leave unknown fields as empty string "" — never guess.
  - Preserve date formats as written (LLM never calculates totals).
  - confidence between 0.0 and 1.0 per record.
"""
from __future__ import annotations

import json
import re
try:
    import ollama
except ImportError:
    ollama = None
from app.config import OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chat(prompt: str) -> dict:
    """Call Ollama and return parsed JSON. Raises on any failure."""
    if ollama is None:
        raise RuntimeError("ollama package not installed")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    raw = response["message"]["content"]
    return json.loads(raw)


def _safe_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip()


def _safe_list(val) -> list:
    if isinstance(val, list):
        return val
    return []


def _safe_float(val, default: float = 0.5) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Contact info — deterministic regex (no LLM)
# ---------------------------------------------------------------------------

_EMAIL_RE    = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE    = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_GITHUB_RE   = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)

# Words that should never be treated as a candidate name
_NAME_SKIP_WORDS = {
    # Locations
    "jaipur", "rajasthan", "india", "delhi", "mumbai", "bangalore", "bengaluru",
    "hyderabad", "pune", "chennai", "kolkata", "noida", "gurugram", "gurgaon",
    "lucknow", "chandigarh", "ahmedabad", "bhopal", "patna", "jodhpur",
    "new", "york", "london", "remote", "hybrid", "onsite",
    # Section headings
    "profile", "summary", "experience", "education", "skills", "projects",
    "certifications", "references", "awards", "publications", "languages",
    "contact", "objective", "overview", "interests", "volunteering",
    # Role / tech words
    "stack", "engineer", "developer", "intern", "aspiring", "seeking", "full",
    "frontend", "backend", "software", "data", "machine", "learning", "science",
    "analyst", "manager", "designer", "architect", "consultant", "specialist",
    "assistant", "professor", "researcher", "technician", "trainee", "fellow",
    "curriculum", "vitae", "resume", "professional",
}


def extract_contact_info(header_text: str, blocks: list | None = None) -> dict:
    """
    Deterministic regex-based contact extraction from the resume header.
    No LLM call — fast and reliable for structured fields.

    Uses a 4-priority chain for name extraction:
      Priority 0: Font-size detection (largest text on page 1 from blocks)
      Priority 1: LinkedIn URL name inference
      Priority 2: Scored line-scan (ALL lines, weighted by position)
      Priority 3: Email handle fallback

    Args:
        header_text: The text to search for contact info.
        blocks: Optional list of TextBlock objects with font_size metadata.

    Returns: name, email, phone, linkedin, github.
    """
    lines = [l.strip() for l in header_text.splitlines() if l.strip()]
    email = phone = linkedin = github = name = ""

    for line in lines:
        if not email:
            m = _EMAIL_RE.search(line)
            if m:
                email = m.group(0)
        if not phone:
            m = _PHONE_RE.search(line)
            if m:
                phone = m.group(0).strip()
        if not linkedin:
            m = _LINKEDIN_RE.search(line)
            if m:
                linkedin = m.group(0)
        if not github:
            m = _GITHUB_RE.search(line)
            if m:
                github = m.group(0)

    # ── Priority 0: Font-size-based name detection ────────────────────────
    # The candidate's name is almost always the largest font on page 1.
    if blocks:
        name = _extract_name_by_font_size(blocks)

    # ── Priority 1: LinkedIn URL name inference ──────────────────────────
    # Handles: linkedin.com/in/firstname-lastname-hash123
    #          linkedin.com/in/firstname-middle-lastname
    if not name:
        # Try 2-part name first (most common: firstname-lastname)
        m = re.search(
            r"linkedin\.com/in/([a-zA-Z]{2,})-([a-zA-Z]{2,})(?:-[a-zA-Z0-9]*)?",
            header_text, re.IGNORECASE,
        )
        if m:
            part1, part2 = m.group(1), m.group(2)
            # Reject if part2 looks like a hash (all lowercase + digits pattern)
            if len(part2) <= 10:
                name = f"{part1.capitalize()} {part2.capitalize()}"

    # ── Priority 2: Scored line-scan (ALL lines, position-weighted) ──────
    # Scans every line, not just the first 15. Lines earlier in the document
    # get a higher score; lines matching more "name-like" traits get bonuses.
    if not name:
        name = _extract_name_by_line_scan(lines)

    # ── Priority 3: Email handle fallback ─────────────────────────────────
    if not name and email:
        handle = email.split("@")[0]
        # Try splitting on dots/underscores first: john.doe -> John Doe
        parts = re.split(r"[._]", handle)
        alpha_parts = [re.sub(r"\d+", "", p).strip() for p in parts if re.sub(r"\d+", "", p).strip()]
        if len(alpha_parts) >= 2 and all(len(p) >= 2 for p in alpha_parts):
            name = " ".join(p.capitalize() for p in alpha_parts)
        elif alpha_parts and len(alpha_parts[0]) >= 3:
            name = alpha_parts[0].capitalize()

    return {
        "name":     name,
        "email":    email,
        "phone":    phone,
        "linkedin": linkedin,
        "github":   github,
    }


def _extract_name_by_font_size(blocks: list) -> str:
    """
    Find the candidate name by identifying the largest-font text on page 1.
    Combines adjacent blocks with matching font sizes if single-word candidates are found.
    Handles character-spaced names (e.g. 'S h o u r y a').
    """
    page1_blocks = [
        b for b in blocks
        if getattr(b, "page", 0) <= 1
        and getattr(b, "font_size", 0) > 0
    ]
    if not page1_blocks:
        return ""

    # Sort by font_size descending — largest first
    page1_blocks.sort(key=lambda b: getattr(b, "font_size", 0), reverse=True)

    candidates = []
    for i, block in enumerate(page1_blocks):
        text = block.text.strip()
        # Collapse character spacing: 'S h o u r y a' -> 'Shourya'
        if re.search(r"\b[a-zA-Z]\s+[a-zA-Z]\b", text):
            text = re.sub(r"\s+", "", text)

        cleaned = re.sub(r"[^a-zA-Z\s]", "", text).strip()
        words = cleaned.split()
        if not words:
            continue

        if any(w.lower() in _NAME_SKIP_WORDS for w in words):
            continue
        if re.search(r"\d", text) or "@" in text or "/" in text:
            continue

        # If we got a 2-5 word full name, return it immediately
        if 2 <= len(words) <= 5:
            return cleaned.title()

        # If single-word first name, look for adjacent block with similar font size to form full name
        if len(words) == 1 and len(words[0]) >= 2:
            single_name = words[0].capitalize()
            # Look at remaining blocks for surname
            for next_b in page1_blocks[i+1:]:
                next_text = next_b.text.strip()
                if re.search(r"\b[a-zA-Z]\s+[a-zA-Z]\b", next_text):
                    next_text = re.sub(r"\s+", "", next_text)
                next_cleaned = re.sub(r"[^a-zA-Z\s]", "", next_text).strip()
                next_words = next_cleaned.split()
                if 1 <= len(next_words) <= 3 and not any(nw.lower() in _NAME_SKIP_WORDS for nw in next_words):
                    if not re.search(r"\d", next_text) and "@" not in next_text:
                        return f"{single_name} {next_cleaned.title()}"
            candidates.append(single_name)

    return candidates[0] if candidates else ""


def _extract_name_by_line_scan(lines: list[str]) -> str:
    """
    Score every line in the document for 'name-likeness' and return the best.

    Scoring:
      +10 : 2-4 alphabetic words with no digits/special chars
      +5  : Appears in first 10 lines (position bonus)
      +3  : Appears in lines 10-25
      +2  : All words start with uppercase
      -50 : Contains skip words (locations, tech terms, headings)
      -50 : Contains email, URL, phone, or digits
    """
    best_score = 0
    best_name = ""

    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue

        # Collapse character spacing: 'S h o u r y a' -> 'Shourya'
        if re.search(r"\b[a-zA-Z]\s+[a-zA-Z]\b", line_clean):
            line_clean = re.sub(r"\s+", "", line_clean)

        # Immediate disqualifiers
        if "@" in line_clean or "/" in line_clean:
            continue
        if any(url_kw in line_clean.lower() for url_kw in ["linkedin", "github", "http", "www"]):
            continue
        if re.search(r"\d", line_clean):
            continue
        if "." in line_clean and not re.fullmatch(r"[A-Za-z\.\s]+", line_clean):
            continue

        # Strip non-alpha chars for name check
        cleaned = re.sub(r"[^a-zA-Z\s]", "", line_clean).strip()
        words = cleaned.split()

        if not (1 <= len(words) <= 5):
            continue

        # Skip if any word is a known non-name word
        if any(w.lower() in _NAME_SKIP_WORDS for w in words):
            continue

        # Build score
        score = 10  # Base: looks like a name

        # Position bonus (earlier = more likely to be a name)
        if idx < 5:
            score += 8
        elif idx < 10:
            score += 5
        elif idx < 25:
            score += 3

        # Word count bonus (2-3 words is ideal for names, 1 word is penalized slightly)
        if 2 <= len(words) <= 3:
            score += 5
        elif len(words) == 1:
            score -= 2

        # Capitalization bonus
        if all(w[0].isupper() for w in words if w):
            score += 2

        # All-caps bonus (common in resume headers)
        if line_clean == line_clean.upper() and len(words) >= 2:
            score += 2

        if score > best_score:
            best_score = score
            best_name = cleaned.title()

    return best_name if best_score >= 10 else ""


# ---------------------------------------------------------------------------
# Experience extractor — LLM
# ---------------------------------------------------------------------------

_EXP_SCHEMA = """{
  "jobs": [
    {
      "company":         "Company name — string, leave empty string if unknown",
      "designation":     "Job title — string, leave empty string if unknown",
      "start_date":      "Start date exactly as written, e.g. Jan 2020, 2020, 01/2020",
      "end_date":        "End date exactly as written, e.g. Dec 2022, Present, Current",
      "employment_type": "Full-time | Part-time | Internship | Contract | Freelance | leave empty if unclear",
      "description":     ["bullet point 1", "bullet point 2"],
      "confidence":      0.95
    }
  ]
}"""


def extract_experience_section(section_text: str) -> dict:
    """
    Extract structured job records from the experience section text.
    The LLM is never asked to calculate total experience.

    Returns {"jobs": [...]} or {"jobs": [], "error": "..."} on failure.
    """
    if not section_text.strip():
        return {"jobs": []}

    prompt = f"""You are an expert HR data parser extracting work experience records.

STRICT RULES:
- Return ONLY a raw JSON object. No markdown. No explanation. No preamble.
- Extract every distinct job role found in the text.
- Preserve dates EXACTLY as written (e.g. "Jan 2021", "2021", "Present").
- Leave any unknown field as an empty string "".
- Do NOT calculate or infer total experience.
- Do NOT use a year number (e.g. "2022") as a company name.
- Set confidence between 0.0 and 1.0 for each job.

EXACT OUTPUT SCHEMA:
{_EXP_SCHEMA}

EXPERIENCE SECTION TEXT TO PARSE:
{section_text}
"""
    try:
        data = _chat(prompt)
        jobs = _safe_list(data.get("jobs"))
        cleaned = []
        for job in jobs:
            cleaned.append({
                "company":         _safe_str(job.get("company")),
                "designation":     _safe_str(job.get("designation")),
                "start_date":      _safe_str(job.get("start_date")),
                "end_date":        _safe_str(job.get("end_date")),
                "employment_type": _safe_str(job.get("employment_type")),
                "description":     _safe_list(job.get("description")),
                "confidence":      _safe_float(job.get("confidence")),
            })
        return {"jobs": cleaned}
    except Exception as exc:
        return {"jobs": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Education extractor — LLM
# ---------------------------------------------------------------------------

_EDU_SCHEMA = """{
  "education": [
    {
      "institution": "University or college name",
      "degree":      "Degree name e.g. B.Tech, MBA, B.Sc",
      "field":       "Field of study e.g. Computer Science",
      "start_date":  "Start year or date as written",
      "end_date":    "End year or date as written",
      "gpa":         "GPA or CGPA as written — empty string if not mentioned",
      "confidence":  0.95
    }
  ]
}"""


def extract_education_section(section_text: str) -> dict:
    """
    Extract structured education records from the education section text.

    Returns {"education": [...]} or {"education": [], "error": "..."} on failure.
    """
    if not section_text.strip():
        return {"education": []}

    prompt = f"""You are an expert HR data parser extracting education records.

STRICT RULES:
- Return ONLY a raw JSON object. No markdown. No explanation. No preamble.
- Extract every distinct educational qualification found in the text.
- Preserve dates exactly as written.
- Leave any unknown field as an empty string "".
- Do NOT infer degree level — extract the degree name exactly as written.
- Set confidence between 0.0 and 1.0 for each entry.

EXACT OUTPUT SCHEMA:
{_EDU_SCHEMA}

EDUCATION SECTION TEXT TO PARSE:
{section_text}
"""
    try:
        data = _chat(prompt)
        entries = _safe_list(data.get("education"))
        cleaned = []
        for entry in entries:
            cleaned.append({
                "institution": _safe_str(entry.get("institution")),
                "degree":      _safe_str(entry.get("degree")),
                "field":       _safe_str(entry.get("field")),
                "start_date":  _safe_str(entry.get("start_date")),
                "end_date":    _safe_str(entry.get("end_date")),
                "gpa":         _safe_str(entry.get("gpa")),
                "confidence":  _safe_float(entry.get("confidence")),
            })
        return {"education": cleaned}
    except Exception as exc:
        return {"education": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Skills extractor — LLM
# ---------------------------------------------------------------------------

_SKILLS_SCHEMA = """{
  "skills":     ["skill1", "skill2", "skill3"],
  "confidence": 0.95
}"""


def extract_skills_section(section_text: str) -> dict:
    """
    Extract a flat list of skills from the skills section text.

    Returns {"skills": [...], "confidence": float}.
    """
    if not section_text.strip():
        return {"skills": [], "confidence": 0.0}

    prompt = f"""You are an expert HR data parser extracting technical and professional skills.

STRICT RULES:
- Return ONLY a raw JSON object. No markdown. No explanation. No preamble.
- Extract individual skill names as a flat list of lowercase strings.
- Remove trailing punctuation from each skill.
- Do NOT include purely soft skills (teamwork, communication) unless explicitly listed.
- Set overall confidence between 0.0 and 1.0.

EXACT OUTPUT SCHEMA:
{_SKILLS_SCHEMA}

SKILLS SECTION TEXT TO PARSE:
{section_text}
"""
    try:
        data = _chat(prompt)
        return {
            "skills":     [_safe_str(s).lower() for s in _safe_list(data.get("skills")) if s],
            "confidence": _safe_float(data.get("confidence")),
        }
    except Exception as exc:
        return {"skills": [], "confidence": 0.0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Certifications extractor — LLM
# ---------------------------------------------------------------------------

def extract_certifications_section(section_text: str) -> dict:
    """
    Extract a list of certifications / credentials from the certifications section.

    Returns {"certifications": [...]}.
    """
    if not section_text.strip():
        return {"certifications": []}

    prompt = f"""You are an expert HR data parser extracting certifications and credentials.

STRICT RULES:
- Return ONLY a raw JSON object. No markdown. No explanation. No preamble.
- Extract each certification, license, or credential as a clean string.
- Include issuing body if mentioned (e.g. "AWS Certified Solutions Architect – Amazon").

EXACT OUTPUT FORMAT:
{{"certifications": ["cert1", "cert2"]}}

CERTIFICATIONS SECTION TEXT TO PARSE:
{section_text}
"""
    try:
        data = _chat(prompt)
        return {
            "certifications": [
                _safe_str(c) for c in _safe_list(data.get("certifications")) if c
            ]
        }
    except Exception:
        return {"certifications": []}


# ---------------------------------------------------------------------------
# Fallback: lightweight full-resume LLM call when section detection fails
# ---------------------------------------------------------------------------

def extract_basic_info_llm(resume_text: str) -> dict:
    """
    Fallback for when section detection produces no usable sections.
    Extracts name, email, phone, skills, education, certifications.
    Does NOT ask the LLM to calculate experience.

    Returns a dict compatible with the orchestrator's needs.
    """
    prompt = f"""You are an expert HR data parser. Extract key information from this resume.

STRICT RULES:
- Return ONLY a raw JSON object. No markdown. No explanation.
- Leave unknown fields as empty string "" or empty list [].
- Do NOT calculate or estimate total years of experience.

EXACT OUTPUT FORMAT:
{{
  "name":              "full name or empty string",
  "email":             "email or empty string",
  "phone":             "phone or empty string",
  "skills":            ["skill1", "skill2"],
  "education_degree":  "degree name e.g. B.Tech or empty string",
  "education_level":   "Bachelor | Master | Doctorate | Diploma | Unknown",
  "certifications":    ["cert1", "cert2"]
}}

RESUME TEXT:
{resume_text}
"""
    try:
        data = _chat(prompt)
        return {
            "name":             _safe_str(data.get("name")),
            "email":            _safe_str(data.get("email")),
            "phone":            _safe_str(data.get("phone")),
            "skills":           [_safe_str(s).lower() for s in _safe_list(data.get("skills")) if s],
            "education_degree": _safe_str(data.get("education_degree")) or "Unknown",
            "education_level":  _safe_str(data.get("education_level")) or "Unknown",
            "certifications":   [_safe_str(c) for c in _safe_list(data.get("certifications")) if c],
        }
    except Exception:
        return {
            "name": "", "email": "", "phone": "",
            "skills": [],
            "education_degree": "Unknown",
            "education_level": "Unknown",
            "certifications": [],
        }
