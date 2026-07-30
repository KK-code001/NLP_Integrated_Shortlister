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
import ollama
from app.config import OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chat(prompt: str) -> dict:
    """Call Ollama and return parsed JSON. Raises on any failure."""
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


def extract_contact_info(header_text: str) -> dict:
    """
    Deterministic regex-based contact extraction from the resume header.
    No LLM call — fast and reliable for structured fields.

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

    # Name heuristic: first non-empty line with 2–4 words, no digits, no @
    for line in lines[:8]:
        cleaned = re.sub(r"[^a-zA-Z\s]", "", line).strip()
        words = cleaned.split()
        if (
            2 <= len(words) <= 4
            and "@" not in line
            and not re.search(r"\d", line)
        ):
            name = line.strip()
            break

    return {
        "name":     name,
        "email":    email,
        "phone":    phone,
        "linkedin": linkedin,
        "github":   github,
    }


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
