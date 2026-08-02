"""
Semantic section detection.

Converts a ResumeDocument into:
    dict[canonical_section_name, list[TextBlock]]

A TextBlock is treated as a section header when ANY of the following is true:
  1. Its block_type is "heading"  (set by Docling)
  2. Its text (lowercased, stripped) exactly matches a known section alias
  3. It is short (≤ 5 words), ALL-CAPS, and has no commas/semicolons
  4. It is bold AND short (≤ 6 words) with no commas/semicolons
     (PyMuPDF sets is_bold=True for bold spans)

Rule (2) is the key guard against false positives: the word "experience"
inside a long summary paragraph will NOT trigger a new section because
it doesn't appear as a standalone, short heading-like line.
"""
from __future__ import annotations

import re
from app.parser.layout import ResumeDocument, TextBlock

# ---------------------------------------------------------------------------
# Section alias registry
# ---------------------------------------------------------------------------

SECTION_ALIASES: dict[str, list[str]] = {
    "experience": [
        "experience",
        "professional experience",
        "work experience",
        "internships",
        "internship",
        "internship experience",
        "internships & experience",
        "internships and experience",
        "work experience & internships",
        "work experience / internships",
        "internships / work experience",
        "practical experience",
        "employment",
        "employment history",
        "career history",
        "work history",
        "job history",
        "professional background",
        "industry experience",
        "work background",
        # Academic / teaching CV variants
        "teaching experience",
        "academic experience",
        "research experience",
        "academic and research experience",
        "teaching and research experience",
        "professional and research experience",
    ],
    "education": [
        "education",
        "academic background",
        "academic qualifications",
        "qualifications",
        "scholastic details",
        "educational background",
        "academic credentials",
        "educational qualifications",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core competencies",
        "technologies",
        "tech stack",
        "areas of expertise",
        "expertise",
        "key skills",
        "skills & expertise",
        "skills and expertise",
        "tools & technologies",
        "tools and technologies",
        "technical competencies",
        "it skills",
    ],
    "projects": [
        "projects",
        "personal projects",
        "academic projects",
        "key projects",
        "project experience",
        "notable projects",
        "selected projects",
        "project highlights",
        "research projects",
        "research and projects",
        "major projects",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "licenses",
        "credentials",
        "professional certifications",
        "courses",
        "courses & certifications",
        "courses and certifications",
        "training",
        "professional development",
        "awards & certifications",
        "certifications & awards",
        "online courses",
        "professional courses",
    ],
    "summary": [
        "summary",
        "professional summary",
        "career summary",
        "profile",
        "about me",
        "about",
        "objective",
        "career objective",
        "professional profile",
        "executive summary",
        "personal statement",
        "introduction",
        "overview",
    ],
    "languages": [
        "languages",
        "language proficiency",
        "spoken languages",
        "language skills",
    ],
    "contact": [
        "contact",
        "contact information",
        "contact details",
        "personal information",
        "personal details",
    ],
    "awards": [
        "awards",
        "honors",
        "honours",
        "achievements",
        "recognition",
        "accomplishments",
    ],
    "publications": [
        "publications",
        "research",
        "papers",
        "research papers",
        "journal articles",
        "conference papers",
        "in conference proceedings",
        "submitted / under review publications",
        "submitted/under review publications",
        "under review",
        "preprints",
        "book chapters",
    ],
    "references": [
        "references",
        "referees",
    ],
    "interests": [
        "interests",
        "hobbies",
        "hobbies & interests",
        "hobbies and interests",
        "extracurricular activities",
        "activities",
    ],
    "volunteering": [
        "volunteering",
        "volunteer experience",
        "social work",
        "community service",
    ],
}

# Build a fast lookup: alias (lowercase, stripped) → canonical section name
_ALIAS_TO_SECTION: dict[str, str] = {}
for _section, _aliases in SECTION_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_SECTION[_alias.lower().strip()] = _section

# Strip leading bullets / numbering before alias matching
_BULLET_PREFIX_RE = re.compile(
    r"^[\u2022\u2023\u25E6\u2043\u2219\u2027\-\*\.\d]+[\.\):]?\s*"
)


def _clean_for_match(text: str) -> str:
    """Remove leading bullets/numbering, lowercase, and normalize character spacing."""
    cleaned = _BULLET_PREFIX_RE.sub("", text).strip().lower()
    if cleaned in _ALIAS_TO_SECTION:
        return cleaned
    
    # Handle character-spaced PDF text headers like "E D U C AT I O N" or "P R O F I L E"
    collapsed = re.sub(r"\s+", "", cleaned)
    if collapsed in _ALIAS_TO_SECTION:
        return collapsed

    return cleaned


def _is_heading(block: TextBlock) -> bool:
    """
    Return True if the block looks like a section heading.
    The logic here is intentionally conservative to avoid false positives.
    """
    text = block.text.strip()
    if not text:
        return False

    # Rule 1: Docling explicitly labelled this as a heading
    if block.block_type == "heading":
        return True

    cleaned = _clean_for_match(text)

    # Rule 2: Direct alias match (most reliable)
    if cleaned in _ALIAS_TO_SECTION:
        return True

    # Rules 3 & 4 only apply to short, punctuation-light lines
    words = text.split()
    has_inline_punctuation = bool(re.search(r"[,;]", text))

    # Rule 3: ALL CAPS short line (no commas/semicolons)
    if (
        len(words) <= 5
        and not has_inline_punctuation
        and text == text.upper()
        and re.search(r"[A-Z]", text)
    ):
        return True

    # Rule 4: Bold + short + no inline punctuation
    if block.is_bold and len(words) <= 6 and not has_inline_punctuation:
        return True

    return False


def _resolve_section(text: str) -> str | None:
    """
    Return the canonical section name for a heading-like block, or None
    if the heading text does not match any known alias.
    """
    cleaned = _clean_for_match(text)
    return _ALIAS_TO_SECTION.get(cleaned)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sections(doc: ResumeDocument) -> dict[str, list[TextBlock]]:
    """
    Segment a ResumeDocument into named sections.

    Returns:
        A dict mapping canonical section names to their content blocks.
        The special "header" key holds all blocks before the first
        recognised section (usually name, contact info, tagline).
    """
    sections: dict[str, list[TextBlock]] = {}
    current_section = "header"
    sections[current_section] = []

    for block in doc.blocks:
        if _is_heading(block):
            resolved = _resolve_section(block.text)
            if resolved:
                # Start a new section; do NOT include the heading itself as content
                current_section = resolved
                if current_section not in sections:
                    sections[current_section] = []
                continue
            # Heading-like block that doesn't match any known alias → treat as content
        sections.setdefault(current_section, []).append(block)

    # Post-process: check header/summary for orphan job blocks (e.g. "Teaching Assistant", date ranges)
    _JOB_TITLE_KEYWORDS = {"assistant", "intern", "developer", "engineer", "manager", "analyst", "lead", "architect", "consultant", "specialist"}
    _DATE_RANGE_RE = re.compile(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4})\s*[\-–—\sto]+\s*(?:present|current|now|\d{4})", re.IGNORECASE)

    orphan_job_blocks = []
    for sec in ("header", "summary"):
        sec_blocks = sections.get(sec, [])
        for b in sec_blocks:
            b_lower = b.text.lower()
            if _DATE_RANGE_RE.search(b_lower) or any(re.search(rf"\b{kw}\b", b_lower) for kw in _JOB_TITLE_KEYWORDS):
                if not any(token in b_lower for token in ["@gmail", "@yahoo", "linkedin.com", "github.com", "aspiring"]):
                    orphan_job_blocks.append(b)

    if orphan_job_blocks:
        if "experience" not in sections:
            sections["experience"] = []
        exp_texts = {b.text for b in sections["experience"]}
        for ob in orphan_job_blocks:
            if ob.text not in exp_texts:
                sections["experience"].insert(0, ob)

    return sections


def section_to_text(blocks: list[TextBlock]) -> str:
    """Flatten a list of TextBlocks into a single string for LLM prompts and rules."""
    lines = []
    for b in blocks:
        text = b.text.strip()
        if not text:
            continue
        if b.block_type == "list_item" and not re.match(r"^[\-\*•·▪▸►‣⁃→✓]", text):
            text = f"- {text}"
        lines.append(text)
    return "\n".join(lines)
