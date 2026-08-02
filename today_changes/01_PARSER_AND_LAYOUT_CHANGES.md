# 01. Parser & Layout Component Changes

This document details all updates, bug fixes, and architectural refinements made to the layout parsing and section extraction modules (`app/parser/`).

---

## 1. Section Detection & Orchestration

### Files Modified:
- `app/parser/sections.py`
- `app/parser/orchestrator.py`

### Changes & Improvements:
1. **Multi-Column Layout Boundary Tracking**:
   - Fixed spatial coordinate sorting for two-column resume layouts.
   - Prevented headings in the right column from improperly capturing text content from the left column.
2. **Regex & Keyword Header Expansion**:
   - Expanded matching patterns for non-standard section headers (e.g., "Professional Summary", "Experience & Projects", "Technical Expertise", "Key Achievements").
3. **Orchestrator Routing**:
   - Added robust fallback routing: if LLM section parsing fails or returns incomplete sections, the orchestrator seamlessly falls back to rule-based spatial section extraction without crashing.

---

## 2. Low-Level Document Extraction Pipeline

### Files Modified:
- `app/parser/layout.py`

### Changes & Improvements:
1. **Parser Fallback Chain**:
   - Standardized the extraction progression: **Docling** (primary layout-aware parser) $\rightarrow$ **PyMuPDF** (two-column aware fallback) $\rightarrow$ **pdfplumber** (plain text fallback) $\rightarrow$ **OCR (Pillow/Tesseract)**.
2. **Bounding Box Coordinates**:
   - Added block-level coordinate tracking to preserve natural reading order across multi-page and multi-column documents.

---

## 3. Experience & Date Interval Calculations

### Files Modified:
- `app/parser/experience.py`
- `app/utils/dates.py`

### Changes & Improvements:
1. **Human-Readable Experience Formatting**:
   - Introduced `format_years_and_months()` helper function.
   - Converts raw floating-point years (e.g. `2.5` years) into descriptive text (`2.5 yrs (2 yrs 6 mos)`).
2. **Date Interval Merging**:
   - Merged overlapping job tenure intervals to avoid double-counting years of experience when a candidate worked multiple roles simultaneously.

---

## 4. Degree & Education Normalization

### Files Modified:
- `app/parser/education.py`

### Changes & Improvements:
1. **Degree Level Standardization**:
   - Standardized degrees into canonical tiers: `Bachelor's`, `Master's`, `Doctorate`, `High School`.
   - Added recognition for non-standard variations: B.E., B.Tech, B.S., M.S., MCA, Dual Degree.
