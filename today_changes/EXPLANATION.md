# Comprehensive Summary of Changes & Updates

This document provides a detailed breakdown of all fixes, feature additions, parser refinements, and code updates implemented in this repository.

---

## 1. File-by-File Technical Breakdown

### `run.py`
- **Purpose**: Main entry point script for running candidate resume screening against job descriptions.
- **Changes Made**:
  - Implemented `_clean_path()` function to strip quotes, whitespace, and PowerShell `& '...'` wrapper symbols from input paths.
  - Added default sample file fallback (`ipdocs/Resume.pdf` and `ipdocs/Celebal_JD.pdf`) when pressing Enter without entering paths.
  - Configured logging handlers to send logs to `sys.stderr` and silenced noisy third-party loggers (`httpx`, `httpcore`, `ollama`, `transformers`, `torch`, `huggingface_hub`).
  - Added exception handling around `screen_candidate()` to report pipeline crashes cleanly.
  - Added automatic JSON diagnostic dump to `extraction_debug.json` and added `--debug` CLI flag support.
  - Updated experience formatting to display decimal years alongside years & months (e.g. `2.5 yrs (2 yrs 6 mos)`).
  - Truncated long recommendation resource URLs to keep terminal report layout aligned at 78 characters.

### `app/parser/orchestrator.py`
- **Purpose**: Top-level coordinator for multi-stage resume parsing (layout extraction -> section splitting -> LLM / rule-based field extraction -> experience calculation).
- **Changes Made**:
  - Improved layout-aware section routing to ensure multi-column text blocks are correctly mapped to their corresponding logical sections.
  - Added fallback handling when section headers are ambiguous or missing.

### `app/parser/sections.py`
- **Purpose**: Semantic section detection and header classification engine.
- **Changes Made**:
  - Refined regex pattern matching and keyword dictionaries for headers like "Professional Experience", "Technical Skills", "Education", "Projects", and "Certifications".
  - Fixed spatial sorting logic so right-side column headers in multi-column layouts do not steal body content from left-side columns.

### `app/parser/layout.py`
- **Purpose**: Low-level document extraction pipeline (Docling -> PyMuPDF -> pdfplumber -> OCR).
- **Changes Made**:
  - Improved fallback transition between Docling layout extraction and PyMuPDF text block extraction.
  - Added block coordinate tracking to maintain correct reading order across complex PDF layouts.

### `app/parser/experience.py`
- **Purpose**: Calculates total work experience in years from extracted job entries.
- **Changes Made**:
  - Integrated `format_years_and_months()` helper function to present experience in human-readable `X yrs Y mos` format.
  - Improved date interval merging logic to handle overlapping job tenure ranges accurately.

### `app/parser/education.py`
- **Purpose**: Normalizes raw degree text into canonical education levels (`Bachelor's`, `Master's`, `Doctorate`, etc.).
- **Changes Made**:
  - Expanded degree keyword matching to support non-standard degree titles and abbreviations (e.g., B.E., B.Tech, B.S., M.S., MCA).

### `app/parser/rules.py` & `app/parser/llm.py`
- **Purpose**: Rule-based fallback extraction routines and per-section LLM prompt extractors.
- **Changes Made**:
  - Standardized JSON parsing schema outputs to guarantee consistent keys (`company`, `designation`, `start_date`, `end_date`) across both LLM and rule-based fallback paths.

### `app/utils/dates.py`
- **Purpose**: Date parsing and year-range calculation utilities.
- **Changes Made**:
  - Added support for flexible date formats including month-year strings (`Jan 2021`), numerical dates (`06/2020`), and present-day keywords (`Present`, `Current`, `Till Date`).

### `app/utils/validation.py`
- **Purpose**: Pre-ML validation and sanitization of parsed candidate data.
- **Changes Made**:
  - Modified `validate_job()` so that job entries missing a company name are assigned `"Company Not Identified"` instead of being completely rejected, preserving experience duration and title data.

### `app/services/skill_matcher.py`
- **Purpose**: Skill extraction and matching against job description requirements.
- **Changes Made**:
  - Fine-tuned exact and RapidFuzz fuzzy matching thresholds to prevent false positives while correctly matching skill variations (e.g. `scikit-learn` vs `sklearn`).
  - Improved classification of matched required skills, missing required skills, and bonus skills.

### `app/services/llm_extractor.py` & `app/services/llm_insights.py`
- **Purpose**: Standalone LLM insight engine for generating qualitative candidate evaluations.
- **Changes Made**:
  - Implemented structured Ollama prompt templates to extract executive fit summaries, key candidate strengths, gap/risk factors (with severity and mitigation), and career trajectory assessments.
  - Built automatic fallback to rule-based qualitative insights when Ollama is offline.

### `app/services/report_generator.py`
- **Purpose**: Formats the final candidate screening report dictionary.
- **Changes Made**:
  - Integrated qualitative LLM insights, SHAP explainability factors, and detailed parser metadata into the unified candidate evaluation report schema.

### `data/skills_db.json` & `data/domain_skills.json`
- **Purpose**: Knowledge base dictionaries for skill aliases and domain taxonomy mapping.
- **Changes Made**:
  - Updated and expanded canonical skill lists and domain keywords for AI/ML, NLP, Software Development, Data Science, and Web Development.

### `.python-version` & `pyproject.toml`
- **Purpose**: Environment and dependency configuration files.
- **Changes Made**:
  - Updated `.python-version` from `3.13` to `3.11`.
  - Updated `pyproject.toml` `requires-python` requirement from `">=3.13"` to `">=3.11"`.

### `today_changes/`
- **Purpose**: Organization folder for today's updates, documentation, and test artifacts.
- **Contents**:
  - `EXPLANATION.md`: Full technical documentation.
  - `experiments/`: Contains screening scripts (`run_gourav_screening.py`, `satveer_raw.py`, `test_output.py`) and raw text/section debug dumps (`mohit_sections_debug.txt`, `satveer_full_raw.txt`, `col_sort_results.txt`, `lines_dump.txt`, `cv_sections_debug.txt`).

---

## 2. High-Level Summary of Improvements

1. **Parser & Layout Reliability**: Multi-column resumes parse cleanly with reduced section header leakage.
2. **Data Retention**: Experience entries with missing company names are retained as `"Company Not Identified"`.
3. **Qualitative Intelligence**: Ollama integration produces executive summaries, strengths, risk mitigations, and career trajectory insights.
4. **Usability & Terminal Output**: `run.py` defaults to sample files on Enter, silences third-party log noise, formats experience in years and months, and exports full JSON to `extraction_debug.json`.
5. **Python 3.11 Alignment**: Configured project metadata for Python 3.11 compatibility.
