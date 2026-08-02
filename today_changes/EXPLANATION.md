# Comprehensive Summary of Changes & Updates

This document provides a detailed breakdown of all fixes, feature additions, parser refinements, and code updates implemented in this repository.

---

## 1. Directory & Code Base Cleaning
- Organized experimental screening scripts and raw text/section debug dumps into `today_changes/experiments/` to maintain a clean root directory.
- Created `today_changes/EXPLANATION.md` to document all architectural decisions, bug fixes, and feature enhancements.

---

## 2. Core Parser & Layout Enhancements (`app/parser/`)

### A. Section Detection (`app/parser/sections.py` & `app/parser/orchestrator.py`)
- **Multi-Column Awareness**: Improved spatial boundary tracking and heading classification for multi-column resumes.
- **Section Parsing Robustness**: Enhanced regex and heuristic heading matching to prevent section misattributions (e.g., handling non-standard headings like "Professional Summary", "Experience & Projects", "Technical Expertise").

### B. Experience Calculation (`app/parser/experience.py` & `app/utils/dates.py`)
- **Date Range Parsing**: Updated date extraction to parse diverse date formats (e.g., `Jan 2021 – Present`, `06/2020 - 12/2020`, `2019-2021`).
- **Overlap Handling**: Implemented date-interval merging to prevent double-counting overlapping work experiences.
- **Formatted Output**: Added `format_years_and_months()` helper to convert raw decimal experience (e.g., `2.5 yrs`) into human-readable years and months (e.g., `2.5 yrs (2 yrs 6 mos)`).

### C. Degree & Education Normalization (`app/parser/education.py`)
- **Degree Standardization**: Mapped varied degree titles (e.g., "B.Tech", "Bachelor of Engineering", "BS in Computer Science") to canonical education levels (`Bachelor's`, `Master's`, `Doctorate`, `High School`).

---

## 3. Pre-ML Validation & Data Cleaning (`app/utils/validation.py`)
- **Empty Company Handling**: Fixed a critical issue where work experience entries with missing company names were being completely dropped during validation.
- **Placeholder Assignment**: Work experience entries with missing companies are now assigned `"Company Not Identified"` rather than being rejected, preserving valid experience duration and designation data.

---

## 4. Skill Extraction & Matching Engine (`app/services/skill_matcher.py` & `data/`)
- **Skill Databases Updated**: Expanded `data/skills_db.json` and `data/domain_skills.json` with broader domain-specific technologies and skill aliases.
- **Fuzzy & Exact Match Tuning**: Optimized RapidFuzz similarity thresholds to reduce false positives while capturing skill variations (e.g., `React.js` vs `React`, `Scikit-Learn` vs `sklearn`).

---

## 5. Qualitative LLM Insights & Report Generation (`app/services/` & `run.py`)
- **Standalone LLM Insight Engine (`app/services/llm_insights.py`)**:
  - Structured extraction using Ollama (`llama3.2`).
  - Generates executive fit summaries, candidate strengths, gap/risk factors (with severity and mitigation), and career trajectory assessments.
  - Automatically falls back to rule-based analysis if Ollama is unavailable.
- **Report Generator Formatting (`app/services/report_generator.py` & `run.py`)**:
  - Enhanced terminal report layout with clean section dividers, model confidence percentages, SHAP feature impact drivers, and value-add parser metadata.
  - Added truncation for long resource URLs in upskilling recommendations to prevent line wrapping on standard 78-character terminals.

---

## 6. Execution & Configuration Updates (`run.py`, `.python-version`, `pyproject.toml`)
- **CLI & Path Input (`run.py`)**:
  - Implemented `_clean_path()` function to strip quotes, trailing spaces, and PowerShell `& '...'` prefixes.
  - Added default sample file fallback: pressing **Enter** without a path automatically runs `ipdocs/Resume.pdf` and `ipdocs/Celebal_JD.pdf`.
- **Logging Clean Up**:
  - Configured logging handlers to send messages to `stderr`.
  - Silenced noisy third-party loggers (`ollama`, `httpx`, `transformers`, `torch`, `huggingface_hub`) to preserve a clean terminal report.
- **Diagnostic Dump**:
  - Automatically dumps full evaluation JSON reports to `extraction_debug.json` after every run.
  - Added `--debug` CLI flag support.
- **Python Version Alignment**:
  - Updated `.python-version` to `3.11`.
  - Updated `pyproject.toml` to `requires-python = ">=3.11"`.
