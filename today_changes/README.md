# Documentation Index for Today's Changes

This directory contains separate documentation files detailing every architectural change, bug fix, feature addition, and environment update implemented today.

---

## Documentation Files

- **[01. Parser & Layout Component Changes](01_PARSER_AND_LAYOUT_CHANGES.md)**
  - Multi-column spatial coordinate fixes, header regex expansion, layout extraction fallback chain (`Docling` -> `PyMuPDF` -> `pdfplumber` -> `OCR`), experience date interval merging, and degree standardization.

- **[02. Validation & Data Cleaning Changes](02_VALIDATION_AND_DATA_CHANGES.md)**
  - Fixed dropped experience entries by populating missing company names with `"Company Not Identified"`, and added flexible date parsing for ongoing/present roles.

- **[03. Skill Matching & Database Taxonomy Changes](03_SKILL_MATCHING_AND_DATABASES.md)**
  - RapidFuzz fuzzy matching threshold tuning, categorization into matched/missing/bonus skills, and expanded AI/ML & Web technology skill databases.

- **[04. Qualitative LLM Insights & Evaluation Reporting Changes](04_LLM_INSIGHTS_AND_REPORTING.md)**
  - Ollama (`llama3.2`) structured qualitative fit insights (executive summary, strengths, risk mitigations, career trajectory), rule-based fallback, and SHAP feature impact drivers.

- **[05. CLI & Entry Point (`run.py`) Script Changes](05_ENTRYPOINT_RUN_SCRIPT_CHANGES.md)**
  - Path cleaning utility, default sample file fallback on Enter, stderr log handling, third-party log noise reduction, `extraction_debug.json` diagnostic dump, and years & months experience display.

- **[06. Environment & Python Version Compatibility Changes](06_ENVIRONMENT_AND_PYTHON_CONFIG.md)**
  - Python version alignment to 3.11 (`.python-version` and `pyproject.toml`).

- **[EXPLANATION.md](EXPLANATION.md)**
  - Consolidated single-file technical overview of all repository changes.

---

## Experimental & Debug Artifacts

- **[experiments/](experiments/)**
  - Contains screening test scripts (`run_gourav_screening.py`, `satveer_raw.py`, `test_output.py`) and raw section/text debug dumps.
