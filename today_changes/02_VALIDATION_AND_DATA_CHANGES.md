# 02. Validation & Data Cleaning Changes

This document details the updates made to pre-ML data validation, job experience sanitization, and date parsing utilities (`app/utils/`).

---

## 1. Work Experience Validation Fixes

### File Modified:
- `app/utils/validation.py`

### Key Changes & Rationale:
1. **Handling Missing Company Names**:
   - **Previous Behavior**: If a candidate resume entry lacked an explicit company name (or the parser failed to extract the company name), `validate_job()` dropped the entire job entry. This resulted in loss of valid work experience duration and designation data.
   - **Updated Behavior**: Empty or missing company fields are now automatically assigned the placeholder `"Company Not Identified"`.
   - **Impact**: The candidate's experience duration and job title are preserved, ensuring accurate total experience calculations and feature building for the ML classifier.

2. **Sanitization Rules**:
   - Validated designation strings and start/end dates.
   - Suppressed non-fatal parsing warnings while retaining data integrity.

---

## 2. Date Parsing & Range Calculation

### File Modified:
- `app/utils/dates.py`

### Key Changes & Rationale:
1. **Flexible Date Format Support**:
   - Enhanced regex patterns to handle date formats such as:
     - Month Year: `Jan 2021`, `September 2022`
     - Numeric: `06/2020`, `2020-12`
     - Ongoing: `Present`, `Current`, `Till Date`, `Ongoing`
2. **Robust Duration Calculation**:
   - Safely defaults missing start or end dates to reasonable fallbacks without raising uncaught exceptions.
