# 04. Qualitative LLM Insights & Evaluation Reporting Changes

This document details updates to the qualitative evaluation engine (`app/services/llm_insights.py`, `app/services/llm_extractor.py`) and report generator (`app/services/report_generator.py`).

---

## 1. Standalone LLM Qualitative Insight Engine

### Files Modified:
- `app/services/llm_insights.py`
- `app/services/llm_extractor.py`

### Key Features & Enhancements:
1. **Ollama Structured Insights**:
   - Integrated local LLM prompt execution (`llama3.2`) to evaluate qualitative fit.
   - Extracts:
     - **Executive Fit Summary**: High-level overview of candidate fit for the role.
     - **Key Candidate Strengths**: Bulleted highlights of candidate expertise.
     - **Gaps & Risk Factors**: Identified skill or experience gaps with severity rating (`HIGH`/`MEDIUM`/`LOW`) and suggested mitigations.
     - **Career Trajectory Assessment**: Analysis of career growth and role progression.
2. **Offline Rule-Based Fallback**:
   - Automatically falls back to deterministic rule-based insights if Ollama is offline or unreachable.

---

## 2. Terminal Candidate Evaluation Report Generator

### File Modified:
- `app/services/report_generator.py`

### Key Enhancements:
1. **SHAP Explainability Factors**:
   - Integrated top decision drivers from the Random Forest model into the report (`TOP FACTORS DRIVING THIS DECISION`).
2. **Unified Data Schema**:
   - Merged skill matches, experience calculations, domain alignment, and LLM insights into a single report dictionary.
