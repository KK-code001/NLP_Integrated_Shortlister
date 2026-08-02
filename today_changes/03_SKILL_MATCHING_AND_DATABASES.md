# 03. Skill Matching & Database Taxonomy Changes

This document details updates to the skill matching engine (`app/services/skill_matcher.py`) and knowledge base dictionaries (`data/`).

---

## 1. Skill Extraction & Fuzzy Matching Engine

### File Modified:
- `app/services/skill_matcher.py`

### Key Changes & Rationale:
1. **Fuzzy Similarity Tuning**:
   - Tuned RapidFuzz ratio thresholds to capture skill variations without producing false positives.
   - Example matches enabled: `scikit-learn` $\leftrightarrow$ `sklearn`, `React.js` $\leftrightarrow$ `React`, `PostgreSQL` $\leftrightarrow$ `Postgres`.
2. **Skill Categorization**:
   - Categorized skills cleanly into:
     - **Matched Required Skills**: Skills explicitly requested by JD and found in Candidate resume.
     - **Missing Required Skills**: Skills requested by JD but missing in Candidate resume.
     - **Bonus Skills**: Candidate skills not explicitly required by JD, but relevant as value-adds.

---

## 2. Skill Knowledge Base Updates

### Files Modified:
- `data/skills_db.json`
- `data/domain_skills.json`

### Key Changes & Rationale:
1. **Expanded Technical Dictionary**:
   - Added modern AI/ML, NLP, Cloud, and Web technologies (e.g., PyTorch, HuggingFace, LangChain, Docling, Ollama, SBERT, FastAPI, Docker, Kubernetes).
2. **Domain Mapping Taxonomy**:
   - Enhanced domain skill dictionaries for accurate domain match scoring (`domain_match`: `YES`/`NO`) across AI/ML, NLP, Web Development, Data Engineering, and DevOps.
