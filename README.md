# Resume Screener — Run Guide

## Project Structure

```
NLP_applied/
├── run.py                  ← IDE entry point  (edit this to run)
├── requirements.txt        ← Python dependencies
├── app/
│   ├── pipeline.py         ← Main pipeline (screen_candidate)
│   ├── parser/             ← New layout-aware parser
│   │   ├── layout.py       ← Docling → PyMuPDF → fallbacks
│   │   ├── sections.py     ← Semantic section detection
│   │   ├── llm.py          ← Per-section Ollama extractors
│   │   ├── experience.py   ← Python-level experience calc
│   │   ├── education.py    ← Degree normalization
│   │   ├── skills.py       ← Skill merge + dedup
│   │   └── orchestrator.py ← Top-level coordinator
│   ├── utils/
│   │   ├── dates.py        ← Date parsing
│   │   ├── validation.py   ← Pre-ML validation
│   │   └── schema.py       ← Canonical ResumeSchema
│   └── services/
│       ├── feature_builder.py
│       ├── ml_classifier.py
│       ├── report_generator.py
│       └── skill_matcher.py
├── models/
│   └── rf_model.joblib     ← Trained Random Forest model
└── tests/
    └── test_parser/        ← Unit tests (72 tests, no LLM needed)
```

---

## Step 1 — Install Python dependencies

Open a terminal inside `NLP_applied/` and run:

```bash
pip install -r requirements.txt
```

> **Note:** `docling` can take a few minutes to install on first run.
> If `docling` fails to install, the parser falls back to PyMuPDF automatically.

---

## Step 2 — Start Ollama (LLM backend)

Ollama must be running locally before you run the screener.

1. Download Ollama from **https://ollama.com/download**
2. Install and open it (it runs as a background service on port 11434)
3. Pull the model once:

```bash
ollama pull llama3.2
```

4. Verify it's running:

```bash
ollama list
```

You should see `llama3.2` in the list.

> If Ollama is offline, the pipeline falls back to rule-based extraction automatically.

---

## Step 3 — Run in your IDE

### VS Code

1. Open the `NLP_applied/` folder in VS Code (`File → Open Folder`)
2. Open `run.py`
3. Edit the two paths at the top:

```python
resume_path = r"C:\path\to\resume.pdf"
jd_path     = r"C:\path\to\job_description.pdf"
```

4. Press **F5** (or click the ▶ Run button)

### PyCharm

1. Open `NLP_applied/` as a project
2. Open `run.py`
3. Edit the two paths at the top (same as above)
4. Right-click → **Run 'run'** (or press **Shift+F10**)

---

## Step 4 — Supported file formats

| Format | Support |
|--------|---------|
| `.pdf` | ✅ Full (Docling → PyMuPDF → pdfplumber → OCR) |
| `.docx` | ✅ Full (Docling → python-docx) |
| `.txt` | ✅ Plain text |
| `.jpg` / `.png` | ✅ OCR via pytesseract |

---

## What happens when you run it

```
[1/4] Parsing resume with Docling ...
[2/4] Detecting sections + running LLM extraction (per section) ...
[3/4] Calculating experience in Python ...
[4/4] Running Random Forest + SHAP ...

════════════════════════════════════════════════════════════════════════
              CANDIDATE SCREENING REPORT
════════════════════════════════════════════════════════════════════════
  Candidate     : ALICE WANG
  Decision      : MATCH  (Confidence: 82%)
  Engine        : Hybrid_Docling_Ollama
  Layout Parser : docling

  PARSED RESUME FIELDS
  ─────────────────────
  Name  : Alice Wang
  Email : alice@example.com
  Jobs  : 2 role(s) extracted
    • ML Engineer @ Google  [Jan 2021 – Present]  (conf: 95%)
    • Intern @ Amazon       [Jun 2020 – Dec 2020]  (conf: 90%)
  ...
```

---

## Run tests (no LLM needed)

```bash
python -m pytest tests/test_parser/ -v
```

All 72 tests are deterministic and do not require Ollama.

---

## Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |

Set in PowerShell before running:

```powershell
$env:OLLAMA_MODEL = "llama3.2"
python run.py
```
