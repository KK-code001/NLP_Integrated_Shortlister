# 05. CLI & Entry Point (`run.py`) Script Changes

This document details updates to the main execution script (`run.py`).

---

## 1. Input Path Cleaning & Sample Defaults

### Improvements:
- **`_clean_path()` Utility**:
  - Strips leading/trailing whitespace, quotes (`"`, `'`), and PowerShell execution wrappers (`& '...'`).
- **Default Sample Fallback**:
  - Pressing **Enter** at the input prompts automatically defaults to sample files (`ipdocs/Resume.pdf` and `ipdocs/Celebal_JD.pdf`).

---

## 2. Terminal Logging & Clean Output

### Improvements:
- **Stderr Log Handler**:
  - Directs logging messages to `sys.stderr` to prevent log interleave with stdout report printing.
- **Third-Party Noise Reduction**:
  - Suppressed verbose log warnings from libraries (`ollama`, `httpx`, `httpcore`, `transformers`, `torch`, `huggingface_hub`).

---

## 3. Error Handling & Diagnostic Dump

### Improvements:
- **Crash Protection**:
  - Wrapped pipeline execution in a `try...except` block with user-friendly error messages.
- **Diagnostic Export**:
  - Automatically exports full evaluation JSON to `extraction_debug.json` on every run.
  - Added `--debug` CLI flag support (`py -3.11 run.py --debug`).

---

## 4. Enhanced Terminal Formatting

### Improvements:
- **Experience Years & Months**:
  - Displays experience as `2.5 yrs (2 yrs 6 mos)`.
- **Width Alignment**:
  - Truncated long recommendation resource strings to fit within standard 78-character terminal widths without breaking table layout formatting.
