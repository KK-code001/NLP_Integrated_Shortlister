# 06. Environment & Python Version Compatibility Changes

This document details changes made to project environment and dependency configuration files (`.python-version`, `pyproject.toml`).

---

## 1. Python Version Requirement Adjustment

### Files Modified:
- `.python-version`
- `pyproject.toml`

### Changes & Rationale:
1. **`.python-version`**:
   - Updated content from `3.13` to `3.11`.
2. **`pyproject.toml`**:
   - Changed `requires-python = ">=3.13"` to `requires-python = ">=3.11"`.

### Impact:
- Aligned project requirements with Python 3.11 environment setup.
- Enables seamless execution via standard Python launcher: `py -3.11 run.py`.
