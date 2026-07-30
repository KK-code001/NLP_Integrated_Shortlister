"""
run.py — IDE entry point for the Resume Screener.

Usage (in VS Code / PyCharm):
    1. Set resume_path and jd_path below to your file paths.
    2. Press Run (F5 / Shift+F10).

Supports: PDF, DOCX, TXT, JPG, PNG
"""

import sys
import os
from pathlib import Path

# ── Make sure the project root is on sys.path ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── File paths — EDIT THESE ───────────────────────────────────────────────────
resume_path = input("Enter the path to the resume file: ")     # ← change this
jd_path     = input("Enter the path to the job description file: ")  # ← change this
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Validate paths ────────────────────────────────────────────────────────
    if not os.path.exists(resume_path):
        print(f"[ERROR] Resume not found: {resume_path}")
        print("  → Edit the resume_path variable at the top of run.py")
        sys.exit(1)

    if not os.path.exists(jd_path):
        print(f"[ERROR] Job description not found: {jd_path}")
        print("  → Edit the jd_path variable at the top of run.py")
        sys.exit(1)

    print("=" * 60)
    print("  RESUME SCREENER")
    print("=" * 60)
    print(f"  Resume : {resume_path}")
    print(f"  JD     : {jd_path}")
    print("=" * 60)
    print()

    # ── Run the pipeline ──────────────────────────────────────────────────────
    try:
        from app.pipeline import screen_candidate
    except ImportError as e:
        print(f"[ERROR] Could not import pipeline: {e}")
        print("  → Make sure all dependencies are installed.")
        print("  → Run:  pip install -r requirements.txt")
        sys.exit(1)

    # ── Ollama Pre-flight Check ───────────────────────────────────────────────
    import urllib.request
    from app.config import OLLAMA_HOST, OLLAMA_MODEL
    print("[*] Checking local Ollama connection...")
    try:
        # Check if the host port responds
        urllib.request.urlopen(OLLAMA_HOST, timeout=2)
        print(f"  → Ollama server found at {OLLAMA_HOST}")
    except Exception:
        print(f"\n[WARNING] Ollama server is not reachable at {OLLAMA_HOST}!")
        print("  → Make sure the Ollama application is running on your machine.")
        print("  → The pipeline will run, but will fall back to rule-based parsing.")
        print()

    print("[1/4] Parsing resume with Docling ...")
    print("[2/4] Detecting sections + running LLM extraction ...")
    print("[3/4] Calculating experience in Python ...")
    print("[4/4] Running Random Forest + SHAP ...")
    print()

    report = screen_candidate(resume_path, jd_path, is_file=True)

    # ── Check for errors ──────────────────────────────────────────────────────
    if "error" in report:
        print(f"[ERROR] {report['error']}")
        sys.exit(1)

    # ── Print report ──────────────────────────────────────────────────────────
    _print_report(report)


def _print_report(report: dict):
    W = 78
    
    # Extract details
    m_count = len(report.get("matched_skills", []))
    tot_req = m_count + len(report.get("missing_skills", []))
    confidence_pct = report.get("confidence", 0.0) * 100
    is_review = report.get("needs_human_review", False)
    status = report.get("status_decision") or ("UNDER HUMAN REVIEW" if is_review else "RELIABLE MATCH FOR JD")
    
    # In the original, the prediction value is also mapped to the status variable
    if "prediction" in report and not report.get("status_decision"):
        pred = str(report["prediction"]).lower()
        if "no" in pred:
            status = "REJECTED / NOT A FIT"
        else:
            status = "RELIABLE MATCH FOR JD" if not is_review else "UNDER HUMAN REVIEW"

    def fmt_years(val):
        if val is None or (isinstance(val, float) and sys.modules.get('pandas') and sys.modules['pandas'].isna(val)):
            return "Not stated / unverified"
        return f"{val:.1f} yrs"

    def fmt_pct(val):
        try:
            return f"{float(val):.1%}" if float(val) <= 1 else f"{float(val):.1f}%"
        except (TypeError, ValueError):
            return "N/A"

    print("\n" + "=" * W)
    print("  CANDIDATE EVALUATION REPORT".center(W))
    print("=" * W)
    print(f"  Candidate Name       : {report.get('candidate_name', 'Unknown').upper()}")
    print(f"  Model Confidence     : {confidence_pct:.1f}%  (Decision Threshold: 55%)")
    print(f"  Final Decision       : {status}")
    print(f"  Human Review Needed  : {'YES - score below threshold, please verify manually' if is_review else 'NO - confidently matched to JD'}")
    print("-" * W)
    
    method = report.get("extraction_method") or report.get("parsing_metadata", {}).get("extraction_method", "N/A")
    parser = report.get("parsing_metadata", {}).get("parser_used", "N/A")
    engine = f"Modular Pipeline ({method})"
    exp_source = f"Layout-aware sections via {parser}"
    
    print(f"  Extraction Engine    : {engine}")
    print(f"  Experience Source    : {exp_source}")

    print("\n" + "-" * W)
    print("  SKILLS ANALYSIS")
    print("-" * W)
    matched_skills_pct = report.get("matched_skills_pct") or (round((m_count / tot_req) * 100, 1) if tot_req > 0 else 0.0)
    print(f"  Overall Skill Match  : {m_count}/{tot_req} required skills matched ({matched_skills_pct}%)")
    if report.get('matched_skills'):
        print(f"  Matched Skills       : {', '.join(s.title() for s in report['matched_skills'])}")
    if report.get('missing_skills'):
        print(f"  Missing Skills       : {', '.join(s.title() for s in report['missing_skills'])}")
    if report.get('unmapped_skills'):
        print(f"  Bonus Skills         : {', '.join(str(s).title() for s in report['unmapped_skills'])}  (not required by JD, but a plus)")

    print("\n" + "-" * W)
    print("  EXPERIENCE & EDUCATION")
    print("-" * W)
    print(f"  Candidate Experience : {fmt_years(report.get('resume_experience'))}")
    print(f"  JD Requires          : {fmt_years(report.get('jd_experience'))}")
    if report.get('is_overqualified'):
        print(f"  Overqualification    : YES - candidate experience significantly exceeds JD requirement")
    
    edu_lvl = report.get('education_level') or report.get('resume_education_level') or 'Unknown'
    print(f"  Education Level      : {edu_lvl}")
    
    edu_deg = report.get('education_degree') or report.get('resume_education') or 'Unknown'
    if edu_deg and edu_deg != 'Unknown':
        print(f"  Education Details    : {edu_deg}")

    print("\n" + "-" * W)
    print("  DOMAIN & SEMANTIC FIT")
    print("-" * W)
    print(f"  Candidate Domain     : {report.get('resume_domain', 'Unknown')}")
    print(f"  JD Domain            : {report.get('jd_domain', 'Unknown')}")
    
    domain_match = report.get('domain_match')
    if domain_match is None and report.get('features'):
        domain_match = report['features'].get('domain_match')
    print(f"  Domain Match         : {'YES' if domain_match else 'NO'}")
    
    sim = report.get('semantic_similarity_score')
    if sim is None and report.get('features'):
        sim = report['features'].get('semantic_similarity_score', 0.0)
    print(f"  Semantic Similarity  : {fmt_pct(sim)}")

    if report.get('certifications'):
        print("\n" + "-" * W)
        print("  CERTIFICATIONS")
        print("-" * W)
        for c in report['certifications']:
            name = c.get('name') if isinstance(c, dict) else str(c)
            print(f"  - {name}")

    # Top Factors (SHAP feature impact)
    reasons = report.get("reasons", [])
    if reasons:
        print("\n" + "-" * W)
        print("  TOP FACTORS DRIVING THIS DECISION (model explainability)")
        print("-" * W)
        for r in reasons:
            if isinstance(r, dict):
                direction = "increased" if r.get("direction") == "supported" else "decreased"
                print(f"  - {r.get('factor')} {direction} match confidence (value: {r.get('value')}, impact: {r.get('impact'):+.3f})")
            else:
                print(f"  - {r}")

    # Value Addition Insights
    warnings = report.get("parsing_metadata", {}).get("warnings", [])
    insights = []
    if warnings:
        insights.extend(warnings)
    parsed = report.get("parsed_resume", {})
    if parsed and parsed.get("jobs"):
        insights.append(f"Parsed jobs list count: {len(parsed['jobs'])}")
        for j in parsed['jobs']:
            insights.append(f"  * Job: {j.get('designation')} @ {j.get('company')} ({j.get('start_date')} to {j.get('end_date')})")
    
    if insights:
        print("\n" + "-" * W)
        print("  VALUE-ADD INSIGHTS")
        print("-" * W)
        for v in insights:
            print(f"  - {v}")

    # Learning Recommendations
    recs = report.get("dynamic_recommendations", [])
    if recs:
        print("\n" + "-" * W)
        print("  SUGGESTED UPSKILLING (for missing JD skills)")
        print("-" * W)
        for rec in recs:
            if isinstance(rec, dict):
                print(f"  - {str(rec.get('skill', '')).title():<22} -> {rec.get('resource', '')}")

    print("\n" + "=" * W + "\n")


if __name__ == "__main__":
    main()
