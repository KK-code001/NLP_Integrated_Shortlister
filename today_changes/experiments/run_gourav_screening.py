import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline import screen_candidate
from app.services.report_generator import print_candidate_report

res_path = r"C:\Users\Vivaan\Downloads\gourav_Resume (1).pdf"
jd_path  = os.path.abspath(r"ipdocs/jd ca.docx")

print("=" * 70)
print(f"RESUME PATH : {res_path}")
print(f"JD PATH     : {jd_path}")
print("=" * 70 + "\n")

report = screen_candidate(res_path, jd_path, is_file=True)

print_candidate_report(report)

print("\n" + "=" * 70)
print("  PARSED RESUME METADATA")
print("=" * 70)
parsed = report.get("parsed_resume", {})
print(f"  Name      : {parsed.get('name')}")
print(f"  Email     : {parsed.get('email')}")
print(f"  Phone     : {parsed.get('phone')}")
print(f"  Total Exp : {report.get('resume_experience')} yrs")
print(f"  Jobs Count: {len(parsed.get('jobs', []))}")
for j in parsed.get("jobs", []):
    print(f"    * {j.get('designation')} @ {j.get('company')} [{j.get('start_date')} to {j.get('end_date')}]")
print(f"  Education : {parsed.get('education')}")
meta = report.get("parsing_metadata", {})
if meta.get("warnings"):
    print(f"  Warnings  : {meta.get('warnings')}")
print("=" * 70 + "\n")
