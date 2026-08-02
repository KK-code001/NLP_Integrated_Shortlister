import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline import screen_candidate

res_path = os.path.abspath(r"ipdocs/Resume.pdf")
jd_path  = os.path.abspath(r"ipdocs/Celebal_JD.pdf")

print("=" * 70)
print(f"Testing Resume Path: {res_path}")
print(f"Testing JD Path    : {jd_path}")
print("=" * 70)

report = screen_candidate(res_path, jd_path, is_file=True)

print("\n" + "=" * 70)
print("              CANDIDATE SCREENING REPORT SUMMARY")
print("=" * 70)
print(f"  Candidate Name : {report.get('candidate_name')}")
print(f"  Decision       : {report.get('prediction', 'N/A').upper()}")
print(f"  Confidence     : {report.get('confidence', 0)*100:.1f}%")
print("=" * 70)
print("  PARSED RESUME DATA:")
parsed = report.get('parsed_resume', {})
print(f"    Name      : {parsed.get('name')}")
print(f"    Email     : {parsed.get('email')}")
print(f"    Phone     : {parsed.get('phone')}")
print(f"    Total Exp : {report.get('resume_experience')} yrs")
print(f"    Jobs Count: {len(parsed.get('jobs', []))}")
for j in parsed.get('jobs', []):
    print(f"      - {j.get('designation')} @ {j.get('company')} ({j.get('start_date')} to {j.get('end_date')})")
print(f"    Education : {parsed.get('education')}")
print("=" * 70)
print("  PARSING METADATA & WARNINGS:")
meta = report.get('parsing_metadata', {})
print(f"    Warnings  : {meta.get('warnings')}")
print("=" * 70 + "\n")
