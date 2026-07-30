import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r'd:\Intern\NLP_applied')

from app.pipeline import screen_candidate
report = screen_candidate(
    r'd:\Intern\NLP_applied\ipdocs\Sujal_Agrawal_Resume.pdf',
    r'd:\Intern\NLP_applied\ipdocs\jd ca.docx',
    is_file=True
)

print('=== SUJAL / CA SCREENING RESULTS ===')
print('Candidate Name:    ', report.get('candidate_name'))
print('Parsed Experience: ', report.get('resume_experience'))
print('Education Level:   ', report.get('education_level'))
print('Education Degree:  ', report.get('education_degree'))
print('Decision:          ', report.get('prediction'))
print('Confidence:        ', report.get('confidence'))
print('Matched Skills:    ', report.get('matched_skills'))
print('Missing Skills:    ', report.get('missing_skills'))
print('Warnings:          ', report.get('parsing_metadata', {}).get('warnings'))
print('Jobs Found count:  ', len(report.get('parsed_resume', {}).get('jobs', [])))
for j in report.get('parsed_resume', {}).get('jobs', []):
    print(f'  * Job: {j.get("designation")} @ {j.get("company")} ({j.get("start_date")} to {j.get("end_date")})')
