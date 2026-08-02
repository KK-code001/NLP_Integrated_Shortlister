import json
import re
try:
    import ollama
except ImportError:
    ollama = None
from app.config import OLLAMA_MODEL

def extract_structured_data_llm(resume_text: str) -> dict:
    """
    Universal LLM Extractor via Ollama (llama3.2).
    Converts any arbitrary resume format (date ranges, project durations,
    written numbers, complex layouts) into a clean, normalized JSON schema.
    """
    if ollama is None:
        return {}
    prompt = f"""
    You are an expert HR data parser. Extract structured information from the following resume text.
    Calculate exact total years of work experience by evaluating dates, date ranges (e.g. Jan 2021 - Present),
    written numbers (e.g. "five years"), and project durations.

    Return ONLY a raw JSON object with no markdown syntax wrapping matching this exact schema:
    {{
      "candidate_name": "extracted full name or Unknown Candidate",
      "total_years_experience": float_or_null,
      "skills": ["list", "of", "all", "technical", "and", "domain", "skills"],
      "education_degree": "degree name like B.Tech, MBA, B.Sc, or Unknown",
      "education_level": "Bachelor, Master, Doctorate, Diploma, or Unknown",
      "certifications": ["list of all certifications, certificates, licenses, or courses mentioned"]
    }}

    RESUME TEXT:
    {resume_text}
    """

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        content = response["message"]["content"]
        data = json.loads(content)

        # Standardize types
        candidate_name = str(data.get("candidate_name") or "Unknown Candidate").strip()
        exp_val = data.get("total_years_experience")
        
        try:
            total_exp = float(exp_val) if exp_val is not None and str(exp_val).lower() != "null" else None
        except (ValueError, TypeError):
            total_exp = None

        skills = [str(s).lower().strip() for s in data.get("skills", []) if s]
        edu_degree = str(data.get("education_degree") or "Unknown").strip()
        edu_level = str(data.get("education_level") or "Unknown").strip()
        certs = [str(c).strip() for c in data.get("certifications", []) if c]

        return {
            "candidate_name": candidate_name,
            "total_years_experience": total_exp,
            "skills": skills,
            "education_degree": edu_degree,
            "education_level": edu_level,
            "certifications": certs,
            "extraction_method": "Ollama_LLM"
        }

    except Exception as e:
        # Fallback to regex heuristic parsing if Ollama fails/offline
        return fallback_extract_structured_data(resume_text, str(e))


def extract_experience_from_date_ranges(text: str):
    """Calculates experience from employment date ranges, excluding Education sections."""
    lines = text.split("\n")
    filtered_lines = []
    in_edu = False
    edu_headers = ['education', 'academic background', 'academic qualifications', 'scholastic', 'qualifications']
    exp_headers = ['employment', 'work experience', 'professional experience', 'experience history', 'work history', 'job history']
    edu_keywords = {'b.tech', 'm.tech', 'ph.d', 'phd', 'bachelor', 'master', 'degree', 'university', 'college', 'school', 'cgpa', 'gpa', 'coursework'}
    
    for line in lines:
        l_str = line.lower().strip()
        if any(h in l_str for h in edu_headers):
            in_edu = True
            continue
        elif any(h in l_str for h in exp_headers):
            in_edu = False
            continue
        if in_edu or any(k in l_str for k in edu_keywords):
            continue
        filtered_lines.append(line)
        
    filtered_text = "\n".join(filtered_lines).lower()
    from datetime import datetime
    current_year = datetime.now().year
    current_month = datetime.now().month
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    pattern = r'(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+)?((?:19|20)\d{2})\s*(?:-|to|–|till)\s*(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+)?(present|current|now|till\s*now|(?:19|20)\d{2})'
    matches = re.findall(pattern, filtered_text)
    total_months = 0
    for sm, sy, em, ey in matches:
        s_yr = int(sy)
        s_mo = month_map.get(sm[:3], 1) if sm else 1
        if re.sub(r'\s+', '', ey) in ['present', 'current', 'now', 'tillnow']:
            e_yr, e_mo = current_year, current_month
        else:
            e_yr, e_mo = int(ey), month_map.get(em[:3], 12) if em else 12
        dur = (e_yr - s_yr) * 12 + (e_mo - s_mo)
        if 0 < dur <= 480:
            total_months += dur
    if total_months > 0:
        return round(total_months / 12.0, 1)
    return None


def fallback_extract_structured_data(resume_text: str, error_msg: str) -> dict:
    """Fallback rule-based parser if LLM service is offline."""
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    name = "Unknown Candidate"
    for line in lines[:5]:
        cleaned = re.sub(r'[^\w\s]', '', line).strip()
        if cleaned and not "@" in line and not re.search(r'\d', line) and 2 <= len(cleaned.split()) <= 4:
            name = cleaned.title()
            break

    # Experience heuristic: date ranges (filtered) or literal "X years"
    date_exp = extract_experience_from_date_ranges(resume_text)
    exp_match = re.search(r'(\d+)\+?\s*(?:years|year|yrs|yr)', resume_text.lower())
    literal_exp = float(exp_match.group(1)) if exp_match else None
    total_exp = date_exp if date_exp is not None else literal_exp

    # Certifications heuristic
    certs = []
    cert_patterns = [
        r'(?i)(?:certificat(?:ions?|es?)|licenses?|courses?)\s*:?\s*(.*)',
        r'(?i)(?:certified|aws certified|google certified|microsoft certified|meta certified|coursera|udemy|nptel|cfa|ca)\b.*'
    ]
    for pat in cert_patterns:
        for m in re.finditer(pat, resume_text):
            val = m.group(0).strip()
            if val and len(val) <= 100:
                certs.append(val)

    return {
        "candidate_name": name,
        "total_years_experience": total_exp,
        "skills": [],
        "education_degree": "Unknown",
        "education_level": "Unknown",
        "certifications": list(set(certs)),
        "extraction_method": f"Regex_Fallback (LLM Error: {error_msg})"
    }
