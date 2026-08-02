import json
import urllib.request
try:
    import ollama
except ImportError:
    ollama = None
from app.config import OLLAMA_MODEL
from app.services.llm_insights import generate_llm_insights


def _is_ollama_online() -> bool:
    if ollama is None:
        return False
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.5)
        return True
    except Exception:
        return False


def suggest_learning_resources(
    missing_skills: list,
    candidate_domain: str = "",
    matched_skills: list = None,
    experience_summary: str = "",
    max_suggestions: int = 5
) -> list:
    if not missing_skills:
        return []

    skills_to_query = missing_skills[:max_suggestions]
    if not _is_ollama_online():
        return [
            {"skill": s, "resource": f"Industry-standard certification or hands-on portfolio project in {s.replace('_', ' ').title()}"}
            for s in skills_to_query
        ]
    matched_str = ", ".join(matched_skills) if matched_skills else "None explicitly listed"
    exp_str = experience_summary.strip() if experience_summary else "No detailed work history provided"

    prompt = f"""
    You are an expert technical career coach evaluating a candidate for a role in {candidate_domain or 'their domain'}.

    CANDIDATE PROFILE & CONTEXT:
    - Matched/Verified Skills: {matched_str}
    - Candidate Work Experience: {exp_str[:800]}

    IDENTIFIED SKILL GAPS TO ADDRESS:
    {", ".join(skills_to_query)}

    INSTRUCTIONS:
    For each missing skill:
    1. Check if the candidate's work history or matched skills already show practical exposure or related experience.
    2. If the candidate ALREADY has practical exposure to this domain, recommend an ADVANCED certification or industry credential (e.g. "Candidate has hands-on experience; recommend Senior/Professional Certification").
    3. If it is a completely new skill gap, recommend a top-rated course, industry certification, or practical portfolio project.

    Return ONLY a raw JSON array of objects with no markdown syntax wrapping matching this exact key format:
    [
      {{"skill": "name_of_skill", "resource": "specific, context-aware recommendation"}}
    ]
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        content = response["message"]["content"]
        data = json.loads(content)
        
        # If wrapped inside a key like {"recommendations": [...]} or {"skills": [...]}
        if isinstance(data, dict):
            for key in ["recommendations", "skills", "results"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

        if isinstance(data, list):
            parsed = []
            for item in data:
                if isinstance(item, dict):
                    skill_name = item.get("skill") or item.get("name") or item.get("skill_name") or "Skill Gap"
                    resource_desc = item.get("resource") or item.get("recommendation") or item.get("course") or "Recommended course or project"
                    parsed.append({"skill": str(skill_name), "resource": str(resource_desc)})
            if parsed:
                return parsed
    except Exception:
        pass

    # Dynamic fallback if LLM service is unreachable
    return [
        {"skill": s, "resource": f"Industry-standard certification or hands-on portfolio project in {s.replace('_', ' ').title()}"}
        for s in skills_to_query
    ]


def assemble_final_report(feature_data: dict, ml_eval: dict) -> dict:
    """
    Assembles final report by combining extracted feature data, ML evaluations,
    dynamic learning resources, and qualitative LLM hiring insights.
    """
    missing_skills = feature_data.get("missing_skills", [])
    matched_skills = feature_data.get("matched_skills", [])
    candidate_domain = feature_data.get("resume_domain", "")
    
    # Construct experience summary string for LLM context
    parsed_resume = feature_data.get("parsed_resume", {})
    jobs = parsed_resume.get("jobs", []) if isinstance(parsed_resume, dict) else []
    exp_summary_parts = []
    for j in jobs:
        comp = j.get("company", "")
        desig = j.get("designation", "")
        desc = " ".join(j.get("description", []))
        exp_summary_parts.append(f"{desig} at {comp}: {desc}")
    exp_summary_str = " | ".join(exp_summary_parts) if exp_summary_parts else f"Total Experience: {feature_data.get('resume_experience', 'Unknown')} yrs"

    recommendations = suggest_learning_resources(
        missing_skills=missing_skills,
        candidate_domain=candidate_domain,
        matched_skills=matched_skills,
        experience_summary=exp_summary_str,
    )

    # Generate qualitative LLM insights
    insights = generate_llm_insights(
        candidate_name=feature_data.get("candidate_name", "Unknown Candidate"),
        prediction=ml_eval.get("prediction", "unknown"),
        confidence=ml_eval.get("confidence", 0.0),
        matched_skills=feature_data.get("matched_skills", []),
        missing_skills=missing_skills,
        resume_experience=feature_data.get("resume_experience"),
        jd_experience=feature_data.get("jd_experience"),
        resume_domain=candidate_domain,
        jd_domain=feature_data.get("jd_domain", ""),
        education_degree=feature_data.get("education_degree", "Unknown"),
        education_level=feature_data.get("education_level", "Unknown"),
        is_overqualified=feature_data.get("is_overqualified", False),
        semantic_similarity=feature_data.get("features", {}).get("semantic_similarity_score", 0.0),
        certifications=feature_data.get("certifications", []),
    )

    report = {}
    report.update(feature_data)
    report.update(ml_eval)
    report["dynamic_recommendations"] = recommendations
    report["llm_insights"] = insights
    return report


def print_candidate_report(report: dict):
    """
    Prints a beautiful, readable terminal report from the candidate report dict.
    """
    print("\n" + "="*70)
    print(f"  CANDIDATE SCREENING REPORT: {report.get('candidate_name', 'Unknown Candidate')}")
    print("="*70)
    pred = report.get('prediction', 'unknown')
    conf = report.get('confidence', 0.0)
    print(f"  PREDICTION   : {pred.upper()}  (Confidence: {conf:.0%})")
    
    engine = report.get('extraction_method') or report.get('extraction_engine') or 'N/A'
    print(f"  ENGINE       : {engine}")
    
    if report.get("needs_human_review"):
        print("\n  [WARNING]  FLAGGED FOR HUMAN REVIEW")
        if report.get("review_reason"):
            print(f"     Reason: {report['review_reason']}")
            
    if report.get("is_overqualified"):
        print("\n  [WARNING]  OVERQUALIFICATION ALERT")
        
    print("\n  " + "-"*50)
    print("  SKILL ANALYSIS")
    print("  " + "-"*50)
    matched = report.get('matched_skills', [])
    missing = report.get('missing_skills', [])
    print(f"  Matched Skills ({len(matched)}): {', '.join(matched) or 'None'}")
    print(f"  Missing Skills ({len(missing)}): {', '.join(missing) or 'None'}")
    
    print("\n  " + "-"*50)
    print("  EDUCATION & CERTIFICATIONS HIGHLIGHTS")
    print("  " + "-"*50)
    edu_deg = report.get('education_degree', 'Unknown')
    edu_lvl = report.get('education_level', 'Unknown')
    print(f"  Education   : {edu_deg} ({edu_lvl})")
    
    certs = report.get('certifications', [])
    if certs:
        print(f"  Certificates ({len(certs)}):")
        for cert in certs:
            print(f"   * {cert}")
    else:
        print("  Certificates: None explicitly detected")

    print("\n  " + "-"*50)
    print("  EXPERIENCE & DOMAIN")
    print("  " + "-"*50)
    r_exp = report.get('resume_experience')
    r_exp_str = f"{r_exp} yrs" if r_exp is not None and not isinstance(r_exp, str) else "Not detected"
    j_exp = report.get('jd_experience')
    j_exp_str = f"{j_exp} yrs" if j_exp is not None and not isinstance(j_exp, str) else "Not specified"
    print(f"  Experience  : {r_exp_str} (Candidate) vs {j_exp_str} (Required)")
    
    r_dom = report.get('resume_domain', 'Unknown')
    j_dom = report.get('jd_domain', 'Unknown')
    match_val = report.get('domain_match')
    if match_val is None:
        match_val = report.get('features', {}).get('domain_match')
    match_str = 'Match' if match_val else 'No Match'
    print(f"  Domain      : {r_dom} (Candidate) vs {j_dom} (JD) -> {match_str}")
    
    sim = report.get('semantic_similarity_score')
    if sim is None:
        sim = report.get('features', {}).get('semantic_similarity_score', 0.0)
    print(f"  Semantic Fit: {sim:.4f}")
    
    reasons = report.get('reasons', [])
    if reasons:
        print("\n  " + "-"*50)
        print("  SHAP FEATURE IMPACT (Why this prediction)")
        print("  " + "-"*50)
        for r in reasons:
            print(f"  * {r['factor']} ({r['value']}) {r['direction']} prediction (impact: {r['impact']:+.3f})")
            
    recs = report.get('dynamic_recommendations', [])
    if recs:
        print("\n  " + "-"*50)
        print("  [RECOMMENDATIONS] DYNAMIC RECOMMENDATIONS FOR MISSING SKILLS")
        print("  " + "-"*50)
        for rec in recs:
            if isinstance(rec, dict):
                print(f"  * {rec.get('skill', '')}: {rec.get('resource', '')}")
    print("="*70 + "\n")
