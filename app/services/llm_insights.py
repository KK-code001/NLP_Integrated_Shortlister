"""
LLM Insight Engine — qualitative hiring intelligence via Ollama.

This module generates human-readable insights that go beyond numeric scores:
  - Fit Summary:              2-3 sentence overall assessment
  - Key Strengths:            Top strengths relevant to the JD
  - Gaps & Risks:             Critical gaps with severity
  - Career Trajectory:        Growth potential assessment

All insights gracefully degrade to template-based output when Ollama is offline.
"""
from __future__ import annotations

import json
import logging
import ollama
from app.config import OLLAMA_MODEL

logger = logging.getLogger(__name__)


def generate_llm_insights(
    candidate_name: str,
    prediction: str,
    confidence: float,
    matched_skills: list,
    missing_skills: list,
    resume_experience: float | None,
    jd_experience: float | None,
    resume_domain: str,
    jd_domain: str,
    education_degree: str,
    education_level: str,
    is_overqualified: bool,
    semantic_similarity: float,
    certifications: list | None = None,
) -> dict:
    """
    Generate qualitative hiring insights by asking the LLM to reason
    about the candidate's profile against the job requirements.

    Returns a dict with keys:
      fit_summary, strengths, gaps_and_risks, career_trajectory, source
    """
    context = _build_context(
        candidate_name=candidate_name,
        prediction=prediction,
        confidence=confidence,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        resume_experience=resume_experience,
        jd_experience=jd_experience,
        resume_domain=resume_domain,
        jd_domain=jd_domain,
        education_degree=education_degree,
        education_level=education_level,
        is_overqualified=is_overqualified,
        semantic_similarity=semantic_similarity,
        certifications=certifications,
    )

    try:
        return _generate_via_llm(context)
    except Exception as exc:
        logger.warning("[LLM Insights] Ollama call failed (%s) — using template fallback", exc)
        return _generate_fallback(context)


# ---------------------------------------------------------------------------
# LLM-powered generation
# ---------------------------------------------------------------------------

_INSIGHT_SCHEMA = """{
  "fit_summary": "A 2-3 sentence assessment of the candidate's overall fit for this role. Be specific about WHY they fit or don't fit.",
  "strengths": ["strength 1 relevant to JD", "strength 2", "strength 3"],
  "gaps_and_risks": [
    {"gap": "description of the gap", "severity": "HIGH | MEDIUM | LOW", "mitigation": "how the candidate could bridge this gap"}
  ],
  "career_trajectory": "1-2 sentence assessment of the candidate's growth potential and trajectory alignment with this role."
}"""


def _generate_via_llm(context: dict) -> dict:
    """Call Ollama to generate qualitative insights."""
    prompt = f"""You are a senior technical recruiter with 15+ years of experience.
Analyze this candidate screening data and provide actionable hiring insights.

CANDIDATE PROFILE:
- Name: {context['candidate_name']}
- ML Model Decision: {context['prediction']} (Confidence: {context['confidence']:.0%})
- Matched Skills: {', '.join(context['matched_skills']) or 'None'}
- Missing Skills: {', '.join(context['missing_skills']) or 'None'}
- Candidate Experience: {context['exp_str']}
- JD Required Experience: {context['jd_exp_str']}
- Candidate Domain: {context['resume_domain']}
- JD Domain: {context['jd_domain']}
- Education: {context['education_degree']} ({context['education_level']})
- Certifications: {', '.join(context['certifications']) or 'None listed'}
- Semantic Fit Score: {context['semantic_similarity']:.2f}
- Overqualified: {'Yes' if context['is_overqualified'] else 'No'}

STRICT RULES:
- Return ONLY a raw JSON object. No markdown. No preamble.
- Be specific and actionable — not generic.
- Reference actual skills and experience from the data above.
- If the candidate is a poor fit, say so directly with specific reasons.
- Each strength must reference a concrete matched skill or qualification.

EXACT OUTPUT SCHEMA:
{_INSIGHT_SCHEMA}
"""

    logger.info("[LLM Insights] Generating qualitative insights via Ollama...")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    raw = response["message"]["content"]
    data = json.loads(raw)

    # Normalize and validate the response
    result = {
        "fit_summary": str(data.get("fit_summary", "")).strip(),
        "strengths": _safe_list(data.get("strengths", [])),
        "gaps_and_risks": _safe_gaps(data.get("gaps_and_risks", [])),
        "career_trajectory": str(data.get("career_trajectory", "")).strip(),
        "source": "ollama_llm",
    }

    logger.info("[LLM Insights] ✓ LLM insights generated successfully")
    return result


# ---------------------------------------------------------------------------
# Template-based fallback (when Ollama is offline)
# ---------------------------------------------------------------------------

def _generate_fallback(context: dict) -> dict:
    """Generate template-based insights when Ollama is unavailable."""

    # Fit summary
    matched_count = len(context['matched_skills'])
    missing_count = len(context['missing_skills'])
    total_skills = matched_count + missing_count

    if context['prediction'].lower() in ('match', 'yes', '1'):
        if context['confidence'] >= 0.75:
            fit = (
                f"{context['candidate_name']} is a strong match for this role, "
                f"with {matched_count}/{total_skills} required skills matched and "
                f"a semantic similarity of {context['semantic_similarity']:.0%}. "
                f"The candidate's {context['resume_domain']} background aligns well with the role."
            )
        else:
            fit = (
                f"{context['candidate_name']} is a tentative match "
                f"({matched_count}/{total_skills} skills matched, "
                f"confidence: {context['confidence']:.0%}). "
                "Human review recommended to verify fit."
            )
    else:
        fit = (
            f"{context['candidate_name']} does not appear to be a strong fit. "
            f"Only {matched_count}/{total_skills} required skills matched, "
            f"and the semantic similarity score is {context['semantic_similarity']:.0%}."
        )

    # Strengths
    strengths = []
    if context['matched_skills']:
        top_skills = context['matched_skills'][:5]
        strengths.append(f"Has {matched_count} of {total_skills} required skills: {', '.join(s.title() for s in top_skills)}")
    if context['resume_domain'] == context['jd_domain'] and context['resume_domain'] != 'Unknown':
        strengths.append(f"Domain alignment: candidate's {context['resume_domain']} background matches JD domain")
    if context['exp_val'] is not None and context['jd_exp_val'] is not None and context['exp_val'] >= context['jd_exp_val']:
        strengths.append(f"Meets experience requirement ({context['exp_str']} vs. {context['jd_exp_str']} required)")
    if context['certifications']:
        strengths.append(f"Holds relevant certifications: {', '.join(context['certifications'][:3])}")
    if context['semantic_similarity'] >= 0.6:
        strengths.append(f"High semantic alignment with JD ({context['semantic_similarity']:.0%})")
    if not strengths:
        strengths.append("No standout strengths identified from available data")

    # Gaps & Risks
    gaps = []
    if context['missing_skills']:
        for skill in context['missing_skills'][:3]:
            severity = "HIGH" if matched_count < total_skills * 0.5 else "MEDIUM"
            gaps.append({
                "gap": f"Missing required skill: {skill.title()}",
                "severity": severity,
                "mitigation": f"Could upskill via targeted training or certification in {skill.title()}",
            })
    if context['resume_domain'] != context['jd_domain'] and context['resume_domain'] != 'Unknown' and context['jd_domain'] != 'Unknown':
        gaps.append({
            "gap": f"Domain mismatch: candidate ({context['resume_domain']}) vs. role ({context['jd_domain']})",
            "severity": "HIGH",
            "mitigation": "Evaluate transferable skills and willingness to transition domains",
        })
    if context['is_overqualified']:
        gaps.append({
            "gap": "Potential overqualification — candidate may find the role under-challenging",
            "severity": "MEDIUM",
            "mitigation": "Discuss growth trajectory and role evolution",
        })
    if context['exp_val'] is not None and context['jd_exp_val'] is not None and context['exp_val'] < context['jd_exp_val']:
        gaps.append({
            "gap": f"Experience gap: {context['exp_str']} vs. {context['jd_exp_str']} required",
            "severity": "MEDIUM",
            "mitigation": "Assess depth of experience and relevant project complexity",
        })
    if not gaps:
        gaps.append({"gap": "No significant gaps identified", "severity": "LOW", "mitigation": "N/A"})

    # Career trajectory
    if context['is_overqualified']:
        trajectory = "The candidate appears overqualified — explore whether the role offers sufficient growth to retain them long-term."
    elif context['exp_val'] is not None and context['exp_val'] < 2.0:
        trajectory = "Early-career candidate with growth potential. Evaluate learning agility and mentorship needs."
    elif context['exp_val'] is not None and context['exp_val'] >= 8.0:
        trajectory = "Senior professional with deep experience. Assess leadership capabilities and strategic thinking."
    else:
        trajectory = "Mid-career professional — evaluate trajectory alignment with the team's growth plans and role evolution."

    return {
        "fit_summary": fit,
        "strengths": strengths,
        "gaps_and_risks": gaps,
        "career_trajectory": trajectory,
        "source": "template_fallback",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(**kwargs) -> dict:
    """Build a standardized context dict for both LLM and fallback paths."""
    resume_exp = kwargs.get("resume_experience")
    jd_exp = kwargs.get("jd_experience")

    try:
        exp_val = float(resume_exp) if resume_exp is not None else None
    except (TypeError, ValueError):
        exp_val = None

    try:
        jd_exp_val = float(jd_exp) if jd_exp is not None else None
    except (TypeError, ValueError):
        jd_exp_val = None

    import pandas as pd
    if jd_exp_val is not None and pd.isna(jd_exp_val):
        jd_exp_val = None

    exp_str = f"{exp_val:.1f} years" if exp_val is not None else "Not stated"
    jd_exp_str = f"{jd_exp_val:.1f} years" if jd_exp_val is not None else "Not specified"

    return {
        "candidate_name": kwargs.get("candidate_name", "Unknown"),
        "prediction": str(kwargs.get("prediction", "unknown")),
        "confidence": float(kwargs.get("confidence", 0.0)),
        "matched_skills": kwargs.get("matched_skills", []),
        "missing_skills": kwargs.get("missing_skills", []),
        "resume_domain": kwargs.get("resume_domain", "Unknown"),
        "jd_domain": kwargs.get("jd_domain", "Unknown"),
        "education_degree": kwargs.get("education_degree", "Unknown"),
        "education_level": kwargs.get("education_level", "Unknown"),
        "is_overqualified": bool(kwargs.get("is_overqualified", False)),
        "semantic_similarity": float(kwargs.get("semantic_similarity", 0.0)),
        "certifications": kwargs.get("certifications") or [],
        "exp_val": exp_val,
        "jd_exp_val": jd_exp_val,
        "exp_str": exp_str,
        "jd_exp_str": jd_exp_str,
    }


def _safe_list(val) -> list:
    if isinstance(val, list):
        return [str(v).strip() for v in val if v]
    return []


def _safe_gaps(val) -> list:
    if not isinstance(val, list):
        return []
    cleaned = []
    for item in val:
        if isinstance(item, dict):
            cleaned.append({
                "gap": str(item.get("gap", "")).strip(),
                "severity": str(item.get("severity", "MEDIUM")).strip().upper(),
                "mitigation": str(item.get("mitigation", "")).strip(),
            })
        elif isinstance(item, str):
            cleaned.append({"gap": item.strip(), "severity": "MEDIUM", "mitigation": ""})
    return cleaned

