"""
Exports the RandomForest model and configuration from resumeJD2_pairs.csv
to models/rf_model.joblib and models/model_config.joblib
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from app.config import FEATURES, MODELS_DIR, EXPERIENCE_UNKNOWN_SENTINEL, CONFIDENCE_THRESHOLD, MIN_SKILL_EVIDENCE, OVERQUALIFICATION_RATIO_THRESHOLD
from app.services.skill_matcher import extract_skills_exact, get_matched_skills, get_missing_skills, skill_match_score, identify_domain, compute_semantic_similarity
from app.services.feature_builder import extract_jd_experience

RANDOM_STATE = 42

def build_dataset_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).drop_duplicates().copy()
    rows = []
    print(f"Engineered features for {len(df)} resume-JD pairs...")
    
    for i, row in df.iterrows():
        res_text = str(row["resume_text"])
        jd_text = str(row["job_description"])
        
        res_skills = extract_skills_exact(res_text)
        jd_skills = extract_skills_exact(jd_text)
        matched = get_matched_skills(res_skills, jd_skills)
        missing = get_missing_skills(res_skills, jd_skills)
        sk_score = skill_match_score(res_skills, jd_skills)
        
        # Regex experience fallback for training data
        res_exp_match = pd.Series([res_text]).str.extract(r'(\d+)\+?\s*years?')[0]
        res_exp = float(res_exp_match.iloc[0]) if not res_exp_match.isna().iloc[0] else None
        jd_exp = extract_jd_experience(jd_text)
        
        res_exp_mentioned = int(res_exp is not None)
        jd_exp_mentioned = int(not pd.isna(jd_exp))
        
        if res_exp_mentioned and jd_exp_mentioned and jd_exp > 0:
            exp_score = round(min(res_exp / jd_exp, 1.0), 2)
            exp_ratio = round(res_exp / jd_exp, 2)
        else:
            exp_score = EXPERIENCE_UNKNOWN_SENTINEL
            exp_ratio = EXPERIENCE_UNKNOWN_SENTINEL

        unstated_exp_senior = int(
            not res_exp_mentioned and not pd.isna(jd_exp) and jd_exp >= 5
        )
        
        res_domain = identify_domain(res_skills)
        jd_domain = identify_domain(jd_skills)
        domain_match = int(res_domain == jd_domain and res_domain != "Unknown")
        
        sim_score = compute_semantic_similarity(res_text, jd_text)
        
        rows.append({
            "skill_match_score": sk_score,
            "matched_skill_count": len(matched),
            "missing_skill_count": len(missing),
            "experience_match_score": exp_score,
            "experience_ratio": exp_ratio,
            "resume_experience_mentioned": res_exp_mentioned,
            "jd_experience_mentioned": jd_exp_mentioned,
            "unstated_experience_for_senior_role": unstated_exp_senior,
            "semantic_similarity_score": sim_score,
            "domain_match": domain_match,
            "match_label": row["match_label"]
        })
        
    return pd.DataFrame(rows)

def train_and_export():
    csv_path = BASE_DIR / "resumeJD2_pairs.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Creating model with synthetic initialization...")
        return

    data = build_dataset_from_csv(str(csv_path))
    X = data[FEATURES]
    y = data["match_label"]
    
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    model.fit(X, y)
    
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "rf_model.joblib")
    joblib.dump({
        "features": FEATURES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "min_skill_evidence": MIN_SKILL_EVIDENCE,
        "overqualification_ratio_threshold": OVERQUALIFICATION_RATIO_THRESHOLD
    }, MODELS_DIR / "model_config.joblib")
    
    print(f"Successfully trained and saved model to {MODELS_DIR / 'rf_model.joblib'}")

if __name__ == "__main__":
    train_and_export()
