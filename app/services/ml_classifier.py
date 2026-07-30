import joblib
import pandas as pd
import numpy as np
import shap
from app.config import (
    MODELS_DIR,
    FEATURES,
    FEATURE_LABELS,
    CONFIDENCE_THRESHOLD,
    MIN_SKILL_EVIDENCE,
    MIN_RESUME_SKILLS
)

_model = None
_explainer = None

def _load_ml_components():
    global _model, _explainer
    if _model is None:
        model_path = MODELS_DIR / "rf_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train/export the model first.")
        _model = joblib.load(model_path)
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer

def evaluate_candidate_ml(feature_data: dict) -> dict:
    """
    Runs the engineered 10-feature vector through the trained RandomForest model.
    Calculates predicted class, confidence, SHAP feature impacts, and safety flags.
    """
    model, explainer = _load_ml_components()
    row_features = feature_data["features"]

    X_row = pd.DataFrame([row_features])[FEATURES]
    proba = model.predict_proba(X_row)[0]
    pred_class = model.classes_[np.argmax(proba)]
    confidence = float(np.max(proba))

    # Evidence & Safety checks
    resume_skill_cnt = len(feature_data["resume_skills"])
    matched_cnt = row_features["matched_skill_count"]
    missing_cnt = row_features["missing_skill_count"]

    sufficient_evidence = (matched_cnt + missing_cnt >= MIN_SKILL_EVIDENCE) and (resume_skill_cnt >= MIN_RESUME_SKILLS)
    low_confidence = confidence < CONFIDENCE_THRESHOLD
    unverifiable_seniority = bool(row_features["unstated_experience_for_senior_role"])

    needs_review = low_confidence or not sufficient_evidence or unverifiable_seniority

    if unverifiable_seniority:
        review_reason = "unstated candidate experience against a senior-level role"
    elif low_confidence:
        review_reason = "low model confidence"
    elif not sufficient_evidence:
        review_reason = "insufficient skill evidence"
    else:
        review_reason = None

    # SHAP Explanations
    sv = explainer.shap_values(X_row)
    class_idx = list(model.classes_).index(pred_class)
    
    # Check SHAP array structure (multi-class vs single array)
    if isinstance(sv, list):
        contributions = sv[class_idx][0]
    elif len(sv.shape) == 3:
        contributions = sv[0, :, class_idx]
    else:
        contributions = sv[0]

    impact = sorted(zip(FEATURES, contributions), key=lambda t: -abs(t[1]))
    reasons = [{
        "factor": FEATURE_LABELS[feat],
        "value": round(float(X_row[feat].values[0]), 2),
        "direction": "supported" if contrib > 0 else "worked against",
        "impact": round(float(contrib), 3)
    } for feat, contrib in impact[:4]]

    return {
        "prediction": pred_class,
        "confidence": round(confidence, 2),
        "needs_human_review": needs_review,
        "review_reason": review_reason,
        "reasons": reasons
    }
