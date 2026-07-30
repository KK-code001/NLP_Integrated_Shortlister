import unittest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.services.skill_matcher import clean_text, extract_skills_exact, get_matched_skills, get_missing_skills
from app.services.feature_builder import extract_jd_experience, build_feature_vector

class TestResumePipeline(unittest.TestCase):

    def test_clean_text_symbols(self):
        text = "Experienced in C++, C#, Node.js, and FP&A."
        cleaned = clean_text(text)
        self.assertIn("cplusplus", cleaned)
        self.assertIn("csharp", cleaned)
        self.assertIn("nodejs", cleaned)
        self.assertIn("fpna", cleaned)

    def test_skill_extraction(self):
        text = "Proficient in Python, React, AWS, Docker, and PostgreSQL."
        skills = extract_skills_exact(text)
        self.assertIn("python", skills)
        self.assertIn("react", skills)
        self.assertIn("aws", skills)
        self.assertIn("javascript", skills)  # Implied from React

    def test_jd_experience_extraction(self):
        jd1 = "Requirements: Must have 5+ years of experience in Software Development."
        jd2 = "Experience: 3 years required."
        self.assertEqual(extract_jd_experience(jd1), 5.0)
        self.assertEqual(extract_jd_experience(jd2), 3.0)

    def test_feature_vector_building(self):
        resume_raw = "John Doe. Experienced Python & React Developer with 4 years working with AWS."
        jd_raw = "Job Title: Software Developer. Requirements: 3+ years experience in Python, React, and Docker."
        llm_data = {
            "candidate_name": "John Doe",
            "total_years_experience": 4.0,
            "skills": ["python", "react", "aws", "git"],
            "education_degree": "B.Tech",
            "education_level": "Bachelor"
        }
        res = build_feature_vector(resume_raw, jd_raw, llm_data)
        feats = res["features"]
        self.assertEqual(res["candidate_name"], "John Doe")
        self.assertEqual(feats["resume_experience_mentioned"], 1)
        self.assertEqual(feats["experience_match_score"], 1.0)

if __name__ == "__main__":
    unittest.main()
