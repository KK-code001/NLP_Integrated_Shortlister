"""
Tests for schema normalization (utils/schema.py) and education normalization
(parser/education.py).
No LLM calls.
"""
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.utils.schema import validate_schema
from app.parser.education import normalize_degree, best_education_record


class TestValidateSchema(unittest.TestCase):

    def test_empty_input_returns_defaults(self):
        schema = validate_schema({})
        self.assertEqual(schema["name"], "")
        self.assertEqual(schema["skills"], [])
        self.assertIsNone(schema["experience"]["total_years"])
        self.assertEqual(schema["experience"]["jobs"], [])
        self.assertIsInstance(schema["parsing_metadata"]["warnings"], list)

    def test_valid_data_preserved(self):
        data = {
            "name":  "Alice Wang",
            "email": "alice@example.com",
            "phone": "+91 9876543210",
            "experience": {"total_years": 3.5, "jobs": [{"company": "Acme"}]},
            "skills": ["python", "aws"],
            "education": [{"degree": "B.Tech"}],
            "certifications": ["AWS SAA"],
        }
        schema = validate_schema(data)
        self.assertEqual(schema["name"], "Alice Wang")
        self.assertEqual(schema["experience"]["total_years"], 3.5)
        self.assertEqual(len(schema["experience"]["jobs"]), 1)
        self.assertIn("python", schema["skills"])
        self.assertIn("AWS SAA", schema["certifications"])

    def test_whitespace_stripped_from_strings(self):
        schema = validate_schema({"name": "  Bob Jones  "})
        self.assertEqual(schema["name"], "Bob Jones")

    def test_non_list_fields_ignored(self):
        # If a list field is accidentally a string, don't overwrite default
        schema = validate_schema({"skills": "python, aws"})
        self.assertEqual(schema["skills"], [])

    def test_metadata_merged(self):
        data = {
            "parsing_metadata": {
                "parser_used": "docling",
                "extraction_method": "Hybrid_Docling_Ollama",
                "warnings": ["w1"],
            }
        }
        schema = validate_schema(data)
        self.assertEqual(schema["parsing_metadata"]["parser_used"], "docling")
        self.assertIn("w1", schema["parsing_metadata"]["warnings"])


class TestNormalizeDegree(unittest.TestCase):

    def test_btech_is_bachelor(self):
        name, level = normalize_degree("B.Tech in Computer Science")
        self.assertEqual(level, "Bachelor")
        self.assertIn("B.Tech", name)

    def test_mba_is_master(self):
        name, level = normalize_degree("MBA")
        self.assertEqual(level, "Master")

    def test_phd_is_doctorate(self):
        name, level = normalize_degree("PhD in Machine Learning")
        self.assertEqual(level, "Doctorate")

    def test_diploma_is_diploma(self):
        name, level = normalize_degree("Diploma in Electrical Engineering")
        self.assertEqual(level, "Diploma")

    def test_mtech_is_master(self):
        name, level = normalize_degree("M.Tech Computer Science")
        self.assertEqual(level, "Master")

    def test_bsc_is_bachelor(self):
        name, level = normalize_degree("B.Sc Physics")
        self.assertEqual(level, "Bachelor")

    def test_unknown_degree(self):
        name, level = normalize_degree("Some Random Qualification")
        self.assertEqual(level, "Unknown")

    def test_empty_string(self):
        name, level = normalize_degree("")
        self.assertEqual(name, "Unknown")
        self.assertEqual(level, "Unknown")


class TestBestEducationRecord(unittest.TestCase):

    def test_picks_highest_qualification(self):
        education = [
            {"degree": "B.Tech"},
            {"degree": "M.Tech"},
            {"degree": "Diploma"},
        ]
        name, level = best_education_record(education)
        self.assertEqual(level, "Master")

    def test_single_record(self):
        education = [{"degree": "MBA"}]
        name, level = best_education_record(education)
        self.assertEqual(level, "Master")

    def test_empty_list(self):
        name, level = best_education_record([])
        self.assertEqual(name, "Unknown")
        self.assertEqual(level, "Unknown")

    def test_doctorate_wins(self):
        education = [
            {"degree": "B.Sc"},
            {"degree": "M.Sc"},
            {"degree": "PhD"},
        ]
        name, level = best_education_record(education)
        self.assertEqual(level, "Doctorate")


if __name__ == "__main__":
    unittest.main()
