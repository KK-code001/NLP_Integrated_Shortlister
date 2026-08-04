"""
Tests for the validation layer (utils/validation.py).
No LLM calls — purely rule-based checks.
"""
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.utils.validation import validate_job, validate_and_clean_jobs, normalize_end_date


class TestNormalizeEndDate(unittest.TestCase):

    def test_present_variants_normalized(self):
        for raw in ["Present", "present", "Current", "current", "Now",
                    "Till Date", "till date", "ongoing", "Ongoing", "To Date"]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_end_date(raw), "Present")

    def test_real_dates_unchanged(self):
        for raw in ["Dec 2022", "2021", "06/2020", "January 2023"]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_end_date(raw), raw)

    def test_empty_string(self):
        self.assertEqual(normalize_end_date(""), "")


class TestValidateJob(unittest.TestCase):

    def _good_job(self, **overrides) -> dict:
        base = {
            "company":     "Acme Corp",
            "designation": "Software Engineer",
            "start_date":  "Jan 2020",
            "end_date":    "Dec 2022",
        }
        base.update(overrides)
        return base

    def test_valid_job_passes(self):
        cleaned, warns = validate_job(self._good_job())
        self.assertTrue(cleaned)
        self.assertEqual(len(warns), 0)

    def test_company_is_year_rejected(self):
        cleaned, warns = validate_job(self._good_job(company="2022"))
        self.assertEqual(cleaned, {})
        self.assertEqual(len(warns), 1)
        self.assertIn("year", warns[0])

    def test_empty_company_handled_with_placeholder(self):
        cleaned, warns = validate_job(self._good_job(company=""))
        self.assertEqual(cleaned.get("company"), "Company Not Identified")

    def test_empty_designation_rejected(self):
        cleaned, warns = validate_job(self._good_job(designation=""))
        self.assertEqual(cleaned, {})
        self.assertTrue(any("designation" in w for w in warns))

    def test_inverted_dates_rejected(self):
        cleaned, warns = validate_job(self._good_job(
            start_date="Jan 2022", end_date="Jan 2020"
        ))
        self.assertEqual(cleaned, {})
        self.assertTrue(any("after" in w for w in warns))

    def test_present_end_date_always_valid(self):
        cleaned, warns = validate_job(self._good_job(
            start_date="Jan 2020", end_date="Present"
        ))
        self.assertTrue(cleaned)
        self.assertEqual(len(warns), 0)

    def test_end_date_normalized_to_present(self):
        cleaned, warns = validate_job(self._good_job(end_date="current"))
        self.assertTrue(cleaned)
        self.assertEqual(cleaned["end_date"], "Present")


class TestValidateAndCleanJobs(unittest.TestCase):

    def test_removes_duplicates(self):
        jobs = [
            {"company": "Acme", "designation": "Dev", "start_date": "Jan 2020", "end_date": "Dec 2021"},
            {"company": "acme", "designation": "Dev", "start_date": "Jan 2020", "end_date": "Dec 2021"},
        ]
        valid, warns = validate_and_clean_jobs(jobs)
        self.assertEqual(len(valid), 1)
        self.assertTrue(any("Duplicate" in w for w in warns))

    def test_filters_bad_records(self):
        jobs = [
            {"company": "2020",  "designation": "Dev",  "start_date": "Jan 2020", "end_date": "Dec 2020"},
            {"company": "Google","designation": "SWE",  "start_date": "Jan 2021", "end_date": "Present"},
        ]
        valid, warns = validate_and_clean_jobs(jobs)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["company"], "Google")

    def test_empty_input(self):
        valid, warns = validate_and_clean_jobs([])
        self.assertEqual(valid, [])
        self.assertEqual(warns, [])

    def test_all_valid_jobs_preserved(self):
        jobs = [
            {"company": "A", "designation": "D1", "start_date": "Jan 2018", "end_date": "Jan 2020"},
            {"company": "B", "designation": "D2", "start_date": "Feb 2020", "end_date": "Present"},
        ]
        valid, warns = validate_and_clean_jobs(jobs)
        self.assertEqual(len(valid), 2)
        self.assertEqual(warns, [])


if __name__ == "__main__":
    unittest.main()
