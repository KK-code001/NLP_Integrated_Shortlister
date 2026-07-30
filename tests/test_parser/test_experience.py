"""
Tests for experience date parsing and calculation.
These tests exercise the deterministic Python pipeline — no LLM calls.
"""
import sys
import unittest
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.utils.dates import parse_date_flexible, months_between, is_present
from app.parser.experience import calculate_total_experience, separate_internships


class TestIsPresent(unittest.TestCase):

    def test_present_variants(self):
        for token in ["Present", "present", "Current", "current", "Now", "now",
                      "Till Date", "till date", "Ongoing", "ongoing", "To Date"]:
            with self.subTest(token=token):
                self.assertTrue(is_present(token), f"Expected '{token}' to be present")

    def test_non_present_strings(self):
        for token in ["2022", "Jan 2022", "December 2021", "2020-06", ""]:
            with self.subTest(token=token):
                self.assertFalse(is_present(token), f"Expected '{token}' to NOT be present")


class TestParseDateFlexible(unittest.TestCase):

    def test_present_returns_now(self):
        result = parse_date_flexible("Present")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, datetime.now().year)

    def test_month_year_short(self):
        result = parse_date_flexible("Jan 2020")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2020)
        self.assertEqual(result.month, 1)

    def test_month_year_full(self):
        result = parse_date_flexible("March 2019")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2019)
        self.assertEqual(result.month, 3)

    def test_year_only(self):
        result = parse_date_flexible("2021")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2021)
        self.assertEqual(result.month, 1)

    def test_mm_yyyy(self):
        result = parse_date_flexible("06/2021")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2021)
        self.assertEqual(result.month, 6)

    def test_yyyy_mm(self):
        result = parse_date_flexible("2021-06")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2021)
        self.assertEqual(result.month, 6)

    def test_empty_string(self):
        self.assertIsNone(parse_date_flexible(""))

    def test_garbage(self):
        self.assertIsNone(parse_date_flexible("not a date"))


class TestMonthsBetween(unittest.TestCase):

    def test_exact_one_year(self):
        d1 = datetime(2020, 1, 1)
        d2 = datetime(2021, 1, 1)
        self.assertEqual(months_between(d1, d2), 12)

    def test_six_months(self):
        d1 = datetime(2020, 1, 1)
        d2 = datetime(2020, 7, 1)
        self.assertEqual(months_between(d1, d2), 6)

    def test_reverse_order(self):
        # Should return positive regardless of order
        d1 = datetime(2021, 1, 1)
        d2 = datetime(2020, 1, 1)
        self.assertEqual(months_between(d1, d2), 12)


class TestCalculateTotalExperience(unittest.TestCase):

    def test_single_job_two_years(self):
        jobs = [{"start_date": "Jan 2020", "end_date": "Jan 2022"}]
        result = calculate_total_experience(jobs)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 2.0, places=0)

    def test_two_non_overlapping_jobs(self):
        jobs = [
            {"start_date": "Jan 2018", "end_date": "Jan 2020"},
            {"start_date": "Mar 2020", "end_date": "Mar 2022"},
        ]
        result = calculate_total_experience(jobs)
        self.assertIsNotNone(result)
        # ~4 years
        self.assertGreater(result, 3.5)
        self.assertLess(result, 4.5)

    def test_overlapping_jobs_not_double_counted(self):
        # Two jobs that overlap completely — should count as one period
        jobs = [
            {"start_date": "Jan 2020", "end_date": "Dec 2022"},
            {"start_date": "Jun 2020", "end_date": "Jun 2021"},   # completely within above
        ]
        result = calculate_total_experience(jobs)
        self.assertIsNotNone(result)
        # Should be ~3 years, not ~4.5 years
        self.assertLess(result, 3.5)

    def test_present_end_date(self):
        current_year = datetime.now().year
        jobs = [{"start_date": f"Jan {current_year - 2}", "end_date": "Present"}]
        result = calculate_total_experience(jobs)
        self.assertIsNotNone(result)
        self.assertGreater(result, 1.5)

    def test_no_valid_jobs(self):
        jobs = [{"start_date": "", "end_date": ""}]
        result = calculate_total_experience(jobs)
        self.assertIsNone(result)

    def test_empty_list(self):
        result = calculate_total_experience([])
        self.assertIsNone(result)

    def test_inverted_dates_skipped(self):
        # end_date before start_date — should be skipped
        jobs = [{"start_date": "Jan 2022", "end_date": "Jan 2020"}]
        result = calculate_total_experience(jobs)
        self.assertIsNone(result)


class TestSeparateInternships(unittest.TestCase):

    def test_separates_correctly(self):
        jobs = [
            {"company": "A", "employment_type": "Full-time"},
            {"company": "B", "employment_type": "Internship"},
            {"company": "C", "employment_type": ""},
        ]
        full_time, internships = separate_internships(jobs)
        self.assertEqual(len(full_time), 2)
        self.assertEqual(len(internships), 1)
        self.assertEqual(internships[0]["company"], "B")


if __name__ == "__main__":
    unittest.main()
