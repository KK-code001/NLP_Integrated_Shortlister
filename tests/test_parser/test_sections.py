"""
Tests for section detection logic.
Uses synthetic TextBlock inputs — no PDF parsing needed.
"""
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.parser.layout import TextBlock, ResumeDocument
from app.parser.sections import detect_sections, section_to_text, _resolve_section, _is_heading


class TestIsHeading(unittest.TestCase):

    def test_docling_heading_type(self):
        block = TextBlock(text="Work Experience", block_type="heading")
        self.assertTrue(_is_heading(block))

    def test_alias_match_exact(self):
        block = TextBlock(text="Work Experience", block_type="paragraph")
        self.assertTrue(_is_heading(block))

    def test_alias_match_case_insensitive(self):
        block = TextBlock(text="PROFESSIONAL EXPERIENCE", block_type="paragraph")
        self.assertTrue(_is_heading(block))

    def test_alias_match_skills_variants(self):
        for alias in ["Skills", "Technical Skills", "Core Competencies", "Tech Stack"]:
            with self.subTest(alias=alias):
                block = TextBlock(text=alias, block_type="paragraph")
                self.assertTrue(_is_heading(block))

    def test_not_heading_long_sentence(self):
        # "experience" inside a full sentence should NOT be a heading
        block = TextBlock(
            text="I have over 5 years of experience in software development.",
            block_type="paragraph",
        )
        self.assertFalse(_is_heading(block))

    def test_all_caps_short(self):
        block = TextBlock(text="EDUCATION", block_type="text")
        self.assertTrue(_is_heading(block))

    def test_bold_short(self):
        block = TextBlock(text="My Projects", block_type="text", is_bold=True)
        self.assertTrue(_is_heading(block))

    def test_bold_long_not_heading(self):
        block = TextBlock(
            text="I am a passionate developer with expertise in Python and ML.",
            block_type="text",
            is_bold=True,
        )
        self.assertFalse(_is_heading(block))

    def test_empty_block_not_heading(self):
        block = TextBlock(text="", block_type="heading")
        self.assertFalse(_is_heading(block))


class TestResolveSection(unittest.TestCase):

    def test_experience_aliases(self):
        for alias in ["experience", "work experience", "employment history",
                      "career history", "professional experience"]:
            with self.subTest(alias=alias):
                self.assertEqual(_resolve_section(alias), "experience")

    def test_education_aliases(self):
        for alias in ["education", "academic background", "qualifications"]:
            with self.subTest(alias=alias):
                self.assertEqual(_resolve_section(alias), "education")

    def test_skills_aliases(self):
        for alias in ["skills", "technical skills", "core competencies",
                      "technologies", "tech stack"]:
            with self.subTest(alias=alias):
                self.assertEqual(_resolve_section(alias), "skills")

    def test_unknown_heading_returns_none(self):
        self.assertIsNone(_resolve_section("My Hobbies section that doesn't match"))

    def test_certifications_aliases(self):
        for alias in ["certifications", "certificates", "licenses", "courses"]:
            with self.subTest(alias=alias):
                self.assertEqual(_resolve_section(alias), "certifications")


class TestDetectSections(unittest.TestCase):

    def _make_doc(self, blocks: list[TextBlock]) -> ResumeDocument:
        raw_text = "\n".join(b.text for b in blocks)
        return ResumeDocument(blocks=blocks, raw_text=raw_text)

    def test_basic_section_split(self):
        blocks = [
            TextBlock("John Doe",           "paragraph"),
            TextBlock("john@example.com",   "paragraph"),
            TextBlock("Experience",          "heading"),
            TextBlock("Software Engineer at Acme, 2020-2022", "paragraph"),
            TextBlock("Education",           "heading"),
            TextBlock("B.Tech Computer Science, 2016-2020",   "paragraph"),
        ]
        doc = self._make_doc(blocks)
        sections = detect_sections(doc)

        self.assertIn("header", sections)
        self.assertIn("experience", sections)
        self.assertIn("education", sections)
        # Header should contain the name/email lines
        self.assertEqual(len(sections["header"]), 2)
        # Experience should contain the job line
        self.assertEqual(len(sections["experience"]), 1)

    def test_experience_in_summary_not_split(self):
        """'experience' inside a long summary paragraph must NOT start a new section."""
        blocks = [
            TextBlock("Jane Smith",                                        "paragraph"),
            TextBlock("Summary",                                           "heading"),
            TextBlock(
                "I have 3 years of experience in full-stack development "
                "and enjoy building scalable systems.",
                "paragraph",
            ),
            TextBlock("Skills",                                            "heading"),
            TextBlock("Python, React, Docker",                             "paragraph"),
        ]
        doc = self._make_doc(blocks)
        sections = detect_sections(doc)

        # The summary sentence should be under "summary", not create an "experience" section
        self.assertIn("summary", sections)
        self.assertNotIn("experience", sections)

    def test_two_column_simulation(self):
        """All blocks are passed in — section detection should still work."""
        blocks = [
            TextBlock("Alice Wang",       "paragraph"),
            TextBlock("Work Experience",  "heading"),
            TextBlock("ML Engineer 2021-Present",   "paragraph"),
            TextBlock("Skills",           "heading"),
            TextBlock("Python TensorFlow",           "paragraph"),
        ]
        doc = self._make_doc(blocks)
        sections = detect_sections(doc)

        self.assertIn("experience", sections)
        self.assertIn("skills", sections)

    def test_no_sections_all_in_header(self):
        """If no section headers found, everything goes into 'header'."""
        blocks = [
            TextBlock("Alice Wang",      "paragraph"),
            TextBlock("Python, AWS",     "paragraph"),
            TextBlock("BSc Computer Science", "paragraph"),
        ]
        doc = self._make_doc(blocks)
        sections = detect_sections(doc)

        self.assertIn("header", sections)
        self.assertEqual(list(sections.keys()), ["header"])


class TestSectionToText(unittest.TestCase):

    def test_basic_join(self):
        blocks = [
            TextBlock("Line one", "paragraph"),
            TextBlock("",         "paragraph"),  # empty — should be skipped
            TextBlock("Line two", "list_item"),
        ]
        text = section_to_text(blocks)
        self.assertIn("Line one", text)
        self.assertIn("Line two", text)
        self.assertNotIn("\n\n", text)   # no double blank lines

    def test_empty_list(self):
        self.assertEqual(section_to_text([]), "")


if __name__ == "__main__":
    unittest.main()
