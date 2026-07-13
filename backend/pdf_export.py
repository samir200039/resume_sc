"""
Turns plain resume text (headers, bullets, paragraphs) into a clean,
properly formatted PDF using reportlab.

Parsing heuristics on the input text:
- Blank line            -> spacing
- ALL CAPS short line   -> section header (e.g. "EXPERIENCE")
- Starts with "- "/"* " -> bullet point
- First non-empty line  -> name/title, styled larger
- Anything else         -> normal paragraph
"""

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

NAME_STYLE = ParagraphStyle(
    "Name",
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    spaceAfter=4,
)

SECTION_STYLE = ParagraphStyle(
    "Section",
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    spaceBefore=14,
    spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"),
    letterSpacing=0.5,
)

BULLET_STYLE = ParagraphStyle(
    "Bullet",
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    leftIndent=14,
    spaceAfter=3,
    bulletIndent=0,
)

BODY_STYLE = ParagraphStyle(
    "Body",
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    spaceAfter=4,
)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _looks_like_section_header(line: str) -> bool:
    stripped = line.strip().rstrip(":")
    if not stripped:
        return False
    return stripped.isupper() and len(stripped) <= 40


def build_resume_pdf(resume_text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
    )

    story = []
    lines = [line.rstrip() for line in resume_text.strip().split("\n")]
    first_content_used = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 6))
            continue

        if not first_content_used:
            story.append(Paragraph(_escape(stripped), NAME_STYLE))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
            story.append(Spacer(1, 8))
            first_content_used = True
            continue

        if _looks_like_section_header(stripped):
            story.append(Paragraph(_escape(stripped.upper()), SECTION_STYLE))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
            continue

        if stripped.startswith(("- ", "* ")):
            bullet_text = stripped[2:].strip()
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{_escape(bullet_text)}", BULLET_STYLE))
            continue

        story.append(Paragraph(_escape(stripped), BODY_STYLE))

    doc.build(story)
    return buffer.getvalue()