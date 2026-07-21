"""Section-aware chunking for SEC filings.

Naive fixed-size splitting throws away the one piece of structure that makes
filings navigable: Item numbers (Item 1A - Risk Factors, Item 7 - MD&A, etc).
This module first splits a filing into its Item sections, then sub-splits any
section that's too long for a single chunk, using a token-window with overlap
so no fact is stranded at a chunk boundary.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# SEC filing documents often open with an XML declaration (inline XBRL) even though the
# body is HTML-like and needs BeautifulSoup's tolerant HTML parser, not a strict XML one —
# this warning is a false positive for our use case.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Recognizes "Item 1A.", "ITEM 7.", "Item 7A", etc. at the start of a heading-ish line.
ITEM_HEADER_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[a-c]?)\.?\s*[-–—]?\s*(.*)$", re.IGNORECASE
)

# Recognizes "PART I" / "PART II" at the start of a heading-ish line.
PART_HEADER_RE = re.compile(r"^\s*part\s+(i{1,2})\b", re.IGNORECASE)

# Canonical item titles for 10-K filings — item numbers are unique across the whole
# document (Part I is items 1-4, Part II is items 5-9A), no ambiguity by part.
KNOWN_SECTIONS_10K: dict[str, str] = {
    "1": "Business",
    "1a": "Risk Factors",
    "1b": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7a": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
}

# 10-Qs are structured differently, and — unlike 10-Ks — reuse the same item numbers
# (1-4) in both parts with entirely different meanings, so the lookup must be part-aware.
KNOWN_SECTIONS_10Q_PART1: dict[str, str] = {
    "1": "Financial Statements",
    "2": "Management's Discussion and Analysis (MD&A)",
    "3": "Quantitative and Qualitative Disclosures About Market Risk",
    "4": "Controls and Procedures",
}
KNOWN_SECTIONS_10Q_PART2: dict[str, str] = {
    "1": "Legal Proceedings",
    "1a": "Risk Factors",
    "2": "Unregistered Sales of Equity Securities and Use of Proceeds",
    "3": "Defaults Upon Senior Securities",
    "4": "Mine Safety Disclosures",
    "5": "Other Information",
    "6": "Exhibits",
}


def _section_title(filing_type: str, part: int, item_number: str, fallback: str) -> str:
    if filing_type == "10-Q":
        table = KNOWN_SECTIONS_10Q_PART2 if part == 2 else KNOWN_SECTIONS_10Q_PART1
    else:
        table = KNOWN_SECTIONS_10K
    return table.get(item_number) or fallback or f"Item {item_number.upper()}"

CHARS_PER_TOKEN = 4  # rough heuristic, avoids a tokenizer dependency for chunking
DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 75


@dataclass
class Section:
    item_number: str  # "" if this is preamble text before the first Item header
    title: str
    text: str


@dataclass
class Chunk:
    text: str
    section: str  # human-readable section label, e.g. "Item 7 - MD&A"
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def html_to_text(html: str) -> str:
    """Strip tags/scripts/styles, collapse whitespace, keep paragraph breaks."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines but keep paragraph breaks
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def split_into_sections(text: str, filing_type: str = "10-K") -> list[Section]:
    """Split filing text into Item-numbered sections using line-start headers.

    Tracks which Part (I / II) we're in, since 10-Qs reuse item numbers 1-4 across
    both parts with different meanings — the item number alone is ambiguous for 10-Qs.
    """
    lines = text.split("\n")
    sections: list[Section] = []
    current_item = ""
    current_title = "Preamble"
    current_lines: list[str] = []
    current_part = 1

    def flush() -> None:
        if current_lines:
            sections.append(Section(current_item, current_title, "\n".join(current_lines).strip()))

    for line in lines:
        part_m = PART_HEADER_RE.match(line)
        if part_m and len(line) < 40:
            flush()
            current_part = 2 if part_m.group(1).lower() == "ii" else 1
            current_item = ""
            current_title = "Preamble"
            current_lines = []
            continue

        m = ITEM_HEADER_RE.match(line)
        # Guard against false positives like "Item 7 of the risk factors above discusses..."
        # mid-sentence by requiring the whole line to be short (i.e. actually heading-shaped).
        is_heading = m is not None and len(line) < 120
        if is_heading:
            flush()
            current_item = m.group(1).lower()
            current_title = _section_title(
                filing_type, current_part, current_item, m.group(2).strip()
            )
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return [s for s in sections if s.text]


def _split_long_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Token-window split (approximated via chars) with overlap, breaking on paragraph
    boundaries where possible so chunks don't cut mid-sentence."""
    chunk_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    if len(text) <= chunk_chars:
        return [text]

    paragraphs = text.split("\n")
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        candidate = f"{buf}\n{para}" if buf else para
        if len(candidate) > chunk_chars and buf:
            chunks.append(buf)
            # carry overlap from the tail of the previous chunk
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail}\n{para}" if tail else para
        else:
            buf = candidate
    if buf:
        chunks.append(buf)

    # Paragraphs longer than a whole chunk (e.g. huge tables) still need a hard split.
    final: list[str] = []
    for c in chunks:
        if len(c) <= chunk_chars * 1.5:
            final.append(c)
            continue
        for i in range(0, len(c), chunk_chars - overlap_chars):
            final.append(c[i : i + chunk_chars])
    return final


def chunk_filing(
    html: str,
    filing_type: str = "10-K",
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Full pipeline: HTML -> text -> Item sections -> token-windowed chunks."""
    text = html_to_text(html)
    sections = split_into_sections(text, filing_type=filing_type)

    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        label = (
            f"Item {section.item_number.upper()} - {section.title}"
            if section.item_number
            else section.title
        )
        for piece in _split_long_text(section.text, chunk_tokens, overlap_tokens):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(Chunk(text=piece, section=label, chunk_index=idx))
            idx += 1
    return chunks
