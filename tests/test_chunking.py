from src.retrieval.chunking import (
    Section,
    chunk_filing,
    html_to_text,
    split_into_sections,
)


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><style>.x{}</style></head><body><script>evil()</script>" \
        "<p>Hello</p><p>World</p></body></html>"
    text = html_to_text(html)
    assert "evil()" not in text
    assert "Hello" in text
    assert "World" in text


def test_html_to_text_collapses_blank_lines():
    html = "<p>A</p>\n\n\n<p></p>\n<p>B</p>"
    text = html_to_text(html)
    lines = text.split("\n")
    assert all(line for line in lines)  # no blank lines survive


def test_split_into_sections_recognizes_known_items():
    text = (
        "Some preamble text.\n"
        "Item 1A. Risk Factors\n"
        "We face many risks.\n"
        "Item 7. Management's Discussion and Analysis\n"
        "Revenue grew this quarter.\n"
    )
    sections = split_into_sections(text)
    assert len(sections) == 3
    assert sections[0].item_number == ""
    assert sections[1].item_number == "1a"
    assert sections[1].title == "Risk Factors"
    assert "many risks" in sections[1].text
    assert sections[2].item_number == "7"
    assert "MD&A" in sections[2].title or "Analysis" in sections[2].title


def test_split_into_sections_ignores_mid_sentence_item_mentions():
    """A long line that happens to start with 'Item 7' mid-discussion should not be
    treated as a new section header — only short heading-like lines split."""
    long_line = (
        "Item 7 of the risk factors described above discusses supply chain constraints "
        "that could materially affect our results of operations in ways that are hard "
        "to predict given current macroeconomic uncertainty across our major markets."
    )
    text = f"Item 1A. Risk Factors\nShort risk intro.\n{long_line}\nMore risk text.\n"
    sections = split_into_sections(text)
    assert len(sections) == 1
    assert long_line in sections[0].text


def test_split_into_sections_empty_text_returns_empty():
    assert split_into_sections("") == []


def test_split_into_sections_10q_disambiguates_item_1_by_part():
    """10-Qs reuse item numbers 1-4 in both Part I and Part II with different meanings —
    Part I Item 1 is Financial Statements, Part II Item 1 is Legal Proceedings. Without
    part-tracking these would collide and mislabel every 10-Q chunk (a real bug found via
    live ingestion: AAPL's Financial Statements section was mislabeled 'Business')."""
    text = (
        "PART I - FINANCIAL INFORMATION\n"
        "Item 1. Financial Statements\n"
        "Balance sheet data here.\n"
        "Item 2. Management's Discussion and Analysis\n"
        "Revenue discussion here.\n"
        "PART II - OTHER INFORMATION\n"
        "Item 1. Legal Proceedings\n"
        "Litigation summary here.\n"
        "Item 1A. Risk Factors\n"
        "Risk discussion here.\n"
    )
    sections = split_into_sections(text, filing_type="10-Q")

    # "Item 1" appears twice (once per part) — each occurrence must resolve independently.
    part1_item1 = next(s for s in sections if "Balance sheet" in s.text)
    assert part1_item1.title == "Financial Statements"
    assert part1_item1.item_number == "1"

    part1_item2 = next(s for s in sections if "Revenue discussion" in s.text)
    assert "Discussion and Analysis" in part1_item2.title or "MD&A" in part1_item2.title

    # the second "Item 1" (in Part II) must NOT be mislabeled as Financial Statements again
    part2_item1 = next(s for s in sections if "Litigation" in s.text)
    assert part2_item1.title == "Legal Proceedings"
    assert part2_item1.item_number == "1"


def test_split_into_sections_10k_vs_10q_same_item_number_different_meaning():
    text = "Item 1. Some heading text\nBody text.\n"
    k_sections = split_into_sections(text, filing_type="10-K")
    q_sections = split_into_sections(text, filing_type="10-Q")
    assert k_sections[0].title == "Business"
    assert q_sections[0].title == "Financial Statements"


def test_chunk_filing_10q_labels_financial_statements_correctly():
    html = (
        "<html><body>"
        "<p>PART I - FINANCIAL INFORMATION</p>"
        "<p>Item 1. Financial Statements</p><p>Balance sheet contents.</p>"
        "<p>Item 2. Management's Discussion and Analysis</p><p>MD&A contents.</p>"
        "</body></html>"
    )
    chunks = chunk_filing(html, filing_type="10-Q")
    sections = {c.section for c in chunks}
    assert any("Financial Statements" in s for s in sections)
    assert not any(s == "Item 1 - Business" for s in sections)


def test_chunk_filing_produces_labeled_chunks():
    html = (
        "<html><body>"
        "<p>Item 1A. Risk Factors</p><p>Risk one. Risk two.</p>"
        "<p>Item 7. Management's Discussion and Analysis</p><p>Revenue discussion.</p>"
        "</body></html>"
    )
    chunks = chunk_filing(html)
    assert len(chunks) >= 2
    sections = {c.section for c in chunks}
    assert any("Risk Factors" in s for s in sections)
    assert any("MD&A" in s or "Analysis" in s for s in sections)
    # chunk_index must be strictly increasing and unique
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(set(indices))


def test_chunk_filing_splits_long_sections_with_overlap():
    long_section = "\n".join(f"Paragraph number {i} with some filler content." for i in range(400))
    html = f"<html><body><p>Item 1. Business</p><p>{long_section}</p></body></html>"
    chunks = chunk_filing(html, chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # every chunk should stay roughly within the requested size bound
    for c in chunks:
        assert len(c.text) <= 100 * 4 * 2  # generous upper bound given paragraph-boundary splitting


def test_chunk_filing_handles_no_item_headers_gracefully():
    """A filing fragment with no recognizable Item headers should still produce chunks
    (as a single 'Preamble' section) instead of crashing or returning nothing."""
    html = (
        "<html><body><p>Just some unstructured text with no item headers at all.</p>"
        "</body></html>"
    )
    chunks = chunk_filing(html)
    assert len(chunks) == 1
    assert chunks[0].section == "Preamble"


def test_section_dataclass_defaults():
    s = Section(item_number="1a", title="Risk Factors", text="body")
    assert s.item_number == "1a"
    assert s.text == "body"
