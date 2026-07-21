import httpx
import pytest
import respx

from src.data.edgar_client import EdgarClient, EdgarRateLimiter, FilingRef


def test_filing_ref_accession_nodash():
    ref = FilingRef(
        accession_number="0000320193-24-000123",
        form="10-K",
        filing_date="2024-11-01",
        report_date="2024-09-28",
        primary_document="aapl-20240928.htm",
        cik="0000320193",
    )
    assert ref.accession_nodash == "000032019324000123"


def test_rate_limiter_enforces_minimum_spacing():
    import time

    limiter = EdgarRateLimiter(max_per_second=50.0)  # 20ms min interval
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.02


@respx.mock
def test_get_submissions_returns_json():
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json={"cik": 320193, "filings": {"recent": {}}})
    )
    client = EdgarClient(user_agent="test test@example.com")
    data = client.get_submissions("0000320193")
    assert data["cik"] == 320193


@respx.mock
def test_list_filings_filters_by_form_and_respects_limit():
    payload = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-Q", "10-Q", "10-K"],
                "accessionNumber": ["a1", "a2", "a3", "a4", "a5"],
                "filingDate": [
                    "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01",
                ],
                "reportDate": [
                    "2023-12-31", "2024-01-31", "2024-02-28", "2024-03-31", "2024-04-30",
                ],
                "primaryDocument": ["d1.htm", "d2.htm", "d3.htm", "d4.htm", "d5.htm"],
            }
        }
    }
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=payload)
    )
    client = EdgarClient(user_agent="test test@example.com")
    filings = client.list_filings("0000320193", forms=("10-K", "10-Q"), limit=2)
    assert len(filings) == 2
    assert all(f.form in ("10-K", "10-Q") for f in filings)
    assert filings[0].accession_number == "a1"


@respx.mock
def test_list_filings_returns_empty_when_no_matching_forms():
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K"],
                "accessionNumber": ["a1"],
                "filingDate": ["2024-01-01"],
                "reportDate": ["2023-12-31"],
                "primaryDocument": ["d1.htm"],
            }
        }
    }
    respx.get("https://data.sec.gov/submissions/CIK0000999999.json").mock(
        return_value=httpx.Response(200, json=payload)
    )
    client = EdgarClient(user_agent="test test@example.com")
    filings = client.list_filings("0000999999", forms=("10-K", "10-Q"))
    assert filings == []


@respx.mock
def test_get_filing_document_builds_correct_archive_url():
    ref = FilingRef(
        accession_number="0000320193-24-000123",
        form="10-K",
        filing_date="2024-11-01",
        report_date="2024-09-28",
        primary_document="aapl-20240928.htm",
        cik="0000320193",
    )
    expected_url = (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )
    respx.get(expected_url).mock(return_value=httpx.Response(200, text="<html>filing</html>"))

    client = EdgarClient(user_agent="test test@example.com")
    text = client.get_filing_document(ref)
    assert "filing" in text


@respx.mock
def test_get_submissions_raises_after_retries_exhausted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # skip real backoff delay
    route = respx.get("https://data.sec.gov/submissions/CIK0000000001.json").mock(
        return_value=httpx.Response(500)
    )
    client = EdgarClient(user_agent="test test@example.com")
    with pytest.raises(httpx.HTTPStatusError):
        client.get_submissions("0000000001")
    assert route.call_count == 4  # stop_after_attempt(4)


def test_client_context_manager_closes():
    with EdgarClient(user_agent="test test@example.com") as client:
        assert client is not None
