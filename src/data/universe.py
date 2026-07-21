"""Fixed universe of companies covered by this system.

18 large-cap tickers across 3 sectors, chosen for clean/consistently-tagged
XBRL data and reliable coverage across EDGAR, yfinance, and Finnhub's free tier.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    sector: str
    cik: str  # zero-padded 10-digit SEC CIK


# CIKs from https://www.sec.gov/cgi-bin/browse-edgar (company search)
COMPANIES: list[Company] = [
    # Tech
    Company("AAPL", "Apple Inc.", "Technology", "0000320193"),
    Company("MSFT", "Microsoft Corporation", "Technology", "0000789019"),
    Company("GOOGL", "Alphabet Inc.", "Technology", "0001652044"),
    Company("NVDA", "NVIDIA Corporation", "Technology", "0001045810"),
    Company("META", "Meta Platforms, Inc.", "Technology", "0001326801"),
    Company("CRM", "Salesforce, Inc.", "Technology", "0001108524"),
    # Finance
    Company("JPM", "JPMorgan Chase & Co.", "Finance", "0000019617"),
    Company("BAC", "Bank of America Corporation", "Finance", "0000070858"),
    Company("GS", "The Goldman Sachs Group, Inc.", "Finance", "0000886982"),
    Company("MS", "Morgan Stanley", "Finance", "0000895421"),
    Company("WFC", "Wells Fargo & Company", "Finance", "0000072971"),
    Company("V", "Visa Inc.", "Finance", "0001403161"),
    # Healthcare
    Company("JNJ", "Johnson & Johnson", "Healthcare", "0000200406"),
    Company("UNH", "UnitedHealth Group Incorporated", "Healthcare", "0000731766"),
    Company("PFE", "Pfizer Inc.", "Healthcare", "0000078003"),
    Company("MRK", "Merck & Co., Inc.", "Healthcare", "0000310158"),
    Company("ABBV", "AbbVie Inc.", "Healthcare", "0001551152"),
    Company("LLY", "Eli Lilly and Company", "Healthcare", "0000059478"),
]

TICKER_TO_COMPANY: dict[str, Company] = {c.ticker: c for c in COMPANIES}


def get_company(ticker: str) -> Company:
    try:
        return TICKER_TO_COMPANY[ticker.upper()]
    except KeyError as e:
        raise ValueError(f"Ticker {ticker!r} is not in the covered universe") from e


def sectors() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in COMPANIES:
        out.setdefault(c.sector, []).append(c.ticker)
    return out
