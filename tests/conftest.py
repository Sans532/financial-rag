import os

# Ensure a well-formed env before any src.config import happens, regardless of test order
# or a developer's local .env — tests must not depend on real secrets.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("SEC_USER_AGENT", "financial-rag-tests test@example.com")
os.environ.setdefault("FINNHUB_API_KEY", "")
