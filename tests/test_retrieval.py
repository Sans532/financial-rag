from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.hybrid_search import HybridSearcher, reciprocal_rank_fusion
from src.retrieval.vector_store import FilingChunkRecord, ScoredChunk


def make_record(text: str, company: str = "AAPL", chunk_index: int = 0, accession: str = "acc-1"):
    return FilingChunkRecord(
        text=text,
        company=company,
        filing_type="10-Q",
        filing_date="2024-01-01",
        section="Item 7 - MD&A",
        chunk_index=chunk_index,
        accession_number=accession,
    )


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Revenue grew 12%!") == ["revenue", "grew", "12"]


def test_bm25_index_empty_corpus_returns_nothing():
    index = BM25Index([])
    assert index.search("revenue") == []


def test_bm25_index_ranks_more_relevant_doc_higher():
    records = [
        make_record("The weather was sunny and pleasant.", chunk_index=0),
        make_record("Revenue increased due to strong iPhone sales growth.", chunk_index=1),
        make_record("Revenue revenue revenue growth revenue.", chunk_index=2),
    ]
    index = BM25Index(records)
    results = index.search("revenue growth", top_k=3)
    assert results, "expected at least one match"
    assert results[0].record.chunk_index in (1, 2)
    assert all(r.source == "bm25" for r in results)


def test_bm25_index_excludes_zero_score_docs():
    records = [make_record("completely unrelated text about weather", chunk_index=0)]
    index = BM25Index(records)
    results = index.search("nonexistent_term_xyz")
    assert results == []


def test_reciprocal_rank_fusion_merges_and_dedupes():
    r1 = make_record("dense hit", chunk_index=0, accession="a1")
    r2 = make_record("sparse hit", chunk_index=1, accession="a1")
    r3 = make_record("shared hit", chunk_index=2, accession="a1")

    dense = [
        ScoredChunk(record=r3, score=0.9, source="dense"),
        ScoredChunk(record=r1, score=0.5, source="dense"),
    ]
    sparse = [
        ScoredChunk(record=r3, score=5.0, source="bm25"),
        ScoredChunk(record=r2, score=3.0, source="bm25"),
    ]

    fused = reciprocal_rank_fusion(dense, sparse)
    keys = [f"{c.record.accession_number}:{c.record.chunk_index}" for c in fused]

    # the doc appearing in both lists (r3, chunk_index=2) should rank first
    assert keys[0] == "a1:2"
    # no duplicates
    assert len(keys) == len(set(keys))
    assert len(fused) == 3


def test_reciprocal_rank_fusion_handles_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []
    r1 = make_record("only dense")
    dense_only = reciprocal_rank_fusion([ScoredChunk(record=r1, score=1.0, source="dense")], [])
    assert len(dense_only) == 1


class _StubEmbeddingModel:
    def embed_one(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _StubVectorStore:
    def __init__(self, dense_results, corpus):
        self._dense_results = dense_results
        self._corpus = corpus

    def search(self, query_vector, top_k=10, company=None, filing_type=None):
        return self._dense_results

    def scroll_all(self, company=None):
        return self._corpus


def test_hybrid_searcher_without_reranker_returns_fused_results():
    r1 = make_record("Revenue grew due to strong demand.", chunk_index=0, accession="acc-x")
    r2 = make_record("Risk factors include competition.", chunk_index=1, accession="acc-x")

    dense_results = [ScoredChunk(record=r1, score=0.8, source="dense")]
    corpus = [r1, r2]

    searcher = HybridSearcher(
        vector_store=_StubVectorStore(dense_results, corpus),
        embedding_model=_StubEmbeddingModel(),
        reranker=None,
    )
    results = searcher.search("revenue demand", final_k=5, use_reranker=False)
    assert results
    assert any(r.record.chunk_index == 0 for r in results)


def test_hybrid_searcher_handles_empty_corpus():
    searcher = HybridSearcher(
        vector_store=_StubVectorStore([], []),
        embedding_model=_StubEmbeddingModel(),
        reranker=None,
    )
    assert searcher.search("anything", use_reranker=False) == []
