"""Hybrid schema retrieval over the table-level index: BM25 + dense (bge-small-en-v1.5), fused with RRF.

Reads the artifacts in data/index/ (built by scripts/build_schema_index.py) after validating their provenance
against index_meta.json and the current SQLite database. The embedding model is loaded lazily, offline, from the
local Hugging Face cache; it is never downloaded here.

Usage: python -m text2sql.retrieval "which customers placed orders?" [--top-k 5] [--no-instruction]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
DB_PATH = ROOT / "data" / "sqlite" / "food_delivery.sqlite"
MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "  # model card, for query->passage search
RRF_K = 60  # conventional RRF constant; configurable on SchemaRetriever
DEFAULT_TOP_K = 5  # Recall@5 is defined on the top five tables (docs/eval-protocol.md)
CONTEXT_TOP_K = 3  # retrieved tables that seed the prompt context before FK parent expansion

# Function words only, so a question's wording ("how many ...") does not match description text such as "has many".
STOPWORDS = frozenset(
    "a an and are as at be been by can could did do does each for from had has have how i if in into is it its me "
    "many my of on or per than that the their them then there these they this those to us was we were what when "
    "where which who whom whose why will with would you your show tell give get list find".split()
)


class IndexProvenanceError(Exception):
    """The index artifacts are missing, altered, or do not belong to the current database/model."""


# ----------------------------------------------------------------------------- text processing
def _fold(word: str) -> str:
    """Minimal plural folding (customers -> customer, categories -> category); applied to documents and queries alike."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; snake_case identifiers yield the whole identifier plus its parts; stop words are dropped."""
    tokens: list[str] = []
    for raw in re.findall(r"[a-z0-9_]+", text.lower()):
        parts = [p for p in raw.split("_") if p]
        if len(parts) != 1:
            tokens.append(raw)
        tokens.extend(_fold(p) for p in parts)
    return [t for t in tokens if t not in STOPWORDS and t.strip("_")]


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), ranks starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


# ----------------------------------------------------------------------------- provenance
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_index(index_dir: Path = INDEX_DIR, db_path: Path = DB_PATH):
    """Validate and load (docs, embeddings, meta). Raises IndexProvenanceError on any mismatch."""
    index_dir, db_path = Path(index_dir), Path(db_path)
    paths = {"meta": index_dir / "index_meta.json", "docs": index_dir / "schema_docs.jsonl", "emb": index_dir / "schema_embeddings.npy"}
    missing = [str(p) for p in [*paths.values(), db_path] if not p.is_file()]
    if missing:
        raise IndexProvenanceError("missing file(s): " + ", ".join(missing))
    try:
        meta = json.loads(paths["meta"].read_text(encoding="ascii"))
        problems = []
        if meta["schema_version"] != 1:
            problems.append(f"unsupported index schema_version {meta['schema_version']}")
        if meta["model"]["id"] != MODEL_ID or meta["model"]["revision"] != REVISION:
            problems.append("index was built with a different embedding model/revision")
        if _sha256(paths["docs"]) != meta["documents"]["sha256"]:
            problems.append("schema_docs.jsonl does not match the hash in index_meta.json")
        if _sha256(paths["emb"]) != meta["embeddings"]["sha256"]:
            problems.append("schema_embeddings.npy does not match the hash in index_meta.json")
        if _sha256(db_path) != meta["sources"]["database"]["sha256"]:
            problems.append("the database changed since the index was built (stale index)")
    except (KeyError, TypeError, ValueError) as e:  # damaged or unexpected meta file
        raise IndexProvenanceError(f"index_meta.json is unreadable or incomplete: {type(e).__name__}: {e}") from e
    if problems:
        raise IndexProvenanceError("; ".join(problems))

    docs = [json.loads(line) for line in paths["docs"].read_text(encoding="ascii").splitlines()]
    emb = np.load(paths["emb"], allow_pickle=False)
    if emb.shape != (len(docs), meta["embeddings"]["shape"][1]) or list(emb.shape) != meta["embeddings"]["shape"] or emb.dtype != np.float32:
        raise IndexProvenanceError(f"embedding matrix has shape {emb.shape}, dtype {emb.dtype}; expected {meta['embeddings']['shape']} float32")
    if len(docs) != meta["documents"]["count"]:
        raise IndexProvenanceError("document count does not match index_meta.json")
    for d in docs:
        if meta["documents"]["text_sha256"].get(d["doc_id"]) != hashlib.sha256(d["text"].encode("utf-8")).hexdigest():
            raise IndexProvenanceError(f"document text for {d['doc_id']} does not match its recorded hash")
    return docs, emb, meta


# ----------------------------------------------------------------------------- retrieval
@dataclass(frozen=True)
class RetrievedTable:
    table: str
    rank: int
    rrf_score: float
    bm25_rank: int | None  # None: no lexical match, so BM25 did not rank this table
    bm25_score: float
    dense_rank: int
    dense_score: float
    matched_terms: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class ContextTable:
    table: str
    kind: str  # "retrieved" (chosen by the retriever) or "fk_parent" (added because a retrieved table's foreign key references it)
    rank: int | None  # original retrieval rank; None for FK-added tables
    added_for: tuple[str, ...]  # fk_parent only: the retrieved tables whose foreign keys reference this table


def expand_parents(retrieved: list[RetrievedTable], fk_parents: dict[str, tuple[str, ...]]) -> list[ContextTable]:
    """Retrieved tables in their original rank order, then their direct FK parents. One level only: no children, no recursion.

    FK-added tables appear in order of first reference (retrieval rank, then FK column order); a parent that was itself
    retrieved stays a "retrieved" table and is not repeated.
    """
    context = [ContextTable(r.table, "retrieved", r.rank, ()) for r in retrieved]
    chosen = {r.table for r in retrieved}
    added: dict[str, list[str]] = {}
    for r in retrieved:
        for parent in fk_parents.get(r.table, ()):
            if parent not in chosen:
                added.setdefault(parent, []).append(r.table)
    return context + [ContextTable(parent, "fk_parent", None, tuple(by)) for parent, by in added.items()]


class SchemaRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR, db_path: Path = DB_PATH, use_query_instruction: bool = True, rrf_k: int = RRF_K):
        self.docs, self._doc_emb, self.meta = load_index(index_dir, db_path)
        self._fk_parents = {}  # table -> tuple of directly referenced tables, ordered by the FK column's position in the table
        for d in self.docs:
            position = {c["name"]: i for i, c in enumerate(d["columns"])}
            fks = sorted(d["foreign_keys_out"], key=lambda f: position[f["column"]])
            self._fk_parents[d["table"]] = tuple(dict.fromkeys(f["ref_table"] for f in fks if f["ref_table"] != d["table"]))
        self.use_query_instruction = use_query_instruction
        self.rrf_k = rrf_k
        self._doc_tokens = [tokenize(d["text"]) for d in self.docs]
        self._bm25 = BM25Okapi(self._doc_tokens)  # default k1=1.5, b=0.75
        self._token_sets = [set(t) for t in self._doc_tokens]
        self._model = None

    def _encode_query(self, query: str) -> np.ndarray:
        if self._model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download; the model must already be cached
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_ID, device="cpu", revision=REVISION, local_files_only=True)
        text = (QUERY_INSTRUCTION if self.use_query_instruction else "") + query
        vec = self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
        if vec.shape[0] != self._doc_emb.shape[1]:
            raise IndexProvenanceError(f"query embedding has {vec.shape[0]} dimensions; the index has {self._doc_emb.shape[1]}")
        return vec

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedTable]:
        if not isinstance(query, str) or not re.search(r"[A-Za-z0-9]", query):
            raise ValueError("query must be a string containing at least one letter or digit")
        n = len(self.docs)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= n:
            raise ValueError(f"top_k must be an integer from 1 to {n}")

        q_tokens = tokenize(query)
        bm25_scores = self._bm25.get_scores(q_tokens) if q_tokens else np.zeros(n)
        # Only tables with a lexical match are ranked by BM25; ties break by document order for determinism.
        bm25_order = [i for i in sorted(range(n), key=lambda i: (-float(bm25_scores[i]), i)) if bm25_scores[i] > 0]
        dense_scores = self._doc_emb @ self._encode_query(query)
        dense_order = sorted(range(n), key=lambda i: (-float(dense_scores[i]), i))
        fused = rrf_fuse([bm25_order, dense_order], self.rrf_k)

        bm25_rank = {i: r for r, i in enumerate(bm25_order, start=1)}
        dense_rank = {i: r for r, i in enumerate(dense_order, start=1)}
        results = []
        for rank, i in enumerate(sorted(range(n), key=lambda i: (-fused.get(i, 0.0), i))[:top_k], start=1):
            matched = tuple(dict.fromkeys(t for t in q_tokens if t in self._token_sets[i]))
            bm = f"BM25 #{bm25_rank[i]} ({float(bm25_scores[i]):.2f}; matched: {', '.join(matched)})" if i in bm25_rank else "BM25 -- (no lexical match)"
            results.append(RetrievedTable(
                table=self.docs[i]["table"], rank=rank, rrf_score=fused.get(i, 0.0),
                bm25_rank=bm25_rank.get(i), bm25_score=float(bm25_scores[i]),
                dense_rank=dense_rank[i], dense_score=float(dense_scores[i]), matched_terms=matched,
                explanation=f"{bm} + dense #{dense_rank[i]} (cos {float(dense_scores[i]):.3f})"))
        return results

    def retrieve_context(self, query: str, top_k: int = CONTEXT_TOP_K) -> list[ContextTable]:
        """The retriever's top_k tables (original ranks kept) plus their direct FK parents."""
        return expand_parents(self.retrieve(query, top_k), self._fk_parents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank the schema tables most relevant to a question.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--no-instruction", action="store_true", help="embed the query without the BGE query instruction")
    args = parser.parse_args()
    retriever = SchemaRetriever(use_query_instruction=not args.no_instruction)
    for r in retriever.retrieve(args.query, args.top_k):
        print(f"{r.rank}. {r.table:20} rrf {r.rrf_score:.4f} | {r.explanation}")


if __name__ == "__main__":
    main()
