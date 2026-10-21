"""Build the schema index: 12 table-level documents, normalized bge-small embeddings, and provenance metadata.

Reads data/sqlite/food_delivery.sqlite (read-only, hash-checked) and the table comments in data/source/01_schema.sql.
Writes schema_docs.jsonl, schema_embeddings.npy and index_meta.json under data/index/ (or the directory given as the
first argument). BM25 is not built here; the retrieval component builds it in memory from the documents' "text".

The embedding model must already be in the local Hugging Face cache (HF_HUB_OFFLINE is forced on).
Existing outputs are never overwritten. Usage: python scripts/build_schema_index.py [output_dir]
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import collections
import hashlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "sqlite" / "food_delivery.sqlite"
SRC_DIR = ROOT / "data" / "source"
OUT_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "data" / "index"
FILES = {"docs": "schema_docs.jsonl", "emb": "schema_embeddings.npy", "meta": "index_meta.json"}

DB_SHA256 = "7efa6e3a56098a02f86785ff666f0d13686d650e42dff7d7d1002d242d451e0c"
SOURCE_SHA256 = {
    "01_schema.sql": "6a0e6266ea6ebb5c33dbf66afa2da6521981b725943b6f5a7800633f20db12a9",
    "02_data.sql": "4975b09b39e5d38305cd9d421d88ceff318c73f55dafdfd4959c96be930d2f91",
}
MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIM = 384
MAX_VALUES = 12

# Explicit allowlist: only these columns get sample values (categorical, non-personal).
SAMPLE_VALUE_COLUMNS = {
    "categories": ["category_name"], "coupons": ["coupon_code"], "customer_addresses": ["city", "state"],
    "customers": ["gender"], "delivery_partners": ["vehicle_type", "city"], "restaurants": ["city", "cuisine_type"],
    "orders": ["order_status"], "order_tracking": ["status"], "payments": ["payment_method", "payment_status"],
}
# Values from these columns must never appear anywhere in the output (checked by the privacy scan).
SENSITIVE_COLUMNS = {
    "customers": ["full_name", "email", "phone", "password_hash"], "customer_addresses": ["address_line", "pincode"],
    "delivery_partners": ["full_name", "phone"], "restaurants": ["email", "phone"], "ratings": ["review_text"],
    "order_tracking": ["notes"], "categories": ["description"],
}

results = []


def check(name, ok, detail=""):
    results.append((bool(ok), name))
    print(("PASS  " if ok else "FAIL  ") + name + (" - " + detail if detail else ""))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def connect_ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


# ----------------------------------------------------------------------------- documents
def parse_descriptions(ddl):
    """Table order and one-line descriptions from the '-- TABLE n: name' comment blocks of the source DDL."""
    out = collections.OrderedDict()
    for m in re.finditer(r"^-- TABLE (\d+): (\w+)[ \t]*\r?\n((?:--[ \t]+[^=\r\n][^\r\n]*\r?\n)*)", ddl, re.M):
        out[m.group(2)] = " ".join(line[2:].strip() for line in m.group(3).splitlines())
    return out


def column_text(c):
    bits = [c["type"]]
    if c["primary_key"]:
        bits.append("primary key")
    elif c["not_null"]:
        bits.append("not null")
    if c["references"]:
        bits.append(f"references {c['references']['table']}.{c['references']['column']}")
    if c["generated"]:
        bits.append("generated")
    if c["always_null"]:
        bits.append("always NULL, not populated")
    s = f"{c['name']} ({', '.join(bits)}"
    if c["sample_values"]:
        s += "; values: " + ", ".join(c["sample_values"])
    return s + ")"


def build_documents(con, descriptions):
    q = lambda sql: con.execute(sql).fetchall()
    tables = list(descriptions)
    fk_out = {t: [] for t in tables}
    referenced_by = {t: [] for t in tables}
    for t in tables:
        for _id, _seq, ref_table, frm, to, _upd, on_delete, _match in q(f"PRAGMA foreign_key_list({t})"):
            fk_out[t].append({"column": frm, "ref_table": ref_table, "ref_column": to, "on_delete": on_delete})
            referenced_by[ref_table].append({"table": t, "column": frm, "ref_column": to})
    docs = []
    for t in tables:
        fk_by_col = {f["column"]: f for f in fk_out[t]}
        n_rows = q(f"SELECT COUNT(*) FROM {t}")[0][0]
        cols = []
        for _cid, name, ctype, notnull, _dflt, pk, hidden in q(f"PRAGMA table_xinfo({t})"):
            samples = None
            if name in SAMPLE_VALUE_COLUMNS.get(t, []):
                samples = [r[0] for r in q(f"SELECT DISTINCT {name} FROM {t} WHERE {name} IS NOT NULL ORDER BY 1")]
                if not 0 < len(samples) <= MAX_VALUES or any(not isinstance(v, str) or ";" in v or "," in v for v in samples):
                    raise ValueError(f"allowlisted column {t}.{name} does not fit the sample-value rules")
            fk = fk_by_col.get(name)
            cols.append({
                "name": name, "type": ctype, "primary_key": bool(pk), "not_null": bool(notnull), "generated": hidden in (2, 3),
                "references": {"table": fk["ref_table"], "column": fk["ref_column"]} if fk else None,
                "always_null": n_rows > 0 and not pk and q(f"SELECT COUNT({name}) FROM {t}")[0][0] == 0,
                "sample_values": samples,
            })
        desc = descriptions[t]
        text = f"Table: {t}." + (f" {desc}" if desc else "") + "\nColumns: " + "; ".join(column_text(c) for c in cols) + "."
        docs.append({"doc_id": t, "table": t, "description": desc, "columns": cols,
                     "foreign_keys_out": fk_out[t], "referenced_by": referenced_by[t], "text": text})
    return docs


# ----------------------------------------------------------------------------- embeddings
def embed(texts):
    import sentence_transformers, torch, transformers
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, device="cpu", revision=REVISION, local_files_only=True)
    ids = model.tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
    token_counts = [len(x) for x in ids]
    emb = model.encode(texts, batch_size=len(texts), normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float32)
    libs = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "sentence-transformers": sentence_transformers.__version__}
    return model, emb, token_counts, libs


# ----------------------------------------------------------------------------- main
def main():
    t_start = time.perf_counter()
    for fname, want in SOURCE_SHA256.items():
        check(f"source hash {fname} matches D-13", sha256_file(SRC_DIR / fname) == want)
    check("database hash matches D-14", sha256_file(DB_PATH) == DB_SHA256, sha256_file(DB_PATH))
    if not all(ok for ok, _ in results):
        sys.exit("inputs do not match the pinned hashes; nothing was built")
    finals = {k: OUT_DIR / v for k, v in FILES.items()}
    tmps = {k: p.with_name(p.name + ".tmp") for k, p in finals.items()}
    for p in list(finals.values()) + list(tmps.values()):
        if p.exists():
            sys.exit(f"refusing to overwrite existing file: {p}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SRC_DIR / "01_schema.sql", encoding="ascii", newline="") as f:
        descriptions = parse_descriptions(f.read())
    con = connect_ro(DB_PATH)
    docs = build_documents(con, descriptions)
    texts = [d["text"] for d in docs]

    t0 = time.perf_counter()
    model, emb, token_counts, libs = embed(texts)
    t_embed = time.perf_counter() - t0

    docs_blob = "".join(json.dumps(d, ensure_ascii=True) + "\n" for d in docs)
    tmps["docs"].write_bytes(docs_blob.encode("ascii"))
    with open(tmps["emb"], "wb") as f:
        np.save(f, emb, allow_pickle=False)
    meta = {
        "schema_version": 1,
        "documents": {"file": FILES["docs"], "count": len(docs), "granularity": "table", "sha256": sha256_file(tmps["docs"]),
                      "text_sha256": {d["doc_id"]: sha256_bytes(d["text"].encode("utf-8")) for d in docs}},
        "embeddings": {"file": FILES["emb"], "sha256": sha256_file(tmps["emb"]), "shape": list(emb.shape), "dtype": "float32",
                       "normalized": True, "row_order": f"same as {FILES['docs']}", "passage_instruction": None},
        "model": {"id": MODEL_ID, "revision": REVISION, "max_seq_length": model.max_seq_length,
                  "pooling": str(model[1].pooling_mode), "token_counts": dict(zip([d["doc_id"] for d in docs], token_counts))},
        "sources": {"database": {"path": "data/sqlite/food_delivery.sqlite", "sha256": DB_SHA256},
                    "source_sql": {k: {"path": f"data/source/{k}", "sha256": v} for k, v in SOURCE_SHA256.items()}},
        "sample_values": {"allowlist": SAMPLE_VALUE_COLUMNS, "max_values_per_column": MAX_VALUES},
        "excluded_columns": SENSITIVE_COLUMNS,
        "libraries": libs, "builder_script_sha256": sha256_file(__file__),
    }
    meta_blob = json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    tmps["meta"].write_bytes(meta_blob.encode("ascii"))

    # ---- verification: everything below re-reads the files that were just written
    docs_r = [json.loads(line) for line in tmps["docs"].read_text(encoding="ascii").splitlines()]
    emb_r = np.load(tmps["emb"], allow_pickle=False)
    meta_r = json.loads(tmps["meta"].read_text(encoding="ascii"))
    q = lambda sql: con.execute(sql).fetchall()
    db_tables = [r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

    check("12 documents, one per table", [d["doc_id"] for d in docs_r] == list(descriptions) and sorted(descriptions) == sorted(db_tables),
          f"{len(docs_r)} documents")
    check("descriptions: 9 from source comments, 3 empty (restaurants, delivery_partners, coupons)",
          sorted(d["table"] for d in docs_r if not d["description"]) == ["coupons", "delivery_partners", "restaurants"])
    fk_docs = {(d["table"], f["column"], f["ref_table"], f["ref_column"], f["on_delete"]) for d in docs_r for f in d["foreign_keys_out"]}
    fk_db = {(t, r[3], r[2], r[4], r[6]) for t in db_tables for r in q(f"PRAGMA foreign_key_list({t})")}
    check("15 foreign keys in the documents equal PRAGMA foreign_key_list", fk_docs == fk_db and len(fk_docs) == 15, f"{len(fk_docs)} FKs")
    check("referenced_by is the exact inverse of foreign_keys_out",
          {(r["table"], r["column"], d["table"], r["ref_column"]) for d in docs_r for r in d["referenced_by"]}
          == {(t, c, rt, rc) for (t, c, rt, rc, _) in fk_docs})
    col_problems = []
    for d in docs_r:
        x = q(f"PRAGMA table_xinfo({d['table']})")
        if [c["name"] for c in d["columns"]] != [r[1] for r in x] or [c["type"] for c in d["columns"]] != [r[2] for r in x] \
                or [c["primary_key"] for c in d["columns"]] != [bool(r[5]) for r in x]:
            col_problems.append(d["table"])
    check("columns (names, order, types, primary keys) equal PRAGMA table_xinfo", not col_problems, str(col_problems))
    check("only order_items.subtotal is flagged generated",
          [(d["table"], c["name"]) for d in docs_r for c in d["columns"] if c["generated"]] == [("order_items", "subtotal")])
    sv_cols = sorted((d["table"], c["name"]) for d in docs_r for c in d["columns"] if c["sample_values"])
    check("sample values exist only for the 13 allowlisted columns",
          sv_cols == sorted((t, c) for t, cs in SAMPLE_VALUE_COLUMNS.items() for c in cs) and len(sv_cols) == 13, f"{len(sv_cols)} columns")
    bad_sv = [(t, c) for t, c in sv_cols for d in docs_r if d["table"] == t for col in d["columns"] if col["name"] == c
              and (col["sample_values"] != [r[0] for r in q(f"SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL ORDER BY 1")]
                   or len(col["sample_values"]) > MAX_VALUES)]
    check("every sample value list equals the database's distinct values (max 12)", not bad_sv, str(bad_sv))
    mi = next(c for d in docs_r if d["table"] == "menu_items" for c in d["columns"] if c["name"] == "created_at")
    all_null = [(d["table"], c["name"]) for d in docs_r for c in d["columns"] if c["always_null"]]
    check("menu_items.created_at is listed as 'always NULL' with no sample values; it is the only such column",
          all_null == [("menu_items", "created_at")] and mi["sample_values"] is None
          and "created_at (DATETIME, always NULL, not populated)" in next(d["text"] for d in docs_r if d["table"] == "menu_items"))

    blob = docs_blob + meta_blob
    leaked = []
    for t, cols in SENSITIVE_COLUMNS.items():
        for c in cols:
            for (v,) in q(f"SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL"):
                if len(str(v)) >= 4 and str(v) in blob:
                    leaked.append(f"{t}.{c}")
    check("privacy: no value from any excluded column (hashes, emails, phones, names, addresses, free text) appears in the output",
          not leaked, str(sorted(set(leaked))))
    patterns = {"email": r"[\w.+-]+@[\w-]+\.[\w.-]+", "10-digit number": r"(?<!\d)\d{10}(?!\d)", "64-hex hash": r"\b[0-9a-f]{64}\b"}
    hits = {k: len(re.findall(p, docs_blob)) for k, p in patterns.items()}
    check("privacy: no email, 10-digit number or 64-hex hash pattern in the documents", not any(hits.values()), str(hits))

    check("token counts: every document fits the model's 512-token limit", max(token_counts) <= model.max_seq_length,
          f"max {max(token_counts)} of {model.max_seq_length}; counts {dict(zip([d['doc_id'] for d in docs], token_counts))}")
    check("embeddings: shape (12, 384), float32, all finite", emb_r.shape == (12, DIM) and emb_r.dtype == np.float32 and bool(np.isfinite(emb_r).all()),
          f"{emb_r.shape} {emb_r.dtype}")
    norms = np.linalg.norm(emb_r, axis=1)
    check("embeddings: unit norm within 1e-5", bool(np.all(np.abs(norms - 1.0) < 1e-5)), f"max deviation {np.abs(norms - 1.0).max():.2e}")
    single = np.vstack([model.encode([d["text"]], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False) for d in docs_r])
    check("embeddings: row i is the embedding of document i (re-encoded one at a time)", bool(np.allclose(single, emb_r, atol=1e-4)),
          f"max abs difference {np.abs(single - emb_r).max():.2e}")
    check("embeddings are not degenerate (distinct documents differ)",
          bool((emb_r @ emb_r.T)[np.triu_indices(12, 1)].max() < 0.999))
    check("provenance: meta hashes match the files and the per-document text hashes",
          meta_r["documents"]["sha256"] == sha256_file(tmps["docs"]) and meta_r["embeddings"]["sha256"] == sha256_file(tmps["emb"])
          and all(meta_r["documents"]["text_sha256"][d["doc_id"]] == sha256_bytes(d["text"].encode("utf-8")) for d in docs_r))
    check("provenance: model id, pinned revision, database and source hashes recorded",
          meta_r["model"]["id"] == MODEL_ID and meta_r["model"]["revision"] == REVISION
          and meta_r["sources"]["database"]["sha256"] == DB_SHA256 and meta_r["sources"]["source_sql"]["02_data.sql"]["sha256"] == SOURCE_SHA256["02_data.sql"])
    check("database unchanged by the build", sha256_file(DB_PATH) == DB_SHA256)
    con.close()

    if not all(ok for ok, _ in results):
        print("\nVERIFICATION FAILED - temporary files left for inspection:", ", ".join(str(p) for p in tmps.values()))
        sys.exit(1)
    for k in finals:
        os.replace(tmps[k], finals[k])
    passed = sum(ok for ok, _ in results)
    print(f"\n{passed}/{len(results)} checks passed | embedding time {t_embed:.1f} s | total {time.perf_counter() - t_start:.1f} s")
    for k, p in finals.items():
        print(f"  {p}  {p.stat().st_size} bytes  sha256 {sha256_file(p)}")


if __name__ == "__main__":
    main()
