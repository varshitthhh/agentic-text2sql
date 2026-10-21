"""Compact semantic layer (D-05): business definitions, conventions, aliases, join guidance and ambiguity rules.

The layer lives in data/semantic/semantic_layer.json. Every entry names the schema columns it refers to (checked against
the index at load time) and its source (gold-set convention, schema, or a data check). Entries that are claims about the
data carry a check_sql that the tests execute read-only. The whole layer is placed in the prompt (no selective retrieval).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .execution import ROOT
from .retrieval import ContextTable

LAYER_PATH = ROOT / "data" / "semantic" / "semantic_layer.json"


class SemanticLayerError(ValueError):
    """The semantic layer file is malformed or refers to columns that do not exist."""


def load_layer(docs: list[dict], path=LAYER_PATH) -> dict:
    """Load the layer and verify its structure and that every table.column reference exists in the index documents."""
    try:
        layer = json.loads(path.read_text(encoding="utf-8"))
        if layer["version"] != 1:
            raise SemanticLayerError(f"unsupported semantic layer version {layer['version']}")
        known = {f"{d['table']}.{c['name']}" for d in docs for c in d["columns"]}
        seen, problems = set(), []
        for section in layer["sections"]:
            for e in section["entries"]:
                if not all(isinstance(e[k], str) and e[k].strip() for k in ("id", "text", "source")):
                    problems.append(f"entry {e.get('id')!r}: id, text and source must be non-empty strings")
                if e["id"] in seen:
                    problems.append(f"duplicate entry id {e['id']!r}")
                seen.add(e["id"])
                problems += [f"entry {e['id']!r}: unknown column {r!r}" for r in e["refs"] if r not in known]
                if not e["refs"]:
                    problems.append(f"entry {e['id']!r}: refs must not be empty")
    except (OSError, ValueError, KeyError, TypeError) as err:
        if isinstance(err, SemanticLayerError):
            raise
        raise SemanticLayerError(f"semantic layer unreadable or incomplete: {type(err).__name__}: {err}") from err
    if problems:
        raise SemanticLayerError("; ".join(problems))
    return layer


def render_layer(layer: dict) -> str:
    """The compact prompt text: one heading per section, one line per entry, in file order."""
    lines = []
    for section in layer["sections"]:
        lines.append(f"{section['title']}:")
        lines += [f"- {e['text']}" for e in section["entries"]]
    return "\n".join(lines)


@dataclass(frozen=True)
class PromptContext:
    text: str
    tables: tuple[str, ...]  # context tables in prompt order (retrieved first, then FK-added)
    n_retrieved: int
    n_fk_added: int


def render_schema(context: list[ContextTable], docs: list[dict]) -> str:
    """Schema documents of the retrieved tables (rank order kept), then of the FK-added tables."""
    by_table = {d["table"]: d["text"] for d in docs}
    parts = ["Tables most relevant to the question:"] + [by_table[c.table] for c in context if c.kind == "retrieved"]
    added = [c for c in context if c.kind == "fk_parent"]
    if added:
        parts += ["", "Related tables (referenced by foreign keys of the tables above):"] + [by_table[c.table] for c in added]
    return "\n".join(parts)


def build_prompt_context(context: list[ContextTable], docs: list[dict], layer: dict) -> PromptContext:
    """Schema documents for the retrieved and FK-added tables (retrieval order kept), then the whole semantic layer."""
    retrieved = [c for c in context if c.kind == "retrieved"]
    added = [c for c in context if c.kind == "fk_parent"]
    text = "\n".join(["Database: SQLite. Use only the tables and columns listed below.", "", render_schema(context, docs), "",
                      "Business rules:", render_layer(layer)])
    return PromptContext(text, tuple(c.table for c in retrieved + added), len(retrieved), len(added))
