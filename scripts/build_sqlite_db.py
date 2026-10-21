"""Build data/sqlite/food_delivery.sqlite from the pinned MySQL source SQL, then verify it.

Standard library only. Reads data/source/01_schema.sql and 02_data.sql (hash-checked against
docs/decisions.md D-13), applies a fixed list of MySQL -> SQLite transformations, builds a temporary
database, verifies it against an independent parse of the original source files, and only then
renames it to the final path. Existing files are never overwritten; the source files are never modified.

Usage: python scripts/build_sqlite_db.py
"""
import collections
import hashlib
import os
import re
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "source"
OUT = ROOT / "data" / "sqlite" / "food_delivery.sqlite"
TMP = OUT.with_name(OUT.name + ".tmp")

SOURCE_SHA256 = {
    "01_schema.sql": "6a0e6266ea6ebb5c33dbf66afa2da6521981b725943b6f5a7800633f20db12a9",
    "02_data.sql": "4975b09b39e5d38305cd9d421d88ceff318c73f55dafdfd4959c96be930d2f91",
}
EXPECTED_ROWS = collections.OrderedDict([
    ("customers", 40), ("customer_addresses", 56), ("restaurants", 15), ("categories", 8),
    ("menu_items", 108), ("delivery_partners", 12), ("coupons", 10), ("orders", 130),
    ("order_items", 307), ("payments", 130), ("ratings", 60), ("order_tracking", 427),
])
EXPECTED_TRANSFORMS = {
    "drop_database": 1, "create_database": 1, "use_schema": 1, "auto_increment_pk": 12, "enum_to_check": 6,
    "use_data": 1, "fk_checks_lines": 2, "sha2": 40, "menu_items_created_at_nulled": 1,
}
EXPECTED_FK = 15
EXPECTED_INDEXES = 10
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # published test vector

results = []


def check(name, ok, detail=""):
    results.append((bool(ok), name))
    print(("PASS  " if ok else "FAIL  ") + name + (" - " + detail if detail else ""))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- conversion
def convert(schema_src, data_src):
    """Apply the fixed MySQL -> SQLite transformations and return (schema_sql, data_sql, counts)."""
    counts = {}
    s = schema_src
    s, counts["drop_database"] = re.subn(r"(?m)^DROP DATABASE IF EXISTS \w+;[ \t]*\r?\n", "", s)
    s, counts["create_database"] = re.subn(r"CREATE DATABASE \w+\s+CHARACTER SET \w+\s+COLLATE \w+;[ \t]*\r?\n", "", s)
    s, counts["use_schema"] = re.subn(r"(?m)^USE \w+;[ \t]*\r?\n", "", s)
    s, counts["auto_increment_pk"] = re.subn(r"\bINT AUTO_INCREMENT PRIMARY KEY\b", "INTEGER PRIMARY KEY", s)
    s, counts["enum_to_check"] = re.subn(r"\b(\w+)(\s+)ENUM\(([^)]*)\)", r"\1\2TEXT CHECK (\1 IN (\3))", s)

    d = data_src
    d, counts["use_data"] = re.subn(r"(?m)^USE \w+;[ \t]*\r?\n", "", d)
    d, counts["fk_checks_lines"] = re.subn(r"(?m)^SET FOREIGN_KEY_CHECKS\s*=\s*[01];[ \t]*\r?\n", "", d)
    d, counts["sha2"] = re.subn(
        r"SHA2\('([^']*)',\s*256\)",
        lambda m: "'" + hashlib.sha256(m.group(1).encode("utf-8")).hexdigest() + "'",
        d,
    )
    # The source INSERT for menu_items omits created_at, so the schema's DEFAULT CURRENT_TIMESTAMP would stamp the
    # build time and make every build differ. The source gives no value, so store NULL (the DEFAULT stays in the DDL).
    d += "\nUPDATE menu_items SET created_at = NULL;\n"
    counts["menu_items_created_at_nulled"] = 1
    return s, d, counts


# ----------------------------------------------------------------------------- independent parsers (verification)
_TOKEN = re.compile(
    r"\s*(?:(NULL)|SHA2\('((?:[^']|'')*)',\s*(\d+)\)|'((?:[^']|'')*)'|(-?\d+(?:\.\d+)?))\s*([,)])"
)


def parse_source_inserts(text):
    """Parse the ORIGINAL source INSERTs into {table: {'cols': [...], 'rows': [[None|str|Decimal, ...]]}}."""
    tables = collections.OrderedDict()
    cur = None
    for line in text.splitlines():
        m = re.match(r"INSERT INTO (\w+)\s*\(([^)]*)\)\s*VALUES\s*$", line)
        if m:
            cur = m.group(1)
            tables[cur] = {"cols": [c.strip() for c in m.group(2).split(",")], "rows": []}
            continue
        s = line.strip()
        if cur and s.startswith("("):
            pos, row = 1, []
            while True:
                t = _TOKEN.match(s, pos)
                if not t:
                    raise ValueError("cannot parse tuple: " + s[:80])
                null, sha_arg, sha_bits, string, number, sep = t.groups()
                if null:
                    row.append(None)
                elif sha_arg is not None:
                    if sha_bits != "256":
                        raise ValueError("unexpected SHA2 bit length " + sha_bits)
                    row.append(hashlib.sha256(sha_arg.encode("utf-8")).hexdigest())
                elif string is not None:
                    row.append(string.replace("''", "'"))
                else:
                    row.append(Decimal(number))
                pos = t.end()
                if sep == ")":
                    break
            if s[pos:].strip() not in (",", ";"):
                raise ValueError("unexpected tuple tail: " + s[pos:pos + 20])
            tables[cur]["rows"].append(row)
    return tables


def split_top_level(body):
    items, depth, cur, quote = [], 0, [], False
    for ch in body:
        if ch == "'":
            quote = not quote
        if not quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                items.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    if "".join(cur).strip():
        items.append("".join(cur).strip())
    return items


def parse_source_ddl(text):
    """Parse the ORIGINAL source DDL: columns, foreign keys, indexes."""
    text = re.sub(r"(?m)^\s*--.*$", "", text)
    tables, fks, indexes = collections.OrderedDict(), set(), {}
    for m in re.finditer(r"CREATE TABLE (\w+) \((.*?)\n\);", text, re.S):
        name, body = m.group(1), m.group(2)
        cols, checks = [], len(re.findall(r"\bCHECK\b", body))
        for item in split_top_level(body):
            head = item.split(None, 1)[0].upper()
            if head in ("CONSTRAINT", "CHECK", "PRIMARY", "UNIQUE", "FOREIGN"):
                f = re.search(r"FOREIGN KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)(?:\s+ON DELETE\s+(CASCADE|RESTRICT|SET NULL))?", item)
                if f:
                    fks.add((name, f.group(1), f.group(2), f.group(3), f.group(4) or "NO ACTION"))
                continue
            cname, rest = item.split(None, 1)
            ctype = re.match(r"[A-Za-z]+(?:\([^)]*\))?", rest).group(0)
            d = re.search(r"DEFAULT\s+('(?:[^']|'')*'|[A-Za-z_0-9.]+)", rest)
            cols.append({
                "name": cname, "type": ctype.replace(" ", "").upper(),
                "pk": "PRIMARY KEY" in rest.upper(), "notnull": "NOT NULL" in rest.upper(),
                "unique": bool(re.search(r"\bUNIQUE\b", rest)), "default": d.group(1) if d else None,
                "enum": ctype.upper().startswith("ENUM"),
            })
        tables[name] = {"cols": cols, "checks": checks}
    for m in re.finditer(r"CREATE INDEX (\w+)\s+ON (\w+)\(([^)]*)\)", text):
        indexes[m.group(1)] = (m.group(2), tuple(c.strip() for c in m.group(3).split(",")))
    return tables, fks, indexes


def same_value(db_value, expected):
    if expected is None:
        return db_value is None
    if isinstance(expected, str):
        return isinstance(db_value, str) and db_value == expected
    if isinstance(db_value, (int, float)) and not isinstance(db_value, bool):
        return Decimal(repr(db_value)) == expected
    return False


# ----------------------------------------------------------------------------- verification
def verify_built(db_path, schema_src, data_src):
    ddl_tables, ddl_fks, ddl_indexes = parse_source_ddl(schema_src)
    src_rows = parse_source_inserts(data_src)
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    q = lambda sql, args=(): con.execute(sql, args).fetchall()

    # 2. tables
    names = [r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    check("12 user tables with the expected names", names == sorted(EXPECTED_ROWS), f"{len(names)} tables")
    other = q("SELECT type, name FROM sqlite_master WHERE type IN ('view','trigger') OR name LIKE 'sqlite_seq%'")
    check("no views, triggers or sqlite_sequence", not other, str(other))

    # 3. row counts
    counts = {t: q(f"SELECT COUNT(*) FROM {t}")[0][0] for t in EXPECTED_ROWS}
    check("per-table row counts match expected", counts == dict(EXPECTED_ROWS), str(counts))
    check("per-table row counts match the independent source parse",
          counts == {t: len(v["rows"]) for t, v in src_rows.items()})
    total = sum(counts.values())
    check("total rows = 1303", total == 1303, str(total))

    # 4. cell-by-cell comparison with the source
    mismatches, cells, null_cells, pk_bad = [], 0, 0, []
    for t, info in src_rows.items():
        pk = [c["name"] for c in ddl_tables[t]["cols"] if c["pk"]][0]
        ids = [r[0] for r in q(f"SELECT {pk} FROM {t} ORDER BY rowid")]
        if ids != list(range(1, len(info["rows"]) + 1)):
            pk_bad.append(t)
        db_rows = q(f"SELECT {', '.join(info['cols'])} FROM {t} ORDER BY rowid")
        for i, (drow, srow) in enumerate(zip(db_rows, info["rows"]), 1):
            for col, dv, sv in zip(info["cols"], drow, srow):
                cells += 1
                null_cells += sv is None
                if not same_value(dv, sv):
                    mismatches.append((t, i, col, dv, sv))
    check("primary keys are 1..N in source order for every table", not pk_bad, str(pk_bad))
    check("every inserted cell equals the source value", not mismatches, f"{cells} cells compared; first mismatches: {mismatches[:3]}")
    raw_nulls = len(re.findall(r"\bNULL\b", re.sub(r"'(?:[^']|'')*'", "''", re.sub(r"--[^\n]*", "", data_src))))
    check("NULL cells: source tokens = parsed NULLs = NULLs stored", raw_nulls == null_cells, f"raw={raw_nulls} parsed={null_cells}")
    db_nulls = sum(
        q(f"SELECT COUNT(*) FROM {t} WHERE {c} IS NULL")[0][0] for t, i in src_rows.items() for c in i["cols"]
    )
    check("NULL cells stored in the database equal source NULLs", db_nulls == null_cells, f"db={db_nulls}")

    not_null_ts = q("SELECT COUNT(*) FROM menu_items WHERE created_at IS NOT NULL")[0][0]
    check("menu_items.created_at is NULL in every row (the source supplies no value)", not_null_ts == 0, f"{not_null_ts} non-NULL")

    # 5. columns, types, NOT NULL, defaults, PK, UNIQUE, CHECK count
    col_problems = []
    for t, info in ddl_tables.items():
        x = q(f"PRAGMA table_xinfo({t})")  # cid, name, type, notnull, dflt_value, pk, hidden
        if [r[1] for r in x] != [c["name"] for c in info["cols"]]:
            col_problems.append((t, "column names/order"))
            continue
        for r, c in zip(x, info["cols"]):
            want_type = "INTEGER" if c["pk"] else ("TEXT" if c["enum"] else c["type"])
            if r[2].replace(" ", "").upper() != want_type:
                col_problems.append((t, c["name"], "type", r[2], want_type))
            if not c["pk"] and bool(r[3]) != c["notnull"]:
                col_problems.append((t, c["name"], "notnull"))
            if r[4] != c["default"]:
                col_problems.append((t, c["name"], "default", r[4], c["default"]))
            if bool(r[5]) != c["pk"]:
                col_problems.append((t, c["name"], "pk"))
        sql = q("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,))[0][0]
        want_checks = info["checks"] + sum(c["enum"] for c in info["cols"])
        if len(re.findall(r"\bCHECK\b", sql)) != want_checks:
            col_problems.append((t, "CHECK count", want_checks))
    check("columns: names, order, types, NOT NULL, defaults and PKs match the source DDL", not col_problems, str(col_problems[:3]))
    src_unique = {(t, c["name"]) for t, i in ddl_tables.items() for c in i["cols"] if c["unique"]}
    db_unique = set()
    for t in EXPECTED_ROWS:
        for _, iname, uniq, origin, _ in q(f"PRAGMA index_list({t})"):
            if uniq and origin == "u":
                db_unique |= {(t, r[2]) for r in q(f"PRAGMA index_info({iname})")}
    check("UNIQUE columns match the source", src_unique == db_unique, f"{len(db_unique)} unique columns")

    # 6. foreign keys
    db_fks = set()
    for t in EXPECTED_ROWS:
        for r in q(f"PRAGMA foreign_key_list({t})"):  # id, seq, table, from, to, on_update, on_delete, match
            db_fks.add((t, r[3], r[2], r[4], r[6]))
    check("15 foreign keys, identical to the source (columns, parents, ON DELETE)", db_fks == ddl_fks and len(db_fks) == EXPECTED_FK, f"{len(db_fks)} FKs")
    fkc = q("PRAGMA foreign_key_check")
    check("PRAGMA foreign_key_check returns no violations (orphan check)", not fkc, str(fkc[:3]))

    # 7. indexes
    db_idx = {}
    for name, tbl in q("SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex%'"):
        db_idx[name] = (tbl, tuple(r[2] for r in sorted(q(f"PRAGMA index_info({name})"))))
    check("10 named indexes, identical to the source (table, columns)", db_idx == ddl_indexes and len(db_idx) == EXPECTED_INDEXES, f"{len(db_idx)} indexes")

    # 8. integrity
    check("PRAGMA integrity_check = ok", q("PRAGMA integrity_check") == [("ok",)])

    # 9. generated column
    bad = q("SELECT COUNT(*) FROM order_items WHERE subtotal IS NULL OR ABS(subtotal - quantity * price_at_order) > 1e-9")[0][0]
    check("order_items.subtotal = quantity * price_at_order for all rows", bad == 0, f"{bad} bad of {counts['order_items']}")

    # 10. password hashes
    check("SHA-256 test vector for 'abc'", hashlib.sha256(b"abc").hexdigest() == SHA256_ABC)
    hashes = [r[0] for r in q("SELECT password_hash FROM customers ORDER BY rowid")]
    check("40 password hashes are 64-char lowercase hex", len(hashes) == 40 and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes))
    check("password hashes equal sha256('pass1'..'pass40')", hashes == [hashlib.sha256(f"pass{i}".encode()).hexdigest() for i in range(1, 41)])

    # 11. constraints enforce, tested on an in-memory copy only
    mem = sqlite3.connect(":memory:")
    con.backup(mem)
    mem.execute("PRAGMA foreign_keys=ON")
    try:
        mem.execute("UPDATE orders SET order_status='Bogus' WHERE order_id=1")
        check("enum CHECK rejects an invalid value", False)
    except sqlite3.IntegrityError as e:
        check("enum CHECK rejects an invalid value", "CHECK" in str(e), str(e))
    referenced = mem.execute("SELECT customer_id FROM orders ORDER BY order_id LIMIT 1").fetchone()[0]
    try:
        mem.execute("DELETE FROM customers WHERE customer_id=?", (referenced,))
        check("foreign key blocks deleting a referenced parent (FK enforcement on)", False)
    except sqlite3.IntegrityError as e:
        check("foreign key blocks deleting a referenced parent (FK enforcement on)", "FOREIGN KEY" in str(e), str(e))
    mem.close()
    con.close()


def verify_readonly(db_path):
    before = sha256_file(db_path)
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    for label, sql in [("INSERT", "INSERT INTO categories (category_name) VALUES ('zz_test')"),
                       ("CREATE", "CREATE TABLE zz_test (x INTEGER)"),
                       ("DROP", "DROP TABLE categories")]:
        try:
            con.execute(sql)
            check(f"read-only connection refuses {label}", False)
        except sqlite3.OperationalError as e:
            check(f"read-only connection refuses {label}", "readonly" in str(e).lower(), str(e))
    con.close()
    check("database file hash unchanged by the read-only tests", sha256_file(db_path) == before)


# ----------------------------------------------------------------------------- main
def main():
    print(f"python {sys.version.split()[0]}  sqlite library {sqlite3.sqlite_version}")
    for fname, want in SOURCE_SHA256.items():
        got = sha256_file(SRC_DIR / fname)
        check(f"source hash {fname} matches D-13", got == want, got)
    if not all(ok for ok, _ in results):
        sys.exit("source files do not match the pinned hashes; nothing was built")
    for p in (OUT, TMP):
        if p.exists():
            sys.exit(f"refusing to overwrite existing file: {p}")
    with open(SRC_DIR / "01_schema.sql", encoding="ascii", newline="") as f:
        schema_src = f.read()
    with open(SRC_DIR / "02_data.sql", encoding="ascii", newline="") as f:
        data_src = f.read()

    schema_sql, data_sql, counts = convert(schema_src, data_src)
    print("transformations applied:", counts)
    check("transformation counts are exactly as expected", counts == EXPECTED_TRANSFORMS, str(counts))
    check("no MySQL leftovers (SHA2, food_delivery_db, AUTO_INCREMENT, ENUM) in converted SQL",
          not re.search(r"SHA2\(|food_delivery_db|AUTO_INCREMENT|ENUM\(|CREATE DATABASE|\bUSE\b", schema_sql + data_sql))
    if not all(ok for ok, _ in results):
        sys.exit("conversion did not produce the expected result; nothing was built")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(TMP)
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(schema_sql)
    con.executescript("BEGIN;\n" + data_sql + "\nCOMMIT;")
    con.execute("VACUUM")  # rewrite the file from live data only, so freed space cannot keep the pre-UPDATE build-time stamps
    con.close()
    print(f"built temporary database {TMP.name}: {TMP.stat().st_size} bytes")

    verify_built(TMP, schema_src, data_src)
    if not all(ok for ok, _ in results):
        print(f"\nVERIFICATION FAILED - temporary file left for inspection: {TMP}")
        sys.exit(1)

    os.replace(TMP, OUT)
    print(f"\nrenamed to final path: {OUT}")
    verify_readonly(OUT)
    passed = sum(ok for ok, _ in results)
    print(f"\n{passed}/{len(results)} checks passed")
    print(f"database: {OUT}  size {OUT.stat().st_size} bytes  sha256 {sha256_file(OUT)}")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
