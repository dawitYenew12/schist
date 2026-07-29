# schist

`schist` is a dependency-free columnar storage engine written in Rust. A
`schist` database is a single self-describing binary file — a `.sht` container —
that holds a table's schema, its columnar data pages, its dictionary pages, its
secondary indexes, and the bookkeeping structures (free-space map, row-id map,
zone maps) that make the whole thing navigable. The crate decodes such a file,
runs a cross-page consistency verifier over it, and then executes an *operation
script* — a small imperative language of inserts, updates, deletes, scans,
index scans, joins, compaction, and checkpoints — against the decoded database.

This README is the reference for the engine: the on-disk format, the in-memory
subsystems and the invariants each one maintains, the operation-script grammar,
and the fuzzing harnesses. It is deliberately long and detailed, because the
engine is large and the interesting behavior lives in the interaction between
subsystems rather than in any single file.

- [1. Overview](#1-overview)
- [2. Design goals and non-goals](#2-design-goals-and-non-goals)
- [3. Repository layout](#3-repository-layout)
- [4. The `.sht` container format](#4-the-sht-container-format)
- [5. Data model: schema, types, encodings](#5-data-model-schema-types-encodings)
- [6. The page buffer manager (`pager`)](#6-the-page-buffer-manager-pager)
- [7. Dictionary pages (`dict`)](#7-dictionary-pages-dict)
- [8. The row-id map (`rowid`)](#8-the-row-id-map-rowid)
- [9. The free-space map (`fsm`)](#9-the-free-space-map-fsm)
- [10. Zone maps (`zonemap`)](#10-zone-maps-zonemap)
- [11. Secondary indexes (`index`)](#11-secondary-indexes-index)
- [12. Mutation: insert, update, delete (`mutation`)](#12-mutation-insert-update-delete-mutation)
- [13. Compaction (`compact`)](#13-compaction-compact)
- [14. Verification (`verify`)](#14-verification-verify)
- [15. Query execution (`query`)](#15-query-execution-query)
- [16. Vectorized execution (`vexec`)](#16-vectorized-execution-vexec)
- [17. Transactions, MVCC, and the write-ahead log](#17-transactions-mvcc-and-the-write-ahead-log)
- [18. The operation script language (`script`)](#18-the-operation-script-language-script)
- [19. The SQL front end (`sql`, `logical`, `cost`, `plan`)](#19-the-sql-front-end-sql-logical-cost-plan)
- [20. Auxiliary subsystems](#20-auxiliary-subsystems)
- [21. Fuzzing harnesses](#21-fuzzing-harnesses)
- [22. Building and testing](#22-building-and-testing)
- [23. Invariant reference](#23-invariant-reference)
- [24. Glossary](#24-glossary)

---

## 1. Overview

A columnar store lays a table out one column at a time rather than one row at a
time. Each column's values for a range of rows are stored contiguously, which
makes analytical scans — "sum this column", "filter on that column" — touch only
the columns a query mentions, and makes the values in a column compress well
because they are all the same type and often repetitive.

`schist` is built around that idea, but it is a *storage engine*, not just a file
format: it maintains the mutable structures that let a columnar table be queried
and modified over its lifetime. The moving parts are:

- A **page buffer manager** that owns the raw bytes of every page and hands out
  typed views into them. This is the one subsystem that uses `unsafe` to work
  with raw pointers, and everything else is layered on top of its accessors.
- A **container decoder** that turns a `.sht` byte blob into the in-memory
  database (schema, pages, indexes, maps) and a **serializer** that writes the
  database back out for checkpoints.
- A **schema and encoding layer** describing the columns, their types, and how
  each column's values are physically laid out (plain, dictionary, or
  run-length encoded).
- A **verifier** that checks cross-page consistency after a decode, so that
  malformed input is rejected before any operation runs against it.
- A set of **bookkeeping structures** — the free-space map, the row-id map, and
  per-page zone maps — that track where free space is, where each row lives, and
  the min/max summary of each page.
- **Secondary indexes** over individual columns, with a fast path that resolves
  dictionary-encoded values without walking the dictionary page from scratch.
- A **mutation layer** (insert / update / delete) that drives all of the above:
  it allocates slots, splits and rewrites pages, maintains indexes, and updates
  the maps.
- A **compaction layer** that reclaims space by splitting overfull pages,
  merging underfull ones, and repacking run-length-encoded pages in place.
- A **query layer** that scans columns (in each encoding), pushes predicates
  down, resolves values through indexes, joins two inputs, and aggregates.
- A **vectorized execution engine** that runs the same operators in a
  batch-at-a-time form over column vectors.
- A **transaction manager**, an **MVCC** version-chain layer, and a
  **write-ahead log** for durability and isolation.

On top of all of that sits an **operation script** interpreter that the fuzzing
harness drives, and a **SQL front end** (lexer, parser, logical planner, cost
model) that lowers queries into plans.

The canonical way untrusted bytes enter the system is the `run_combined`
entry point, which the primary fuzz harness calls:

```rust
pub fn run_combined(data: &[u8]) {
    let (script_bytes, db_bytes) = split_combined(data);
    let db = match format::decode(db_bytes) { Ok(db) => db, Err(_) => return };
    if verify::verify(&db).is_err() { return; }
    let src = match std::str::from_utf8(script_bytes) { Ok(s) => s, Err(_) => return };
    let mut db = db;
    let _ = script::run_script(&mut db, src);
}
```

The combined blob is `[u32 little-endian script length][script bytes][.sht
database bytes]`. The function decodes the database, verifies it, then runs the
script against it. Every stage is best-effort: a decode or verify failure stops
before the script runs, and a malformed statement is skipped rather than
aborting the whole run. This `decode → verify → run` pipeline is the reachable
code path that matters, and it is long: a single script statement can flow
through the mutation layer, the dictionary, the indexes, the compaction routines,
the free-space and row-id maps, and the page buffer manager before it returns.

## 2. Design goals and non-goals

**Goals.**

- *Self-describing files.* A `.sht` file carries everything needed to interpret
  it. There is no external catalog required to read a database back.
- *Columnar layout with real encodings.* Columns are stored plain, dictionary
  encoded, or run-length encoded, and scans understand each encoding directly
  rather than materializing a row representation first.
- *Mutability.* The engine is not a write-once format. Rows can be inserted,
  updated, and deleted, and the physical layout is maintained (pages split,
  merge, and repack) as that happens.
- *A single, auditable unsafe core.* Raw pointer work is confined to the page
  buffer manager. Every other subsystem uses safe accessors, so the memory model
  is easy to reason about in one place.
- *Determinism.* The build has no network or filesystem dependencies, the
  decoder is deterministic, and the random-number generators used internally
  (skip-list heights, reservoir sampling) are seeded from fixed constants.

**Non-goals.**

- *Multiple tables per file.* One `.sht` file is one table. The `catalog`
  subsystem models multiple objects at the metadata level, but the on-disk
  container is single-table.
- *SQL completeness.* The `sql` module parses a practical subset of SQL and
  lowers it to logical plans, but the engine is exercised mainly through the
  lower-level operation script. SQL is a front end, not the whole story.
- *Concurrency.* The engine is single-threaded. The transaction manager models
  isolation and the wait-for graph, but the actual execution is serial.
- *Crash-perfect durability.* The write-ahead log and checkpoint serializer
  model durability, but this is a study engine, not a production database.

## 3. Repository layout

The source is organized as one crate with many modules. The modules fall into a
few groups.

**The core storage pipeline** (the code the canonical harness drives):

| Module | Responsibility |
| ------ | -------------- |
| `pager` | Page buffer manager; the only `unsafe` core. |
| `format` | `.sht` container decode and encode. |
| `schema` | Columns, types, encodings, the schema record. |
| `value` | The runtime `Value` type and its total ordering. |
| `verify` | Cross-page consistency checking after decode. |
| `dict` | Dictionary pages and their generation counter. |
| `rowid` | Row-id map: row id → (page, slot) plus tombstones. |
| `fsm` | Free-space map: free slots, free pages, free-page list. |
| `zonemap` | Per-page min/max summaries for scan pruning. |
| `index` | Secondary indexes with a dictionary fast path. |
| `mutation` | Insert / update / delete; drives all of the above. |
| `compact` | Page split, underfull merge, in-place RLE repack. |
| `query` | Scans, predicate pushdown, index scans, joins, aggregation. |
| `script` | The operation-script lexer, parser, and interpreter. |

**Encoding and column machinery:**

| Module | Responsibility |
| ------ | -------------- |
| `encoding` | Bit-packing, frame-of-reference, delta, delta-of-delta, null suppression, constant, zig-zag, byte-aligned dictionary. |
| `rle` | Run-length-encoded column pages. |
| `array` | Typed in-memory columnar arrays and record batches. |
| `frontcode` | Front-coded sorted string dictionary blocks. |
| `compress` | Byte-oriented block codecs (store, RLE, LZ77) with framing. |
| `batchio` | Self-describing serialization of record batches. |

**Query and execution:**

| Module | Responsibility |
| ------ | -------------- |
| `vexec` | Vectorized batch execution engine. |
| `physical` | Volcano-style pull-based physical operators. |
| `exprvm` | Expression bytecode compiler and stack evaluator. |
| `expr` | Expression AST, parser, and tree-walking evaluator. |
| `scalar` | Scalar function registry. |
| `text` | String scalar functions. |
| `datefn` | Date/timestamp scalar functions. |
| `join` | Hash / nested-loop / sort-merge / semi / anti / cross joins. |
| `sort` | Comparators, top-N, external merge sort, group partitions. |
| `window` | Window functions (row_number, rank, lag/lead, running aggregates). |
| `plan` | Logical-to-physical planning with cost estimates. |
| `logical` | Logical plan lowering and a rule-based optimizer. |
| `cost` | Cardinality and cost model. |
| `sql` | SQL lexer, Pratt parser, and statement/expression AST. |

**Transactions and durability:**

| Module | Responsibility |
| ------ | -------------- |
| `txn` | MVCC transaction manager, version chains, GC, deadlock detection. |
| `wal` | Write-ahead log with segmented records and checksums. |
| `snapshot` | Copy-on-write page versioning for time-travel reads. |
| `bufferpool` | Fixed-capacity buffer pool with clock eviction. |

**Statistics and estimation:**

| Module | Responsibility |
| ------ | -------------- |
| `stats` | Column statistics and histograms. |
| `analyze` | Column profiling (`ANALYZE`). |
| `hll` | HyperLogLog distinct-count estimator. |
| `quantile` | Reservoir sampling and quantile sketches. |
| `welford` | Streaming variance/covariance/correlation. |
| `topk` | Space-saving heavy hitters. |
| `minhash` | MinHash Jaccard similarity. |

**General-purpose data structures and utilities:**

| Module | Responsibility |
| ------ | -------------- |
| `bitmap` | WAH-compressed bitmap index. |
| `bitset` | Growable dense bit set. |
| `bitmatrix` | Dense bit matrix with transitive closure. |
| `roaring` | Roaring compressed bitmap. |
| `btree` | Paged B+tree secondary index. |
| `hashindex` | Paged hash index with overflow chains. |
| `skiplist` | Probabilistic skip-list ordered map. |
| `radix` | Compressed radix trie. |
| `arena` | Generational arena allocator. |
| `slab` | Slab allocator with reusable keys. |
| `lru` | Bounded LRU cache. |
| `pqueue` | Keyed priority queue with decrease-key. |
| `ringbuf` | Fixed-capacity ring buffer. |
| `interval` | Augmented interval tree. |
| `rangeset` | Disjoint integer range set. |
| `graph` | Directed-graph algorithms (topo sort, SCC). |
| `dsu` | Disjoint-set / union-find. |
| `kmerge` | K-way merge via a loser tree. |
| `radixsort` | LSD radix sort. |
| `partition` | Hash partitioning. |
| `catalog` | System catalog of tables, indexes, sequences, views. |
| `session` | Prepared statements and session settings. |
| `wire` | Binary request/response protocol codec. |
| `pretty` | Result-set table rendering. |
| `json` | JSON parser and serializer. |
| `csvio` | CSV import/export with type inference. |
| `codec` | Hex and Base64 codecs. |
| `checksum` | CRC-32, Adler-32, FxHash. |
| `bloom` | Bloom and counting-bloom filters. |
| `fingerprint` | Rabin-Karp rolling hashes and content-defined chunking. |
| `varint` | LEB128 varints and a bit reader/writer. |
| `decimal` | Fixed-point decimal arithmetic. |
| `temporal` | Calendar date and timestamp types. |
| `pattern` | `LIKE`, glob, and small-regex matching. |
| `editdist` | Levenshtein / Damerau-Levenshtein distance. |
| `intern` | String interner. |
| `diag` | Diagnostics and pretty-printing. |

Every module has unit tests; the whole suite runs with `cargo test`.

---

## 4. The `.sht` container format

A `.sht` file is a directory of records followed by the record bodies. The
layout is:

```text
  [8 bytes  magic "SCHIST1\x01"]
  [u32      format version]
  [u32      record count N]
  N × directory entries, each 17 bytes:
      [u8   kind]
      [u32  id]        page id for page records; 0 for singletons
      [u32  off]       body offset, relative to the start of the body region
      [u32  len]       body length in bytes
      [u32  checksum]  additive checksum over the body
  ... record bodies, concatenated in directory order ...
```

The magic string identifies the format and version. The directory is a flat
array of fixed-width entries so the decoder can find every record's body without
parsing anything else first. The `off` field is relative to the first byte after
the directory, and the bodies are concatenated in the same order as the
directory entries.

### 4.1 Record kinds

| kind | name | id meaning | body |
| ---- | ---- | ---------- | ---- |
| 0 | schema | 0 | the encoded schema record |
| 1 | data | page id | a data-page body (see below) |
| 2 | dict | page id | a dictionary-page buffer |
| 3 | index | column index | an encoded secondary index |
| 4 | fsm | 0 | the encoded free-space map |
| 5 | rowid | 0 | the encoded row-id map |
| 6 | zonemap | 0 | the encoded zone-map array |
| 0xfe | next-row-id | 0 | the 8-byte next row id counter |

There is exactly one schema record, one fsm record, one rowid record, one
zonemap record, and one next-row-id record. There are zero or more data pages,
zero or more dictionary pages, and zero or more index records (one per indexed
column).

### 4.2 The data-page body

A data record's body wraps the columnar chunk together with its slot directory:

```text
  [u32 body_len]
  [body_len bytes: the columnar chunk]
  [u32 slot_count]
  slot_count × slot entries
```

The columnar chunk is:

```text
  [u32 num_rows]
  [u32 num_cols]
  for each column:
      [u8  kind]        column storage class (bool/int/real/text)
      [u8  encoding]    plain / dictionary / rle
      [u32 data_len]
      [data_len bytes]  the column's encoded values for this page
```

Each column stores its own `data_len` bytes, so the decoder can walk column by
column. The `kind` and `encoding` bytes are checked against the schema during
verification.

A slot entry describes one physical row slot on the page:

```text
  [u64 row_id]   the logical row id occupying this slot
  [u32 off]      byte offset of the row's cells within the page body
  [u32 len]      length of the row's cells
  [u8  live]     1 if the slot holds a live row, 0 if tombstoned
```

The slot directory is what turns a page's packed bytes into addressable rows.
The row-id map (kind 5) points at slots by `(page id, slot index)`, and a slot's
`off`/`len` locate the row's cells within the page body. Live/tombstoned status
lets a delete mark a row dead without immediately rewriting the page.

### 4.3 Checksums

Every directory entry carries an additive checksum over its body. The decoder
recomputes the checksum as it reads each body and rejects the file if any body
does not match. This catches truncation and bit-rot at the record granularity
before any structural interpretation happens. The checksum is intentionally
simple (an additive fold); the `checksum` module also provides CRC-32 and
Adler-32 for callers that want stronger integrity, but the container uses the
additive form for speed and because it is only a first-line guard ahead of the
structural verifier.

### 4.4 Encoding a database

The serializer (`format::encode_database`) is the inverse of the decoder. It
walks the in-memory database and emits, in order: the schema record; a dict
record for every column that has a dictionary page; a data record for every data
page (body plus slot directory); an index record for every indexed column; the
fsm, rowid, and zonemap records; and finally the next-row-id counter. Directory
entries are laid out first with their offsets computed, then the bodies are
appended. Encoding then decoding a database round-trips it exactly, which the
checkpoint path relies on.

### 4.5 What the decoder validates inline

The decoder performs the structural checks needed to build the in-memory
structures safely: the magic and version match; the directory fits within the
blob; each body's `off`/`len` stay within bounds; each body's checksum matches;
the schema decodes to a consistent column list; each data page's column count
matches the schema; encodings are recognized; and slot entries fit within the
page. Anything the decoder cannot check locally — relationships *between*
records — is left to the verifier (section 14). The division of labor is
deliberate: the decoder guarantees each record is individually well-formed, and
the verifier guarantees the records are mutually consistent.

---
## 5. Data model: schema, types, encodings

### 5.1 The schema

A `schist` database has exactly one schema: a fixed, ordered list of columns.
The first column is always the **row-id column** — implicitly an integer with
plain encoding and not nullable — and it is what the row-id map and the secondary
indexes key off. The remaining columns are user columns.

Each column carries a name, a storage class (`ColKind`), an encoding, and a
nullability flag. The schema is serialized as record kind 0 and decoded back
into a `Schema` value. The verifier checks that every data page's column count
and per-column `kind` match the schema.

### 5.2 Value types

The runtime value type is `Value`, a small enum:

| Variant | Meaning |
| ------- | ------- |
| `Null` | the absence of a value |
| `Bool(bool)` | a boolean |
| `Int(i64)` | a 64-bit signed integer |
| `Real(f64)` | a 64-bit float |
| `Text(u32)` | a dictionary id referencing a string in a dictionary page |

A `Text` value is not a string — it is the integer id of a string that lives in
the column's dictionary page. Two `Text` ids are equal if and only if their
dictionary entries are equal, which holds within a single dictionary-page
generation. Comparing two `Text` ids compares their dictionary positions.

`Value` defines a total order used throughout the query engine: null sorts
first, then values compare within their type, with a cross-type fallback by type
tag that keeps the order total. Integer and real values compare numerically
against each other.

The `ColKind` storage classes and their plain widths are:

| ColKind | plain width | notes |
| ------- | ----------- | ----- |
| `Bool` | 1 byte | |
| `Int` | 8 bytes | |
| `Real` | 8 bytes | IEEE-754 double |
| `Text` | 4 bytes | a dictionary id |

### 5.3 Encodings

A column's `Encoding` decides how its values are physically laid out inside a
data page:

- **Plain.** One fixed-width cell per slot, back to back. The width comes from
  the column's `ColKind`. This is the simplest layout and the default.
- **Dictionary.** The column stores dictionary ids, and the ids index into the
  column's dictionary pages. Two ids are equal iff their dictionary entries are
  equal. Dictionary encoding is used for low-cardinality text columns, where
  storing a 4-byte id per row is far cheaper than storing the string per row.
- **Rle (run-length).** A column page is a sequence of `(value, count)` runs.
  Text RLE runs store dictionary ids. Run-length encoding is used for columns
  with long stretches of repeated values.

The `encoding` module implements the lower-level integer codecs that these
column encodings build on: bit-packing, frame-of-reference, delta,
delta-of-delta, null suppression, constant folding, zig-zag, and a byte-aligned
dictionary codec. The `rle` module implements the run-length page layout, and
the `frontcode` module implements front-coded string dictionary blocks for the
sorted-string case.

### 5.4 Null handling

Nullability is a per-column property. Columns store a null bitmap alongside their
values so a null is distinguishable from a zero or an empty string. The `array`
module's `Validity` type is the in-memory form of that bitmap: one bit per slot,
`true` meaning valid, with a lazily materialized backing so an all-valid column
costs nothing.

---

## 6. The page buffer manager (`pager`)

The `pager` module is the foundation everything else stands on, and it is the
only place in the crate that works with raw memory through `unsafe`.

### 6.1 Pages and slots

A **page** is a fixed region of bytes plus a slot directory. The page buffer
holds the packed column data; the slot directory maps slot indices to
`(row_id, offset, length, live)` tuples that locate each row's cells within the
buffer. Pages come in kinds — data pages hold rows, dictionary pages hold
strings — and the pager tracks them together.

The pager exposes accessors that return views into a page's buffer:

- `page_ptr` / `raw_ptr` — the base pointer of a page's buffer.
- `read_at(off)` — read bytes at an offset within a page.
- `slot_at(off)` — read a slot entry.
- `free_page` / `evict` — return a page to the free list.

These accessors are the trusted primitives that the index, scan, mutation, and
compaction code call while holding offsets and pointers they computed from
higher-level structures. Because they are used pervasively, the presence of
`unsafe` inside the pager does not by itself localize where a memory-safety
problem could originate: the pager's accessors are correct *given valid inputs*,
and the responsibility for supplying valid inputs is spread across the callers
that maintain the offsets, generations, and slot indices those accessors consume.

### 6.2 The database handle

The `Database` type (re-exported at the crate root) bundles the pager with the
schema, the indexes, the free-space map, the row-id map, the zone maps, and the
next-row-id counter. It is the object the script interpreter mutates and the
query layer reads. A decoded `.sht` file becomes a `Database`; encoding a
`Database` produces a `.sht` file.

### 6.3 The buffer pool

Sitting conceptually in front of a backing store is the `bufferpool` module: a
fixed-capacity cache of page frames with clock (second-chance) eviction, pin/
unpin reference counting, and dirty-page tracking. It is a textbook design — a
frame table, a page table, a free list, and a clock hand that sweeps reference
bits to choose a victim, with pinned frames never evicted and dirty victims
reported to the caller for write-back. The core engine owns its pages directly
through the pager; the buffer pool is the caching layer used where a bounded
working set over a larger store is wanted.

---

## 7. Dictionary pages (`dict`)

A dictionary page holds the distinct strings of a dictionary-encoded (or
RLE-text) column, addressed by integer id. A `Text(id)` value is resolved to its
bytes by looking `id` up in the column's dictionary page.

### 7.1 Layout and generations

A dictionary page stores its strings in a buffer and maintains an id → offset
mapping so a lookup is a direct index rather than a scan. As a column grows, its
dictionary accumulates new distinct values, and the page may need to be
**rewritten** — grown to a larger buffer, split, or repacked — to accommodate
them. Each dictionary page carries a **generation counter** that is advanced
whenever the page is rewritten. The generation is a version stamp on the page's
current physical layout: an id resolved against one generation of a page refers
to a specific byte range, and after the page is rewritten to a new generation
the byte ranges may differ.

The generation counter exists so that consumers which cache information derived
from a dictionary page's layout can record which generation their cached
information was computed against. The dictionary reports its current generation
through its accessors, and rewrites advance it.

### 7.2 Resolving values

The straightforward way to resolve a `Text(id)` is to look the id up in the
current dictionary page. The secondary index (section 11) additionally keeps a
faster path that avoids repeating that lookup for values it has already located;
that fast path is described with the index, because it is a property of the
index's cache rather than of the dictionary itself.

---

## 8. The row-id map (`rowid`)

The row-id map answers "where does row *r* live?" It maps a logical row id to a
`(page id, slot index)` pair, and it also holds a **deletion (tombstone)
bitmap** marking which row ids have been deleted.

### 8.1 Role

Every live row has exactly one entry in the row-id map, pointing at the slot that
currently holds it. The map is consulted whenever an operation needs to reach a
row by id rather than by scanning: a point lookup by id, an index probe that
yields a row id and must then fetch the row's cells, an update that must locate
the row to overwrite. Callers that reach a row through the row-id map take the
`(page, slot)` it returns and hand the slot to the pager's accessors to read or
write the row's cells.

### 8.2 Invariant

The row-id map is trusted, throughout the engine, to point every live row id at a
valid, live slot on the page it names. Code that resolves a row through the map
does not re-validate the slot index against the page's current slot directory
before using it — the map's correctness is assumed. Keeping the map in agreement
with the physical slot layout as pages change over their lifetime is therefore
the responsibility of every operation that moves rows between or within pages.

### 8.3 Tombstones

A delete marks a row's slot `live = 0` and sets the row id's bit in the deletion
bitmap, rather than immediately removing the entry and rewriting the page.
Tombstoned rows are skipped by scans and are eventually reclaimed by compaction.
The next-row-id counter (record kind 0xfe) hands out fresh ids and never reuses a
tombstoned id within a file's lifetime.

---

## 9. The free-space map (`fsm`)

The free-space map tracks reusable space so inserts and page allocations do not
have to grow the file unnecessarily. It records, per page, how much free space
remains and which slots are free, and it maintains a **free-page list** of pages
that have been fully vacated and can be handed out again.

When a delete tombstones a row, the slot becomes reclaimable; when compaction
frees a page, it goes on the free-page list. An insert consults the free-space
map to find a page with room before allocating a new page. The verifier checks
that the free-space map's totals are consistent with the pages actually present
(section 14) — a page on the free list must not also hold live rows, and the free
counts must not exceed the pages' capacities.

---

## 10. Zone maps (`zonemap`)

A zone map is a per-page summary: the minimum and maximum value of each column on
that page, plus a liveness flag. Zone maps let a scan *prune* pages: if a
predicate is `x > 100` and a page's zone map says its maximum `x` is 50, the scan
skips the page without reading it.

Each data page has a corresponding zone map, stored together in the zonemap
record (kind 6). Mutations keep the zone map's min/max in step with the values
actually on the page — an insert widens the range if the new value falls outside
it; a full-page rewrite recomputes the range. The verifier checks that a page's
zone map is present and that its liveness flag agrees with whether the page holds
live rows. The `interval` and `rangeset` modules provide the range machinery a
more elaborate pruner would use to combine a predicate's range domain with the
per-page ranges.

---

## 11. Secondary indexes (`index`)

A secondary index maps the values of one column to the row ids that hold them, so
an equality predicate on that column can be answered by an index probe instead of
a full scan.

### 11.1 Structure

An index over a column is a set of entries, one per distinct indexed value, each
holding the value together with the list of row ids that carry it. For a
dictionary-encoded text column the indexed value is a dictionary id, and the
index's entries are keyed by that id. The `btree`, `hashindex`, `bitmap`, and
`roaring` modules provide alternative index structures (ordered, hashed, and
bitmap forms); the primary `index` module is the one wired into the mutation and
query paths.

### 11.2 The dictionary fast path

Resolving an indexed dictionary value to its bytes could be done by looking the
dictionary id up in the dictionary page on every probe. To avoid repeating that
work, each index entry additionally caches a direct reference into the dictionary
page's buffer for the value it indexes — the location within the page buffer
where that value's bytes sit — together with the dictionary page's generation at
the time the reference was captured. A probe that hits such an entry can read the
value's bytes through the cached reference rather than walking the dictionary
from scratch.

The cached reference and the recorded generation are stored on the index entry.
The index serializes these as a byte offset within the dictionary page (so a
freshly decoded index can re-capture the reference against the freshly decoded
dictionary page) plus the generation stamp. Because the fast path reads through a
reference the index holds into another subsystem's buffer, the correctness of a
probe depends on the relationship between the index entry's cached state and the
dictionary page it refers into — a relationship that spans the index, the
dictionary, and the free-space map that governs the lifetime of dictionary-page
buffers.

### 11.3 Index maintenance

Inserts add a row id to the entry for the inserted value (creating the entry if
the value is new). Updates move a row id from the old value's entry to the new
value's entry. Deletes remove the row id from its entry. The verifier checks that
every row id an index names exists and is live, and that the index covers the
column it claims to (section 14).

---

## 12. Mutation: insert, update, delete (`mutation`)

The mutation layer is where a script statement's effect is actually carried out.
It is the busiest subsystem because a single mutation touches many others.

### 12.1 Insert

An insert appends a row in schema (positional) order. It:

1. Allocates a row id from the next-row-id counter.
2. Finds a data page with room via the free-space map, or allocates a new page
   from the free-page list (or grows the file).
3. For each column, encodes the cell according to the column's encoding —
   interning a text value into the column's dictionary page if the column is
   dictionary or RLE text, which may **extend** the dictionary.
4. Writes the row's cells into the page buffer and appends a slot-directory
   entry pointing at them.
5. Records the row id → (page, slot) mapping in the row-id map.
6. Adds the row id to each relevant secondary index entry.
7. Widens the page's zone map if the new values fall outside its current range.

Interning a new distinct text value can grow a dictionary page past its current
capacity, which triggers a dictionary rewrite (section 7) and advances the page's
generation.

### 12.2 Update

An update overwrites a column's value for the rows matching a predicate. It
locates each matching row (through the row-id map or a scan), rewrites the cell,
moves the row id between index entries if the indexed column changed, and updates
the zone map. If the new value is a text value not yet present in the column's
dictionary, the update interns it, which — as with insert — can force a
dictionary rewrite and advance the generation.

### 12.3 Delete

A delete tombstones the rows matching a predicate: it sets each slot's `live`
flag to 0, sets the row id's bit in the deletion bitmap, removes the row id from
its index entries, and updates the free-space map to reflect the reclaimable
slot. The row's bytes stay on the page until compaction reclaims them.

### 12.4 Fast row access

Reading a row's cells given its `(page, slot)` goes through a fast accessor
(`read_row_cell_fast`) that indexes the page's slot directory and reads the
cell bytes at the slot's offset through the pager. This accessor is on the hot
path for point lookups and index probes; it trusts the slot index it is given to
be in range for the page's current slot directory, which is the invariant the
row-id map is responsible for upholding (section 8).

---

## 13. Compaction (`compact`)

Over time a table accumulates tombstoned rows and unevenly filled pages.
Compaction reclaims that space. It has three jobs.

### 13.1 Page split

When a page grows overfull, compaction splits it into two pages, distributing the
live rows between them, allocating a new page from the free-space map, and
updating the row-id map so each moved row points at its new home. The zone maps
of both resulting pages are recomputed.

### 13.2 Underfull merge

When two adjacent pages are each underfull, compaction merges their live rows
onto one page and returns the other to the free-page list. As with a split, the
row-id map entries for moved rows are updated to point at their new page and slot.

### 13.3 In-place RLE repack

A run-length-encoded page can become fragmented: after updates and deletes its
runs no longer coalesce, and tombstoned rows leave gaps. The **RLE repack**
rewrites the page's runs in place — recomputing the `(value, count)` runs from
the live rows, compacting the live rows to the front of the page, and rewriting
the slot directory so slot indices are dense again. Repacking shifts where each
surviving row's slot sits within the page: a row that occupied slot 7 before the
repack may occupy slot 2 after it, because the tombstoned rows ahead of it were
removed and the live rows were renumbered from the front.

Repack is an in-place operation on a single page: it does not allocate a new
page, and it works directly on the page's buffer and slot directory through the
pager. It is the space-reclamation counterpart to split and merge, applied to the
RLE encoding specifically. The order in which repack rewrites runs, renumbers
slots, and touches the surrounding bookkeeping is what determines the page's — and
the rest of the engine's — view of where each live row lives after compaction.

---
## 14. Verification (`verify`)

The verifier runs after a successful decode and before any operation executes. Its
job is to reject a structurally-decodable file that is nonetheless internally
inconsistent, so that the operation layers can assume a coherent starting state.

The verifier checks cross-record consistency that the decoder cannot check while
reading a single record in isolation:

- **Page accounting.** The number and ids of data pages are consistent with the
  slot directories and the row-id map; every page referenced is present.
- **Schema agreement.** Every data page's column count equals the schema's, and
  each column's stored `kind` and `encoding` match the schema's declaration.
- **Dictionary ranges.** Every dictionary id referenced by a data page or an
  index is within the range of the referenced dictionary page.
- **Index coverage.** Each index record names the column it indexes, and every
  row id an index entry lists exists in the row-id map and is live; the index
  covers the column it claims to.
- **Free-space-map totals.** The free counts and the free-page list are
  consistent with the pages present: a page on the free list holds no live rows,
  and the free counts do not exceed capacities.
- **Zone-map liveness.** Every data page has a zone map whose liveness flag
  agrees with whether the page holds live rows, and whose min/max bracket the
  values present.
- **Row-id bounds.** Every row-id-map entry names a page that exists and a slot
  index within that page's slot directory at decode time.

The verifier validates the state of a database *as decoded* — the static,
decode-time relationships among the records in the file. It is not a runtime
monitor: relationships that only come into existence as operations run, and that
depend on the order in which subsystems are updated during a mutation or a
compaction, are outside a decode-time verifier's remit by construction. The
verifier establishes a coherent starting point; maintaining coherence as
operations compose is the job of the operation layers themselves.

A file that fails verification is rejected: `run_combined` returns without running
the script. This is why the fuzzer's crafted inputs must be valid enough to pass
verification — an inconsistency the verifier catches never reaches the operation
layers.

---

## 15. Query execution (`query`)

The `query` module reads a decoded database. Its operators are:

### 15.1 Scans

A **scan** walks a table's data pages and yields live rows. It understands each
column encoding directly:

- A **plain** scan reads fixed-width cells at slot offsets.
- A **dictionary** scan reads dictionary ids and resolves them against the
  column's dictionary page when the value's bytes are needed.
- An **RLE** scan walks `(value, count)` runs, expanding them to per-row values.

Scans skip tombstoned slots (via the `live` flag) and can prune whole pages using
zone maps when a predicate is present.

### 15.2 Predicate pushdown

A `scan where <col> <op> <v>` pushes the predicate into the scan so that only
matching rows are produced. Integer, real, boolean, and text (dictionary-id)
comparisons are supported, using the `Value` total order. Zone maps prune pages
whose min/max cannot satisfy the predicate before their rows are read.

### 15.3 Index scans

An `index_scan <col> <v>` answers an equality predicate through the column's
secondary index instead of scanning. It probes the index for the value, obtains
the list of row ids, and fetches each row through the row-id map and the pager's
fast accessor. For a dictionary-encoded column, the probe resolves the value
through the index's dictionary fast path (section 11.2).

### 15.4 Joins

A `join` combines two inputs on a key. The `join` module implements hash join
(build a hash table on one side, probe with the other), nested-loop join,
sort-merge join, and the semi/anti/cross variants. The planner (`plan`) chooses
an algorithm from cardinality estimates; the sort-merge path uses the `kmerge`
loser-tree merge and the `sort` module's external merge sort for inputs that do
not fit in memory.

### 15.5 Aggregation

An `agg <col>` computes aggregates over a column. Count, sum, min, and max are
supported directly; the `welford` module supplies numerically-stable variance,
covariance, and correlation, and the `hll`, `quantile`, and `topk` modules supply
approximate distinct counts, quantiles, and heavy hitters for the statistics
path.

---

## 16. Vectorized execution (`vexec`)

Alongside the row-at-a-time query path, `vexec` implements a **vectorized** engine
that processes a batch of rows at a time. Its central type is the column vector: a
contiguous, type-homogeneous buffer with a validity bitmap, mirroring the `array`
module's typed arrays. Operators — filter, project, hash aggregate, join —
consume and produce column vectors, so the per-row interpreter overhead is paid
once per batch rather than once per row.

The vectorized operators compose into a pipeline: a scan produces batches, each
batch flows through the filter and projection operators, and blocking operators
(aggregate, sort, join build) accumulate batches before emitting their results.
The `physical` module provides the pull-based (Volcano) counterpart, where each
operator implements a `next()` that pulls a row from its child; `physical` reuses
the `exprvm` compiled-expression path so predicate and projection evaluation is
the flat-bytecode form rather than a tree walk.

The `exprvm` module compiles an expression tree into a stack-machine program once
and then evaluates that program per row, reading inputs from a row slice by column
index. This removes the pointer chasing of tree-walking evaluation on hot paths.
The `expr` module is the tree-walking evaluator and parser used where a compiled
program is not warranted.

---

## 17. Transactions, MVCC, and the write-ahead log

### 17.1 The transaction manager (`txn`)

The `txn` module models transactions with snapshot isolation. It supports begin /
commit / rollback, per-transaction snapshots, version chains for updated rows,
garbage collection of versions no live snapshot can see, deadlock detection over
a wait-for graph, and savepoints. Execution is serial, so the manager models the
isolation semantics rather than enforcing them against real concurrency, but the
version-chain and GC logic is real: a reader sees the version of a row current as
of its snapshot, and GC reclaims versions older than the oldest live snapshot.

### 17.2 Copy-on-write snapshots (`snapshot`)

The `snapshot` module provides the page-level counterpart: a copy-on-write page
store where each page has a version chain, a snapshot records a version stamp, and
a read through a snapshot resolves to the newest page version at or before that
stamp. Writes append a new version rather than mutating in place, and a GC pass
reclaims versions older than the oldest live snapshot. This is the mechanism a
time-travel read (`AS OF`) would use.

### 17.3 The write-ahead log (`wal`)

The `wal` module implements a segmented write-ahead log: records are appended to
fixed-size segments, each record carries a checksum, and the log can be replayed
to recover committed but not-yet-checkpointed changes. A checkpoint (the `script`
`checkpoint` statement) serializes the current database with `format::
encode_database` and truncates the log up to that point. The `ringbuf` module
provides the bounded queue the group-commit path batches records through.

---

## 18. The operation script language (`script`)

The operation script is the imperative language the primary fuzz harness drives.
A script is a sequence of statements separated by `;` or newlines. Each statement
is parsed and executed independently, and a malformed statement is skipped so a
partially-valid script still exercises the engine.

### 18.1 Grammar

```text
  script     := statement (";" statement)* ";"?
  statement  := insert | update | delete | scan | scan_where
              | index_scan | join | compact | checkpoint | agg
  insert     := "insert" value+
  update     := "update" "set" col value "where" col op value
  delete     := "delete" "where" col op value
  scan       := "scan"
  scan_where := "scan" "where" col op value
  index_scan := "index_scan" col value
  join       := "join"
  compact    := "compact"
  checkpoint := "checkpoint"
  agg        := "agg" col
  col        := identifier | integer          (column name or 0-based index)
  op         := "=" | "==" | "!=" | "<>" | "<" | "<=" | ">" | ">="
  value      := integer | real | "true" | "false" | "null" | string
  string     := '"' ... '"'                   (interned into the column dictionary)
```

### 18.2 Statements

- **`insert <v> <v> ...`** — append a row, values positional in schema order.
  Extra or missing values are handled per the interpreter's coercion rules; a
  text value is a double-quoted string that is interned into the target column's
  dictionary.
- **`update set <col> <v> where <col> <op> <v>`** — overwrite `<col>` with `<v>`
  for every row matching the predicate.
- **`delete where <col> <op> <v>`** — tombstone every row matching the predicate.
- **`scan`** — read and materialize all live rows.
- **`scan where <col> <op> <v>`** — read live rows matching the predicate, with
  zone-map pruning and, where possible, index use.
- **`index_scan <col> <v>`** — answer an equality on `<col>` through its index.
- **`join`** — join the table against itself (or a second decoded input) on the
  row-id key, exercising the join operators.
- **`compact`** — run compaction (split, merge, and RLE repack) over the table.
- **`checkpoint`** — serialize the database and truncate the write-ahead log.
- **`agg <col>`** — aggregate `<col>` (count/sum/min/max).

### 18.3 Example

A script that inserts rows, forces a dictionary to grow, indexes a value, deletes
part of the table, compacts, and then reads back:

```text
insert 0 "apple";
insert 0 "banana";
insert 0 "cherry";
update set 1 "date" where id = 0;
index_scan 1 "banana";
delete where id < 2;
compact;
scan;
```

Each statement flows through the full pipeline: the inserts and update drive the
mutation layer, the dictionary, the indexes, and the maps; the index scan drives
the index's resolution path; the delete tombstones rows; and `compact` runs the
split/merge/repack routines before the final scan reads what remains.

### 18.4 Coercion and error handling

Values are coerced to the target column's type where sensible (an integer into a
real column, a boolean into an integer column). A value that cannot be coerced,
or a statement that fails to parse, is skipped without aborting the script. This
best-effort execution is what lets the fuzzer explore deep states: a long script
with one bad statement still runs the rest.

---

## 19. The SQL front end (`sql`, `logical`, `cost`, `plan`)

Above the operation script sits a SQL front end, used for the higher-level query
path and for testing the planner.

- **`sql`** is a hand-written lexer and Pratt (precedence-climbing) parser for a
  practical SQL subset: `SELECT` with projection, `WHERE`, `GROUP BY`, `HAVING`,
  `ORDER BY`, `LIMIT`/`OFFSET`, joins, and the DML/DDL statements `INSERT`,
  `UPDATE`, `DELETE`, `CREATE TABLE`, and `DROP TABLE`. It produces a `Statement`
  AST with a full expression grammar (arithmetic, comparison, logic, `IS NULL`,
  `LIKE`, `BETWEEN`, `IN`, aggregates).
- **`logical`** lowers a parsed `SELECT` into a `LogicalPlan` tree (scan, filter,
  project, join, aggregate, sort, limit, distinct) and runs a rule-based
  optimizer over it: constant folding, filter splitting and pushdown, adjacent-
  filter and projection collapsing.
- **`cost`** estimates the cardinality and cost of a logical plan from per-table
  statistics and predicate selectivity (equality `1/ndv`, range a third,
  conjunction multiplies, disjunction adds and subtracts the overlap).
- **`plan`** turns a logical plan into a physical one, choosing scan paths and
  join algorithms from those cost estimates.

The SQL path and the operation-script path are two front ends over the same
storage engine. The operation script is the one the fuzzer drives, because it is
the most direct route into the mutation, compaction, and index machinery.

---

## 20. Auxiliary subsystems

The engine carries a substantial library of supporting data structures and
utilities. They exist for their own sake — real subsystems a storage engine of
this size accumulates — and several are used by the core paths above.

**Statistics and sketches.** `stats` builds equi-width and equi-depth histograms
and most-frequent-value lists; `analyze` profiles a column (row/null counts,
min/max, distinct estimate, histogram) in one pass; `hll` is a HyperLogLog
distinct-count estimator; `quantile` provides reservoir sampling and quantile
sketches; `welford` computes streaming variance, covariance, and correlation;
`topk` tracks heavy hitters with the space-saving algorithm; `minhash` estimates
Jaccard similarity.

**Indexes and ordered structures.** `btree` is a paged B+tree; `hashindex` is a
paged hash index with overflow chains; `bitmap` is a WAH-compressed bitmap index;
`roaring` is a roaring compressed bitmap; `skiplist` is a probabilistic ordered
map; `radix` is a compressed radix trie; `interval` is an augmented interval tree;
`rangeset` is a disjoint integer range set.

**Allocation and caching.** `arena` is a generational arena with stale-handle
detection; `slab` is a slab allocator with reusable keys; `lru` is a bounded LRU
cache; `bufferpool` is the clock-eviction page cache; `ringbuf` is a fixed-
capacity ring buffer.

**Algorithms.** `graph` provides topological sort and strongly-connected
components; `dsu` is union-find; `kmerge` is a loser-tree k-way merge; `radixsort`
is an LSD radix sort; `partition` is a hash partitioner; `pqueue` is a keyed
priority queue with decrease-key; `bitmatrix` is a dense bit matrix with
transitive closure; `bitset` is a growable dense bit set.

**Encoding and serialization.** `encoding` and `rle` are the column codecs;
`frontcode` is a front-coded string dictionary; `compress` provides byte-block
codecs (store/RLE/LZ77); `batchio` serializes record batches; `varint` provides
LEB128 varints and a bit reader/writer; `codec` provides hex and Base64;
`checksum` provides CRC-32, Adler-32, and FxHash; `fingerprint` provides
Rabin-Karp rolling hashes and content-defined chunking.

**Types and text.** `decimal` is fixed-point decimal arithmetic; `temporal` is
calendar dates and timestamps; `datefn` is date/timestamp functions; `text` is
string functions; `pattern` matches `LIKE`, glob, and a small regex subset;
`editdist` computes edit distances; `scalar` is the scalar function registry;
`intern` is a string interner.

**I/O and presentation.** `json` is a JSON parser/serializer; `csvio` imports and
exports CSV with type inference; `wire` is a binary request/response protocol
codec; `pretty` renders result sets as aligned tables; `catalog` is a system
catalog of tables, indexes, sequences, and views; `session` manages prepared
statements and settings; `diag` is diagnostics and pretty-printing.

---

## 21. Fuzzing harnesses

The fuzz targets live under `fuzz/fuzz_targets/` and are built by
`.clusterfuzzlite/build.sh` into `$OUT`.

### 21.1 `query_fuzzer` (primary / canonical)

`query_fuzzer` is the canonical harness. Its input is the combined blob
(`[u32 LE script length][script bytes][.sht database bytes]`), and it calls
`schist::run_combined`, driving the full `decode → verify → run` pipeline. This is
the target that exercises the whole engine end to end, and it is the one every
crafted input should be validated against.

### 21.2 `decode_fuzzer`

`decode_fuzzer` runs `decode → verify` only, on raw `.sht` bytes. It reaches the
container decoder and the cross-page verifier but runs no script. It is bonus
coverage of the decode path.

### 21.3 `script_fuzzer`

`script_fuzzer` decodes a small fixed empty database and then runs the input
bytes as an operation script. It exercises the script interpreter and the
mutation/compaction/query layers without requiring a structured container. It is
bonus coverage of the operation path.

### 21.4 Seeds

The `fuzz/corpus/<target>/` directories hold seed inputs that reach deep paths:
valid `.sht` databases with dictionary-encoded and RLE columns, and scripts that
insert past split thresholds, run index scans and joins, and trigger compaction
and RLE repack. The `fuzz/dictionary.txt` file lists format tokens and script
keywords to help the fuzzer form structured input.

---

## 22. Building and testing

`schist` is a standard Cargo crate with no external dependencies.

```sh
# Build the library and run the full test suite.
cargo build
cargo test

# Build a fuzz target under AddressSanitizer (requires nightly + cargo-fuzz).
cargo +nightly fuzz build --release query_fuzzer

# Run a target against a saved input.
./fuzz/target/x86_64-unknown-linux-gnu/release/query_fuzzer path/to/input
```

The build is deterministic and non-interactive. The test suite covers each module
directly; the storage pipeline additionally has integration tests that decode a
database, run a script, and check the result.

---

## 23. Invariant reference

The invariants each subsystem maintains, gathered in one place. These are the
properties operations must preserve; the verifier checks the decode-time subset,
and the operation layers are responsible for preserving them as state evolves.

| # | Subsystem | Invariant |
| - | --------- | --------- |
| 1 | schema | The first column is the row-id column; every data page's column count and per-column kind match the schema. |
| 2 | pager | A slot index handed to a page accessor is within that page's current slot directory. |
| 3 | dict | A dictionary id is within its page's range; the generation counter advances on every page rewrite. |
| 4 | rowid | Every live row id maps to a valid, live slot on the page it names. |
| 5 | fsm | A page on the free-page list holds no live rows; free counts do not exceed capacities. |
| 6 | zonemap | Each page has a zone map whose min/max bracket the page's values and whose liveness flag matches the page. |
| 7 | index | Every row id an entry lists exists and is live; a cached dictionary reference is consistent with the dictionary page and generation it was captured against. |
| 8 | mutation | After a mutation, the row-id map, indexes, zone maps, and free-space map agree with the physical page contents. |
| 9 | compact | After compaction, every surviving row is reachable at the slot the row-id map names for it. |
| 10 | verify | A decoded database that passes verification is internally consistent at decode time. |

The subtle invariants are the cross-subsystem ones (rows 3–4, 7–9): they relate
state held in one subsystem to state held in another, and they are preserved by
the *sequence* of updates an operation performs across subsystems, not by any
single subsystem in isolation.

---

## 24. Glossary

- **Cell.** The bytes of one column's value for one row within a page.
- **Checkpoint.** Serializing the current database to a `.sht` blob and
  truncating the write-ahead log up to that point.
- **Data page.** A page holding rows' column data plus a slot directory.
- **Dictionary id.** The integer a `Text` value carries; an index into a
  dictionary page.
- **Dictionary page.** A page holding the distinct strings of a dictionary- or
  RLE-text column, addressed by id, carrying a generation counter.
- **Generation.** A version stamp on a dictionary page's physical layout,
  advanced on rewrite.
- **Row id.** The logical identity of a row, assigned by the next-row-id counter
  and never reused.
- **Row-id map.** The structure mapping a row id to the `(page, slot)` currently
  holding it, plus the tombstone bitmap.
- **RLE repack.** In-place rewriting of a run-length-encoded page's runs,
  compacting live rows to the front and renumbering their slots.
- **Slot.** An entry in a page's slot directory: `(row_id, offset, length,
  live)`, locating a row's cells within the page buffer.
- **Tombstone.** A slot marked `live = 0` and a row id marked deleted, awaiting
  reclamation by compaction.
- **Zone map.** A per-page min/max summary used to prune pages during a scan.
- **Combined blob.** The primary harness input: a length-prefixed script followed
  by a `.sht` database, split and run by `run_combined`.

---

*`schist` is a study engine built to be fuzzed. It is intentionally large and its
subsystems interact in non-obvious ways; the reference above is the map, but the
territory is the source.*

---

## Appendix A. The integer codecs in detail (`encoding`)

The `encoding` module implements the low-level codecs that the column encodings
build on. Each takes a slice of integers and produces a compact byte string, and
each is exactly invertible. They are combined by the column layer: a plain
integer column may be stored frame-of-reference plus bit-packed, a monotically
increasing column delta-of-delta encoded, and so on.

### A.1 Bit-packing

Bit-packing stores each value in the minimum fixed number of bits needed for the
column's range, rather than a full 64 bits. If every value in a block fits in 5
bits, the block stores 5 bits per value, packed across byte boundaries with the
`varint` module's bit writer. Decoding reads the same fixed width back. The width
is stored in the block header. Bit-packing is the workhorse: most other codecs
reduce their input to small integers and then bit-pack the result.

### A.2 Frame of reference

Frame-of-reference subtracts a per-block base (typically the block minimum) from
every value, so a block of large but close-together values becomes a block of
small values that bit-pack tightly. The base is stored once per block; decoding
adds it back. This is the standard technique for columns whose values cluster in
a narrow band far from zero (timestamps, ids).

### A.3 Delta and delta-of-delta

Delta encoding stores the first value, then the difference between each value and
its predecessor. For a monotonically increasing column the deltas are small and
positive and bit-pack well. Delta-of-delta goes one step further: it stores the
differences *between the deltas*, which are near zero for a column that increases
at a steady rate (a clock, a sequence). Both use zig-zag mapping (below) so that
a negative delta stays short.

### A.4 Zig-zag

Zig-zag mapping turns a signed integer into an unsigned one whose magnitude
tracks the signed value's distance from zero: 0 → 0, -1 → 1, 1 → 2, -2 → 3, and so
on. This is what lets delta codecs bit-pack negative deltas as tightly as
positive ones, because a small-magnitude negative number maps to a small unsigned
number rather than a huge one.

### A.5 Null suppression

Null suppression stores a null bitmap plus the values of only the non-null slots,
rather than a placeholder for each null. For a sparse column (mostly null) this
is a large saving. The bitmap is one bit per slot; decoding walks the bitmap and
consumes a value from the packed stream for each set bit.

### A.6 Constant and RLE

A block whose values are all identical is stored as a single value plus a count —
the constant codec. Run-length encoding generalizes this to a sequence of
`(value, count)` runs, which is also the on-page layout for RLE columns
(section 5.3). The `rle` module implements the page-level form; the `encoding`
module's constant codec is the degenerate single-run case used inside blocks.

### A.7 Byte-aligned dictionary

The byte-aligned dictionary codec builds a per-block dictionary of the distinct
values in a block and stores each value as its dictionary index, byte-aligned for
fast random access. This is distinct from the column-level dictionary encoding
(which shares one dictionary page across all of a column's pages); the block-level
form is used where a single block has low cardinality but the column as a whole
does not.

### A.8 How the column layer combines them

A column's encoding decides which codecs apply. A plain integer column with a
narrow range is frame-of-reference plus bit-packed. A monotonic column is
delta-of-delta plus zig-zag plus bit-packed. A dictionary text column stores
dictionary ids, which are themselves small integers and so bit-pack. An RLE
column is stored as runs. The decoder reads the encoding byte and applies the
inverse chain. The point of the layered design is that the same handful of
integer codecs compose to cover every column shape without a bespoke format per
column type.

---

## Appendix B. The lifecycle of a row

To see how the subsystems interact, follow one row from insertion to
reclamation. Suppose the schema is `[id (row-id), name (text, dictionary), score
(int, plain)]`, and the database already holds a few rows.

**Insert.** The statement `insert 0 "carol" 88` runs the insert path
(section 12.1):

1. The next-row-id counter hands out row id, say 7.
2. The free-space map is consulted for a data page with room; say page 3 has a
   free slot.
3. The `name` column is dictionary-encoded, so `"carol"` is interned into the
   `name` dictionary page. If `"carol"` is new, this appends it to the dictionary
   and, if the dictionary buffer is now full, rewrites the page and advances its
   generation. The value becomes a dictionary id, say 12.
4. The `score` column is plain int, so 88 is written as 8 bytes.
5. The row's cells (`id=7`, `name=12`, `score=88`) are written into page 3's
   buffer, and a slot-directory entry `(row_id=7, off=…, len=…, live=1)` is
   appended.
6. The row-id map records `7 → (page 3, slot k)`.
7. The `name` index gains row id 7 under dictionary id 12, and — because the
   index caches a dictionary reference for its values (section 11.2) — the entry
   for 12 carries a reference into the `name` dictionary page and that page's
   current generation.
8. Page 3's zone map widens its `score` range to include 88 if needed.

**Read.** A later `index_scan name "carol"` probes the `name` index for the
dictionary id of `"carol"`, finds the entry for id 12, resolves the value through
the entry's dictionary fast path, obtains the row-id list `[…, 7, …]`, and for
each row id asks the row-id map for its `(page, slot)` and reads the row's cells
through the pager's fast accessor.

**Update.** `update set score 90 where id = 7` locates row 7 through the row-id
map, overwrites its `score` cell in place (both are the same width), and updates
page 3's zone map. Because `score` is not indexed, no index entry moves. Had the
update changed `name`, the row id would move from the old value's index entry to
the new value's, and a new text value would be interned (possibly rewriting the
dictionary and advancing its generation).

**Delete.** `delete where id = 7` tombstones the row: page 3's slot k gets
`live = 0`, row id 7's bit is set in the deletion bitmap, row id 7 is removed from
the `name` index entry for id 12, and the free-space map records slot k as
reclaimable. The row's bytes remain on page 3.

**Compaction.** A later `compact` reclaims the space. If page 3 is a plain page
with tombstones, compaction may split or merge it, moving live rows and updating
their row-id-map entries to their new `(page, slot)`. If page 3 (or the `name`
column's storage) is RLE, the RLE repack rewrites the runs, compacts live rows to
the front, and renumbers the slots, so a surviving row that was at slot k may
land at a lower slot index. Either way, after compaction a subsequent read of a
surviving row must reach it at whatever slot the row-id map now names for it, and
a subsequent index probe on a dictionary value must resolve that value against
whatever generation the dictionary page is now at.

The row's whole life thus touches the pager, the dictionary, the row-id map, the
free-space map, the zone map, and the index, and the *order* in which those are
updated during the insert, the update, the delete, and the compaction is what
keeps them mutually consistent.

---

## Appendix C. A worked byte-layout example

Consider a minimal database: schema `[id (row-id, int), x (int, plain)]` with two
rows, `(id=1, x=10)` and `(id=2, x=20)`, on a single data page. Its `.sht` bytes
begin with the header:

```text
  53 43 48 49 53 54 31 01   magic "SCHIST1\x01"
  01 00 00 00               version = 1
  05 00 00 00               record count = 5   (schema, data, fsm, rowid, zonemap,
                                                 next-row-id — count varies with
                                                 what is present)
```

followed by five 17-byte directory entries, each `[kind][id][off][len][checksum]`,
for example the schema entry:

```text
  00                        kind = 0 (schema)
  00 00 00 00               id = 0
  00 00 00 00               off = 0 (first body)
  0e 00 00 00               len = 14 (illustrative)
  ..  ..  ..  ..            additive checksum over the body
```

then the data entry:

```text
  01                        kind = 1 (data)
  03 00 00 00               id = 3 (page id)
  0e 00 00 00               off = 14
  ..  ..  ..  ..            len and checksum
```

The data body itself is `[u32 body_len][chunk][u32 slot_count][slots]`. The chunk
is `[u32 num_rows=2][u32 num_cols=2]` then, per column, `[kind][encoding][u32
data_len][data]`. For the `id` column (int, plain) the data is the two 8-byte
values `1` and `2`; for the `x` column the two 8-byte values `10` and `20`. The
slot directory then has two entries, `(row_id=1, off, len, live=1)` and
`(row_id=2, off, len, live=1)`, whose offsets point into the chunk at the two
rows' cells.

The row-id record maps `1 → (page 3, slot 0)` and `2 → (page 3, slot 1)`. The
zonemap record holds page 3's per-column min/max: `id ∈ [1, 2]`, `x ∈ [10, 20]`.
The fsm record notes page 3's remaining capacity, and the next-row-id record
holds `3` (the next id to hand out).

To read row 2, a consumer asks the row-id map for row id 2, gets `(page 3, slot
1)`, indexes page 3's slot directory at slot 1 to get that row's `(off, len)`, and
reads the cells at that offset in page 3's buffer through the pager. Every layer
in that chain — the row-id map's slot index, the slot directory's offset, the
pager's read — trusts the layer below it to have been kept consistent by whatever
operations ran since the page was decoded.

This end-to-end path — container header, directory, data body, slot directory,
row-id map, zone map, and the pager accessors underneath — is the same one the
`query_fuzzer` drives for every input, and it is where the engine's behavior is
determined by the interaction of the pieces rather than by any one of them alone.

---

## Appendix D. Query planning, end to end

A SQL query and an operation script are two ways to reach the same operators.
This appendix traces a SQL query through the front end so the relationship between
the modules in section 19 is concrete.

Take `SELECT name, COUNT(*) FROM t WHERE score > 50 GROUP BY name ORDER BY
COUNT(*) DESC LIMIT 10`.

1. **Lexing.** `sql::Lexer` turns the text into tokens: the keywords `SELECT`,
   `FROM`, `WHERE`, `GROUP`, `BY`, `ORDER`, `DESC`, `LIMIT`; the identifiers
   `name`, `t`, `score`; the aggregate `COUNT`; the operator `>`; and the
   integers `50` and `10`. Whitespace and `--` comments are skipped, and string
   and quoted-identifier literals are recognized.

2. **Parsing.** `sql::Parser` consumes the tokens with a recursive-descent
   statement parser and a Pratt expression parser. The result is a `SelectStmt`
   with its projection items (`name`, and the aggregate `COUNT(*)`), its `from`
   table, its `filter` expression (`score > 50`), its `group_by` list (`name`),
   its `order_by` list (`COUNT(*)` descending), and its `limit` (10). Operator
   precedence is handled by the Pratt parser's binding powers, so `a = 1 AND b >
   2 OR c < 3` groups as `(a = 1 AND b > 2) OR (c < 3)`.

3. **Lowering.** `logical::LogicalPlan::from_select` builds the logical tree
   bottom-up: a `Scan` of `t`, wrapped in a `Filter` on `score > 50`, wrapped in
   an `Aggregate` grouping by `name` and computing `COUNT(*)`, wrapped in a
   `Sort` on the count descending, wrapped in a `Limit` of 10. Because the query
   has an aggregate, the projection is folded into the aggregate node rather than
   a separate `Project`.

4. **Optimizing.** `logical::Optimizer` runs rewrite rules to a fixpoint. It folds
   constants in the predicate, splits conjunctive filters and pushes each conjunct
   as far down the tree as it can, collapses adjacent filters and projections, and
   drops always-true filters. For this query the filter is already directly above
   the scan, so pushdown is a no-op, but the rule set is what would move a filter
   below a projection in a more complex plan.

5. **Costing.** `cost::CostModel` walks the optimized plan bottom-up. The scan's
   cardinality comes from `t`'s row count; the filter's from the scan's rows times
   the range selectivity of `score > 50` (a third by default, or `1/ndv`-derived
   if statistics are present); the aggregate's from a distinct-groups heuristic;
   and the limit caps the final row estimate at 10. Each node also accrues a cost
   in abstract units (row touches plus a page-I/O surcharge on the scan).

6. **Physical planning.** `plan` turns the logical plan into physical operators,
   choosing a scan path (sequential, or an index scan if `score` were indexed and
   the predicate were an equality) and, for a join, a join algorithm from the
   cardinality estimates — hash join when one side is small, sort-merge for larger
   balanced inputs, nested-loop only for tiny inputs.

The operation script skips the front end and drives the operators directly, which
is why it is the fuzzer's route: it reaches the mutation, compaction, and index
machinery with the least ceremony.

---

## Appendix E. Why the unsafe core is where it is

Rust's safety guarantees mean that a memory-safety fault can only originate at an
`unsafe` boundary. `schist` confines that boundary to the pager's page accessors,
and the reasoning behind that choice is worth stating, because it shapes how the
whole engine is meant to be read.

A page's bytes are a flat buffer, and the structures that give those bytes meaning
— the slot directory, the row-id map, the dictionary's id-to-offset mapping, the
index's cached references — are integers and offsets computed by higher layers.
The pager's accessors take such an integer or offset and return the bytes it
points at. They are correct exactly when their inputs are valid: a slot index
within the current slot directory, an offset within the page buffer, a pointer
into a live page. They cannot themselves check validity in general, because
"valid" is defined by the higher-level structure that produced the input, not by
anything local to the page.

This means the interesting correctness conditions are not *in* the pager; they are
in the agreements between the subsystems that feed it. The row-id map must agree
with the slot directories. The index's cached references must agree with the
dictionary pages and generations they were captured against. The compaction
routines must leave every surviving row reachable at the slot the row-id map names
for it. Each of these is a property of a *sequence* of updates across subsystems,
preserved by the operation that performs them, and none of them is visible by
reading the pager alone — the pager just does what it is told.

Concentrating `unsafe` in one small, heavily-used module is therefore not a way to
make the memory model trivial; it is a way to make it *legible*. Every raw access
in the engine goes through a handful of accessors, so the question "could this
access be out of bounds or stale?" always reduces to "was the offset or slot or
generation that reached this accessor kept consistent by the operations that ran
before it?" — a question about the cross-subsystem invariants in section 23, not
about scattered pointer arithmetic. That is the intended way to reason about the
engine, and it is why the accessors are small, uncommented about their callers'
assumptions, and used everywhere.

---

## Appendix F. Notes on determinism and reproducibility

Everything in `schist` that could introduce nondeterminism is pinned:

- The internal random-number generators — skip-list promotion heights, reservoir
  sampling, the hash-coefficient seeds in `minhash` — are seeded from fixed
  constants, so two runs over the same input produce identical structures.
- The decoder is a pure function of its input bytes; there is no reliance on
  allocation addresses, iteration order of a `HashMap` for anything observable,
  or wall-clock time.
- Hash-based structures that iterate for output (`catalog` listings, `dsu`
  classes, `partition` histograms) sort before returning, so their observable
  order does not depend on hasher state.
- The build has no network access, no code generation, and no environment
  dependence beyond the Rust toolchain.

This determinism is what makes the engine fuzzable in the first place: a crashing
input reproduces exactly, every time, which is a prerequisite for both the fuzzer
finding a fault and a fix being verified against it.


