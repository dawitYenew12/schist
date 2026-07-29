//! `schist` — an original columnar storage engine.
//!
//! `schist` models a single-rooted columnar database on top of a page buffer
//! manager. A `schist` database is a self-describing binary file (`.sht`): it
//! carries its schema, its columnar data pages, its dictionary pages, its
//! secondary indexes, its free-space map, its row-id map, and its zone maps —
//! all in one blob. Decoding, verifying, and executing an operation script
//! against that file is the entire job of this crate.
//!
//! The crate is split into a number of subsystems, each of which exists for its
//! own sake:
//!
//! - [`pager`] — the page buffer manager and the only place that touches raw
//!   page memory through `unsafe` accessors.
//! - [`format`] — the `.sht` container decoder and the checkpoint serializer.
//! - [`schema`] — the column model, value types, and column encodings.
//! - [`verify`] — the cross-page consistency checker.
//! - [`fsm`] — the free-space map.
//! - [`rowid`] — the row-id map.
//! - [`index`] — the secondary index.
//! - [`dict`] — the dictionary pages.
//! - [`zonemap`] — the per-page min/max summaries.
//! - [`mutation`] — insert / update / delete.
//! - [`compact`] — page split, underfull-page merge, and in-place RLE repack.
//! - [`query`] — scans, index scans, merge join, and aggregation.
//! - [`script`] — the small imperative operation language the harness drives.
//! - [`diag`] — diagnostics and pretty-printing.
//!
//! See the project `README.md` for the full architecture, the `.sht` byte
//! format, the per-subsystem invariants, and the script grammar.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]

pub mod array;
pub mod bitmap;
pub mod bloom;
pub mod btree;
pub mod bufferpool;
pub mod catalog;
pub mod pattern;
pub mod radix;
pub mod roaring;
pub mod skiplist;
pub mod sql;
pub mod checksum;
pub mod compress;
pub mod csvio;
pub mod decimal;
pub mod json;
pub mod temporal;
pub mod diag;
pub mod encoding;
pub mod error;
pub mod expr;
pub mod hashindex;
pub mod format;
pub mod fsm;
pub mod index;
pub mod join;
pub mod logical;
pub mod mutation;
pub mod pager;
pub mod query;
pub mod rle;
pub mod rowid;
pub mod schema;
pub mod script;
pub mod sort;
pub mod stats;
pub mod value;
pub mod verify;
pub mod wal;
pub mod window;
pub mod zonemap;
pub mod dict;
pub mod compact;
pub mod txn;
pub mod vexec;

pub use error::{Error, Result};
pub use pager::Database;
pub use schema::{ColKind, Column, Encoding, Schema};
pub use value::{CmpOp, Value};

/// The single canonical entry point used by the primary fuzz harness.
///
/// `data` is a combined blob: a 4-byte little-endian script length, followed by
/// that many script bytes, followed by the `.sht` database bytes. The function
/// decodes the database, runs the cross-page verifier, and then executes the
/// script against it. Every stage is best-effort: a decode or verify failure
/// simply stops before the script runs, and a malformed statement is skipped
/// rather than aborting the whole run.
pub fn run_combined(data: &[u8]) {
    use crate::format::decode;
    use crate::script::run_script;
    use crate::verify::verify;

    let (script_bytes, db_bytes) = split_combined(data);
    let db = match decode(db_bytes) {
        Ok(db) => db,
        Err(_) => return,
    };
    if verify(&db).is_err() {
        return;
    }
    let src = match std::str::from_utf8(script_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut db = db;
    let _ = run_script(&mut db, src);
}

/// Split a combined harness blob into `(script_bytes, db_bytes)`.
///
/// Layout: `[u32 LE script_len][script bytes][db bytes]`. If the length prefix
/// is malformed or overruns the buffer, the whole blob is treated as database
/// bytes with an empty script.
fn split_combined(data: &[u8]) -> (&[u8], &[u8]) {
    if data.len() < 4 {
        return (&[], data);
    }
    let mut len = [0u8; 4];
    len.copy_from_slice(&data[..4]);
    let script_len = u32::from_le_bytes(len) as usize;
    if script_len > data.len() - 4 {
        return (&[], data);
    }
    let script = &data[4..4 + script_len];
    let db = &data[4 + script_len..];
    (script, db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_combined_round_trips() {
        let script = b"scan;";
        let db = b"SCHIST1dummy";
        let mut blob = (script.len() as u32).to_le_bytes().to_vec();
        blob.extend_from_slice(script);
        blob.extend_from_slice(db);
        let (s, d) = split_combined(&blob);
        assert_eq!(s, script);
        assert_eq!(d, db);
    }

    #[test]
    fn split_combined_malformed_falls_back() {
        let (s, d) = split_combined(&[9, 9, 9]);
        assert!(s.is_empty());
        assert_eq!(d.len(), 3);
        let (s, d) = split_combined(&[0xff, 0xff, 0xff, 0xff, 0x00]);
        assert!(s.is_empty());
        assert_eq!(d.len(), 5);
    }
}
