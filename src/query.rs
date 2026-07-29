//! Scans, index scans, merge join, and aggregation.
//!
//! The query layer reads rows back out of the database. Two materialization
//! paths matter for correctness:
//!
//! - The *safe* path iterates a page's slot directory directly and reads each
//!   live slot with a bounds-checked accessor. Full scans use this.
//! - The *fast* path resolves a row through the row-id map and reads its slot
//!   with the unchecked accessor. Point lookups (`where id = …`) and
//!   index-driven access use this; they trust the row-id map to supply an
//!   in-range slot index.
//!
//! Index scans additionally probe the secondary index cache, which compares a
//! query value against each indexed value's cached byte string.

use crate::error::{Error, Result};
use crate::format::{decode_body, read_cell};
use crate::mutation::{collect_matching, read_row_cell, read_row_cell_fast};
use crate::value::{CmpOp, Value};
use crate::Database;

/// A materialized row: its id and its column values (schema order).
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub id: u64,
    pub values: Vec<Value>,
}

/// Read every column of a row via the row-id fast path.
pub fn materialize_row_fast(db: &Database, row_id: u64) -> Option<Row> {
    let mut values = Vec::with_capacity(db.schema.arity());
    for ci in 0..db.schema.arity() {
        values.push(read_row_cell_fast(db, row_id, ci)?);
    }
    Some(Row { id: row_id, values })
}

/// Read every column of a row via the safe path.
pub fn materialize_row_safe(db: &Database, row_id: u64) -> Option<Row> {
    let mut values = Vec::with_capacity(db.schema.arity());
    for ci in 0..db.schema.arity() {
        values.push(read_row_cell(db, row_id, ci)?);
    }
    Some(Row { id: row_id, values })
}

/// A full scan over every live row, materialized via the safe path.
pub fn scan_all(db: &mut Database) -> Vec<Row> {
    db.stats.scans += 1;
    let mut out = Vec::new();
    let page_ids: Vec<u32> = db.pager.data_pages().map(|p| p.id).collect();
    for pid in page_ids {
        let page = match db.pager.get(pid) {
            Some(p) => p,
            None => continue,
        };
        let body = match decode_body(&page.buf) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let dir = match page.slot_dir.as_ref() {
            Some(d) => d,
            None => continue,
        };
        for (slot, entry) in dir.iter().enumerate() {
            if !entry.live {
                continue;
            }
            let mut values = Vec::with_capacity(body.regions.len());
            for (ci, region) in body.regions.iter().enumerate() {
                let dict = db.dicts.get(ci).and_then(|o| o.as_ref());
                values.push(read_cell(region, slot, dict, &db.pager));
            }
            out.push(Row { id: entry.row_id, values });
        }
    }
    out
}

/// A filtered scan. Point lookups on the id column and equality predicates on
/// indexed columns use the fast paths; everything else falls back to a full
/// scan with a safe predicate evaluation.
pub fn scan_where(
    db: &mut Database,
    col: usize,
    op: CmpOp,
    target: Value,
) -> Result<Vec<Row>> {
    db.stats.scans += 1;
    // Point lookup on the id column.
    if col == db.schema.row_id_index() && op == CmpOp::Eq {
        if let Some(id) = target.as_int() {
            if db.rowid_map.contains(id as u64) {
                if let Some(row) = materialize_row_fast(db, id as u64) {
                    return Ok(vec![row]);
                }
                return Ok(Vec::new());
            }
            return Ok(Vec::new());
        }
    }
    // Index-driven equality on an indexed column.
    if op == CmpOp::Eq && db.schema.columns.get(col).map_or(false, |c| c.index_page.is_some()) {
        return index_scan(db, col, target);
    }
    // Full-scan fallback.
    let matching = collect_matching(db, col, op, target)?;
    let mut out = Vec::new();
    for row_id in matching {
        if let Some(row) = materialize_row_safe(db, row_id) {
            out.push(row);
        }
    }
    Ok(out)
}

/// An index scan: probe the index for rows whose indexed value equals `target`,
/// then materialize them via the fast path.
pub fn index_scan(db: &mut Database, col: usize, target: Value) -> Result<Vec<Row>> {
    db.stats.index_scans += 1;
    let idx = db
        .index_of(col)
        .ok_or_else(|| Error::NotFound(format!("no index on column {col}")))?;
    let dict = db
        .dicts
        .get(col)
        .and_then(|o| o.as_ref())
        .ok_or_else(|| Error::NotFound(format!("no dict for column {col}")))?;
    let query_bytes: Vec<u8> = match target {
        Value::Text(id) => dict
            .bytes_of(&db.pager, id)
            .ok_or_else(|| Error::NotFound(format!("dict id {id}")))?
            .to_vec(),
        other => {
            // Non-text target: encode by the column kind for byte comparison.
            let kind = db.schema.columns[col].kind;
            crate::format::encode_cell(kind, &other)
        }
    };
    let row_ids = idx.probe(&db.pager, &query_bytes);
    let mut out = Vec::new();
    for row_id in row_ids {
        if let Some(row) = materialize_row_fast(db, row_id) {
            out.push(row);
        }
    }
    Ok(out)
}

/// A self merge-join on the id column (a demonstration of the join machinery).
pub fn merge_join(db: &mut Database) -> Vec<(Row, Row)> {
    let mut rows = scan_all(db);
    rows.sort_by_key(|r| r.id);
    rows.windows(2)
        .filter_map(|w| {
            if w[0].values.get(1) == w[1].values.get(1) {
                Some((w[0].clone(), w[1].clone()))
            } else {
                None
            }
        })
        .collect()
}

/// An aggregate over a column: count, sum (ints), min, max.
#[derive(Debug, Clone, Default)]
pub struct Agg {
    pub count: u64,
    pub sum: i128,
    pub min: Option<Value>,
    pub max: Option<Value>,
}

pub fn aggregate(db: &mut Database, col: usize) -> Agg {
    let rows = scan_all(db);
    let mut agg = Agg::default();
    for r in &rows {
        let v = r.values.get(col).copied().unwrap_or(Value::Null);
        if v.is_null() {
            continue;
        }
        agg.count += 1;
        if let Some(i) = v.as_int() {
            agg.sum += i as i128;
        }
        agg.min = Some(match agg.min {
            Some(m) => if v.total_cmp(&m) == std::cmp::Ordering::Less { v } else { m },
            None => v,
        });
        agg.max = Some(match agg.max {
            Some(m) => if v.total_cmp(&m) == std::cmp::Ordering::Greater { v } else { m },
            None => v,
        });
    }
    agg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutation::insert;
    use crate::schema::{ColKind, Column, Encoding, Schema};

    fn int_db() -> Database {
        let mut cols = vec![Column::row_id()];
        cols.push(Column::new("x", ColKind::Int, Encoding::Plain));
        Database::new(Schema::new(cols))
    }

    #[test]
    fn scan_all_returns_inserted_rows() {
        let mut db = int_db();
        insert(&mut db, &[Value::Int(0), Value::Int(10)]).unwrap();
        insert(&mut db, &[Value::Int(0), Value::Int(20)]).unwrap();
        let rows = scan_all(&mut db);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[1], Value::Int(10));
    }

    #[test]
    fn point_lookup_uses_fast_path() {
        let mut db = int_db();
        let id = insert(&mut db, &[Value::Int(0), Value::Int(7)]).unwrap();
        let rows = scan_where(&mut db, 0, CmpOp::Eq, Value::Int(id as i64)).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], Value::Int(7));
    }

    #[test]
    fn aggregate_counts_and_sums() {
        let mut db = int_db();
        insert(&mut db, &[Value::Int(0), Value::Int(3)]).unwrap();
        insert(&mut db, &[Value::Int(0), Value::Int(5)]).unwrap();
        let agg = aggregate(&mut db, 1);
        assert_eq!(agg.count, 2);
        assert_eq!(agg.sum, 8);
        assert_eq!(agg.min, Some(Value::Int(3)));
        assert_eq!(agg.max, Some(Value::Int(5)));
    }
}
