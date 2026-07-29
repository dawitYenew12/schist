//! Insert / update / delete.
//!
//! The mutation layer appends rows to data pages, resolves text values through
//! dictionary pages, and keeps the row-id map and the secondary indexes in
//! sync with the data.

use crate::error::{Error, Result, ScriptError};
use crate::format::{
    cell_width, decode_body, encode_body, encode_cell, read_cell, ColumnRegion, DataPageBody,
};
use crate::index::IndexCache;
use crate::pager::{PageId, PageKind, SlotEntry};
use crate::rle::{encode_runs, Run};
use crate::schema::{ColKind, Encoding};
use crate::value::Value;
use crate::Database;

/// A small data page, in bytes, before a new one is allocated.
pub const PAGE_DATA_CAPACITY: usize = 256;

/// Intern a text byte string for a column, returning its dictionary id. The
/// column must be a text column with a non-plain encoding and a dictionary
/// page; if the page has not been allocated yet, it is allocated here.
pub fn intern_text(db: &mut Database, col: usize, bytes: &[u8]) -> Result<u32> {
    {
        let col_def = db
            .schema
            .columns
            .get(col)
            .ok_or_else(|| Error::NotFound(format!("column {col}")))?;
        if col_def.kind != ColKind::Text {
            return Err(Error::TypeMismatch {
                col: col_def.name.clone(),
                expected: "text".to_string(),
            });
        }
        if col_def.dict_page.is_none() {
            let pid = crate::dict::Dict::allocate(&mut db.pager);
            db.schema.columns[col].dict_page = Some(pid);
            db.dicts[col] = Some(crate::dict::Dict::new(pid));
        }
    }
    let dict = db
        .dicts
        .get_mut(col)
        .and_then(|o| o.as_mut())
        .ok_or_else(|| Error::Internal("missing dict mirror".into()))?;
    let id = dict.intern(&mut db.pager, bytes)?;
    Ok(id)
}

/// Insert a row. `values` must be in schema column order and already resolved
/// (text values are `Value::Text(dict_id)`). The first column is the row id:
/// if it is a positive integer, that id is used (and `next_row_id` is advanced
/// past it); otherwise the next auto id is assigned. Returns the new row id.
pub fn insert(db: &mut Database, values: &[Value]) -> Result<u64> {
    if values.len() != db.schema.arity() {
        return Err(Error::Script(ScriptError::Arity {
            stmt: "insert".into(),
            got: values.len(),
        }));
    }
    let row_id = match values.first() {
        Some(Value::Int(i)) if *i > 0 => *i as u64,
        _ => db.next_row_id,
    };
    db.next_row_id = db.next_row_id.max(row_id + 1);
    db.stats.inserts += 1;

    let mut cells: Vec<Value> = values.to_vec();
    cells[0] = Value::Int(row_id as i64);

    // Find or allocate a data page with room for one more row.
    let row_width: usize = db
        .schema
        .columns
        .iter()
        .map(|c| cell_width(c.kind))
        .sum();
    let need = row_width.max(1);
    let page_id = match db.fsm.page_with_space(need as u32) {
        Some(pid) => pid,
        None => {
            let buf = vec![0u8; 0].into_boxed_slice();
            let pid = db.pager.alloc(PageKind::Data, buf);
            db.fsm.set_space(pid, PAGE_DATA_CAPACITY as u32);
            pid
        }
    };

    append_row(db, page_id, row_id, &cells)?;
    db.fsm.adjust_space(page_id, -(need as i64));

    // Update the secondary indexes for indexed columns.
    update_indexes_for_row(db, row_id, &cells);
    // Update zone maps.
    update_zone_maps_for_row(db, page_id, &cells);

    Ok(row_id)
}

fn append_row(
    db: &mut Database,
    page_id: PageId,
    row_id: u64,
    values: &[Value],
) -> Result<()> {
    // Read the current body (or start fresh).
    let (mut body, mut slots) = match db.pager.get(page_id) {
        Some(page) => {
            let body = if page.buf.is_empty() {
                fresh_body(db)
            } else {
                decode_body(&page.buf)?
            };
            let slots = page
                .slot_dir
                .as_ref()
                .map_or(Vec::new(), |d| d.to_vec());
            (body, slots)
        }
        None => (fresh_body(db), Vec::new()),
    };

    let slot_index = slots.len();
    for (ci, region) in body.regions.iter_mut().enumerate() {
        let v = values.get(ci).copied().unwrap_or(Value::Null);
        append_cell(region, &v);
    }
    body.num_rows = body.regions.first().map_or(0, |r| {
        if r.encoding == Encoding::Rle {
            crate::rle::run_total(&crate::rle::decode_runs(&r.data, r.kind.name()).unwrap_or_default())
                as u32
        } else {
            (r.data.len() / cell_width(r.kind).max(1)) as u32
        }
    });
    let row_width: usize = db
        .schema
        .columns
        .iter()
        .map(|c| cell_width(c.kind))
        .sum();
    slots.push(SlotEntry::new(row_id, slot_index as u32, row_width as u32));

    let body_bytes = encode_body(&body);
    let page = db
        .pager
        .get_mut(page_id)
        .ok_or_else(|| Error::Internal("page vanished".into()))?;
    page.buf = body_bytes.into_boxed_slice();
    page.replace_slots(slots);

    db.rowid_map.insert(row_id, page_id, slot_index);
    db.page_rows.entry(page_id).or_default().push(row_id);
    Ok(())
}

fn fresh_body(db: &Database) -> DataPageBody {
    let regions = db
        .schema
        .columns
        .iter()
        .map(|c| ColumnRegion::new(c.kind, c.encoding))
        .collect();
    DataPageBody { num_rows: 0, regions }
}

fn append_cell(region: &mut ColumnRegion, value: &Value) {
    match region.encoding {
        Encoding::Plain | Encoding::Dictionary => {
            region.data.extend_from_slice(&encode_cell(region.kind, value));
        }
        Encoding::Rle => {
            let mut runs = crate::rle::decode_runs(&region.data, region.kind.name()).unwrap_or_default();
            runs.push(Run::new(*value, 1));
            let coalesced = crate::rle::coalesce(&runs);
            region.data = encode_runs(&coalesced);
        }
    }
}

fn update_indexes_for_row(db: &mut Database, row_id: u64, values: &[Value]) {
    // Collect the (col, value_id, dict_pid) triples we need without holding an
    // immutable borrow of `db.schema` across the mutable updates below.
    let mut work: Vec<(usize, u32, u32)> = Vec::new();
    for (ci, col) in db.schema.columns.iter().enumerate() {
        if col.index_page.is_none() {
            continue;
        }
        let v = values.get(ci).copied().unwrap_or(Value::Null);
        let value_id = match v {
            Value::Text(id) => id,
            _ => continue,
        };
        let dict_pid = match col.dict_page {
            Some(p) => p,
            None => continue,
        };
        work.push((ci, value_id, dict_pid));
    }
    for (ci, value_id, dict_pid) in work {
        let (ptr, gen) = match db.dicts.get(ci).and_then(|o| o.as_ref()) {
            Some(dict) => {
                let off = match dict.location_of(value_id) {
                    Some((o, _)) => o as usize,
                    None => continue,
                };
                let page = match db.pager.get(dict_pid) {
                    Some(p) => p,
                    None => continue,
                };
                (IndexCache::capture_ptr(page, off), page.gen)
            }
            None => continue,
        };
        if let Some(idx) = db.index_of_mut(ci) {
            idx.add_or_append(value_id, ptr, gen, row_id);
        }
    }
}

fn update_zone_maps_for_row(db: &mut Database, page_id: PageId, values: &[Value]) {
    for (ci, _col) in db.schema.columns.iter().enumerate() {
        let v = values.get(ci).copied().unwrap_or(Value::Null);
        let zm = &mut db.zone_maps[ci];
        let existing = zm.get(page_id).copied();
        let (min, max) = match existing {
            Some(z) if !z.min.is_null() => {
                let mn = if v.total_cmp(&z.min) == std::cmp::Ordering::Less { v } else { z.min };
                let mx = if v.total_cmp(&z.max) == std::cmp::Ordering::Greater { v } else { z.max };
                (mn, mx)
            }
            _ => (v, v),
        };
        let live = existing.map_or(1, |z| z.live_rows + 1);
        zm.set(
            page_id,
            crate::zonemap::Zone { page: page_id, min, max, live_rows: live },
        );
    }
}

/// Delete every row matching a predicate on a column. Matching rows are
/// tombstoned in their slot directory and removed from the row-id map and the
/// indexes.
pub fn delete_where(db: &mut Database, col: usize, op: crate::value::CmpOp, target: Value) -> Result<usize> {
    db.stats.deletes += 1;
    let matching = collect_matching(db, col, op, target)?;
    let mut count = 0;
    for row_id in matching {
        if let Some((pid, slot)) = db.rowid_map.remove(row_id) {
            if let Some(page) = db.pager.get_mut(pid) {
                if let Some(dir) = page.slot_dir.as_mut() {
                    if slot < dir.len() {
                        dir[slot].live = false;
                    }
                }
            }
            // Remove from indexes.
            for idx in db.indexes.iter_mut().flatten() {
                for e in idx.entries.iter_mut() {
                    e.row_ids.retain(|&r| r != row_id);
                }
            }
            if let Some(rows) = db.page_rows.get_mut(&pid) {
                rows.retain(|&r| r != row_id);
            }
            count += 1;
        }
    }
    Ok(count)
}

/// Update matching rows: set `set_col` to `set_value`.
pub fn update_where(
    db: &mut Database,
    set_col: usize,
    set_value: Value,
    pred_col: usize,
    op: crate::value::CmpOp,
    target: Value,
) -> Result<usize> {
    db.stats.updates += 1;
    let matching = collect_matching(db, pred_col, op, target)?;
    let n = matching.len();
    for row_id in matching {
        let (pid, slot) = match db.rowid_map.get(row_id) {
            Some(v) => v,
            None => continue,
        };
        rewrite_cell(db, pid, slot, set_col, &set_value)?;
    }
    Ok(n)
}

fn rewrite_cell(
    db: &mut Database,
    page_id: PageId,
    slot: usize,
    col: usize,
    value: &Value,
) -> Result<()> {
    let page = db
        .pager
        .get_mut(page_id)
        .ok_or_else(|| Error::Internal("page vanished".into()))?;
    let mut body = decode_body(&page.buf)?;
    if col >= body.regions.len() {
        return Ok(());
    }
    let region = &mut body.regions[col];
    match region.encoding {
        Encoding::Plain | Encoding::Dictionary => {
            let w = cell_width(region.kind);
            let off = slot * w;
            let cell = encode_cell(region.kind, value);
            if off + w <= region.data.len() {
                region.data[off..off + w].copy_from_slice(&cell);
            }
        }
        Encoding::Rle => {
            // For RLE, rewrite by materializing, replacing, re-encoding.
            let mut runs =
                crate::rle::decode_runs(&region.data, region.kind.name()).unwrap_or_default();
            let mut flat = crate::rle::materialize(&runs);
            if slot < flat.len() {
                flat[slot] = *value;
            }
            // Rebuild runs of count 1 each (coalesced).
            let new_runs: Vec<Run> = flat.iter().map(|&v| Run::new(v, 1)).collect();
            runs = crate::rle::coalesce(&new_runs);
            region.data = encode_runs(&runs);
        }
    }
    let body_bytes = encode_body(&body);
    let page = db
        .pager
        .get_mut(page_id)
        .ok_or_else(|| Error::Internal("page vanished".into()))?;
    page.buf = body_bytes.into_boxed_slice();
    Ok(())
}

/// Collect row ids matching a predicate. Uses the index for equality on an
/// indexed column; otherwise falls back to a full scan.
pub fn collect_matching(
    db: &Database,
    col: usize,
    op: crate::value::CmpOp,
    target: Value,
) -> Result<Vec<u64>> {
    let col_def = db
        .schema
        .columns
        .get(col)
        .ok_or_else(|| Error::NotFound(format!("column {col}")))?;
    // Equality on an indexed text column uses the index cache fast path.
    if op == crate::value::CmpOp::Eq && col_def.index_page.is_some() {
        if let Some(idx) = db.index_of(col) {
            if let Some(dict) = db.dicts.get(col).and_then(|o| o.as_ref()) {
                if let Value::Text(id) = target {
                    if let Some(bytes) = dict.bytes_of(&db.pager, id) {
                        let rows = idx.probe(&db.pager, bytes);
                        if !rows.is_empty() {
                            return Ok(rows);
                        }
                    }
                }
                // If the target is a literal string not yet interned, try to
                // resolve it by probing with its bytes directly.
                if let Value::Text(id) = target {
                    if let Some(bytes) = dict.bytes_of(&db.pager, id) {
                        return Ok(idx.probe(&db.pager, bytes));
                    }
                }
            }
        }
    }
    // Full scan fallback.
    let mut out = Vec::new();
    for row_id in db.rowid_map.row_ids() {
        if let Some(v) = read_row_cell(db, row_id, col) {
            if op.apply(&v, &target) {
                out.push(row_id);
            }
        }
    }
    Ok(out)
}

/// Read one cell of a row by id, going through the row-id map.
pub fn read_row_cell(db: &Database, row_id: u64, col: usize) -> Option<Value> {
    let (pid, slot) = db.rowid_map.get(row_id)?;
    let page = db.pager.get(pid)?;
    let body = decode_body(&page.buf).ok()?;
    let region = body.regions.get(col)?;
    let dict = db.dicts.get(col).and_then(|o| o.as_ref());
    Some(read_cell(region, slot, dict, &db.pager))
}

/// Read a cell using the row-id fast path's unchecked slot accessor.
///
/// This is the path used by id-driven scans and lookups.
pub fn read_row_cell_fast(db: &Database, row_id: u64, col: usize) -> Option<Value> {
    let (pid, slot) = db.rowid_map.get(row_id)?;
    let page = db.pager.get(pid)?;
    let entry = unsafe { page.slot_at_unchecked(slot) };
    let body = decode_body(&page.buf).ok()?;
    let region = body.regions.get(col)?;
    let dict = db.dicts.get(col).and_then(|o| o.as_ref());
    Some(read_cell(region, entry.off as usize, dict, &db.pager))
}

/// Total live rows.
pub fn live_rows(db: &Database) -> usize {
    db.rowid_map.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ColKind, Column, Encoding, Schema};

    fn int_db() -> Database {
        let mut cols = vec![Column::row_id()];
        cols.push(Column::new("x", ColKind::Int, Encoding::Plain));
        Database::new(Schema::new(cols))
    }

    #[test]
    fn insert_then_read() {
        let mut db = int_db();
        let id = insert(&mut db, &[Value::Int(0), Value::Int(42)]).unwrap();
        assert_eq!(id, 1);
        assert_eq!(read_row_cell(&db, id, 1), Some(Value::Int(42)));
    }

    #[test]
    fn delete_tombstones() {
        let mut db = int_db();
        let a = insert(&mut db, &[Value::Int(0), Value::Int(1)]).unwrap();
        let b = insert(&mut db, &[Value::Int(0), Value::Int(2)]).unwrap();
        let n = delete_where(&mut db, 1, crate::value::CmpOp::Eq, Value::Int(1)).unwrap();
        assert_eq!(n, 1);
        assert!(db.rowid_map.get(a).is_none());
        assert!(db.rowid_map.get(b).is_some());
    }
}
