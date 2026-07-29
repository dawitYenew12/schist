//! Compaction: page split, underfull-page merge, and in-place RLE repack.
//!
//! Compaction reclaims space. Three operations live here:
//!
//! - [`repack`] removes tombstoned slots from a data page, compacting the live
//!   rows to the front and rebuilding the page's slot directory over only the
//!   live rows. For RLE columns it also coalesces the surviving runs.
//! - [`split`] divides an overfull page into two.
//! - [`merge`] folds an underfull page into a neighbor and frees the emptied
//!   page.
//!
//! [`compact`] walks the data pages and applies whichever operation each page
//! needs.

use crate::error::{Error, Result};
use crate::format::{cell_width, decode_body, encode_body, encode_cell, ColumnRegion, DataPageBody};
use crate::pager::{PageId, SlotEntry};
use crate::rle::{coalesce, decode_runs, encode_runs, materialize, Run};
use crate::schema::Encoding;
use crate::value::Value;
use crate::Database;

/// A page is considered overfull above this many bytes of body data.
pub const SPLIT_THRESHOLD: usize = 512;
/// A page is considered underfull below this many live rows.
pub const MERGE_THRESHOLD: usize = 2;

/// Run a full compaction pass over every data page.
pub fn compact(db: &mut Database) -> Result<usize> {
    db.stats.compactions += 1;
    let pages = db.pager.ids();
    let mut data_pages: Vec<PageId> = pages
        .into_iter()
        .filter(|&id| db.pager.get(id).map_or(false, |p| p.kind == crate::pager::PageKind::Data))
        .collect();
    let mut ops = 0;
    for pid in data_pages.drain(..) {
        if db.pager.get(pid).is_none() {
            continue;
        }
        let body_len = db.pager.get(pid).map_or(0, |p| p.buf.len());
        let live = db.pager.get(pid).map_or(0, |p| p.live_count());
        if body_len > SPLIT_THRESHOLD {
            split(db, pid)?;
            ops += 1;
        } else if live < MERGE_THRESHOLD && db.pager.data_pages().count() > 1 {
            merge(db, pid)?;
            ops += 1;
        } else if has_tombstones(db, pid) {
            repack(db, pid)?;
            ops += 1;
        }
    }
    Ok(ops)
}

fn has_tombstones(db: &Database, pid: PageId) -> bool {
    db.pager
        .get(pid)
        .map_or(false, |p| p.slot_count() != p.live_count())
}

/// Remove tombstoned slots from a page, compacting live rows to the front and
/// rebuilding the slot directory over only the live rows.
pub fn repack(db: &mut Database, pid: PageId) -> Result<()> {
    let (body, slots) = {
        let page = db
            .pager
            .get(pid)
            .ok_or_else(|| Error::Internal("page vanished".into()))?;
        let body = if page.buf.is_empty() {
            return Ok(());
        } else {
            decode_body(&page.buf)?
        };
        let slots: Vec<SlotEntry> = page
            .slot_dir
            .as_ref()
            .map_or(Vec::new(), |d| d.to_vec());
        (body, slots)
    };

    let live_indices: Vec<usize> = slots.iter().enumerate().filter_map(|(i, s)| s.live.then_some(i)).collect();
    if live_indices.is_empty() {
        // Page is entirely tombstoned; clear it.
        let fresh = fresh_body(db);
        let body_bytes = encode_body(&fresh);
        let page = db.pager.get_mut(pid).unwrap();
        page.buf = body_bytes.into_boxed_slice();
        page.replace_slots(Vec::new());
        db.fsm.set_space(pid, crate::mutation::PAGE_DATA_CAPACITY as u32);
        db.page_rows.remove(&pid);
        return Ok(());
    }

    let mut new_regions: Vec<ColumnRegion> = Vec::with_capacity(body.regions.len());
    for region in &body.regions {
        new_regions.push(repack_region(region, &live_indices));
    }
    let mut new_body = DataPageBody {
        num_rows: live_indices.len() as u32,
        regions: new_regions,
    };
    let _ = &mut new_body;

    // Rebuild the slot directory over the live rows, compacted to the front.
    let mut new_slots: Vec<SlotEntry> = Vec::with_capacity(live_indices.len());
    for (new_i, &old_i) in live_indices.iter().enumerate() {
        let old = slots[old_i];
        new_slots.push(SlotEntry {
            row_id: old.row_id,
            off: new_i as u32,
            len: old.len,
            live: true,
        });
    }

    let body_bytes = encode_body(&new_body);
    let page = db.pager.get_mut(pid).unwrap();
    page.buf = body_bytes.into_boxed_slice();
    page.replace_slots(new_slots);
    db.fsm.set_space(pid, crate::mutation::PAGE_DATA_CAPACITY as u32);

    // Keep the page_rows convenience mirror in sync (this is not the
    // authoritative structure and updating it does not affect the row-id map).
    let live_rows: Vec<u64> = live_indices.iter().map(|&i| slots[i].row_id).collect();
    db.page_rows.insert(pid, live_rows);
    Ok(())
}

fn repack_region(region: &ColumnRegion, live_indices: &[usize]) -> ColumnRegion {
    let mut new_region = ColumnRegion::new(region.kind, region.encoding);
    match region.encoding {
        Encoding::Plain | Encoding::Dictionary => {
            let w = cell_width(region.kind).max(1);
            for &idx in live_indices {
                let off = idx * w;
                if off + w <= region.data.len() {
                    new_region.data.extend_from_slice(&region.data[off..off + w]);
                } else {
                    new_region.data.extend_from_slice(&encode_cell(region.kind, &Value::Null));
                }
            }
        }
        Encoding::Rle => {
            let runs = decode_runs(&region.data, region.kind.name()).unwrap_or_default();
            let flat = materialize(&runs);
            let mut kept: Vec<Value> = Vec::with_capacity(live_indices.len());
            for &idx in live_indices {
                kept.push(flat.get(idx).copied().unwrap_or(Value::Null));
            }
            let new_runs: Vec<Run> = kept.iter().map(|&v| Run::new(v, 1)).collect();
            new_region.data = encode_runs(&coalesce(&new_runs));
        }
    }
    new_region
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

/// Split an overfull page into two, moving roughly half the live rows to a new
/// page. The row-id map is updated for the moved rows.
pub fn split(db: &mut Database, pid: PageId) -> Result<()> {
    let (body, slots) = {
        let page = db.pager.get(pid).ok_or_else(|| Error::Internal("page vanished".into()))?;
        (decode_body(&page.buf)?, page.slot_dir.as_ref().map_or(Vec::new(), |d| d.to_vec()))
    };
    let live: Vec<SlotEntry> = slots.iter().filter(|s| s.live).copied().collect();
    if live.len() < 4 {
        return Ok(());
    }
    let mid = live.len() / 2;
    let (left, right) = live.split_at(mid);

    let mut left_regions = clone_regions(&body.regions);
    let mut right_regions = clone_regions(&body.regions);
    for (ci, region) in body.regions.iter().enumerate() {
        truncate_region(&mut left_regions[ci], region, &left.iter().map(|s| s.off as usize).collect::<Vec<_>>());
        truncate_region(&mut right_regions[ci], region, &right.iter().map(|s| s.off as usize).collect::<Vec<_>>());
    }
    let left_body = DataPageBody { num_rows: left.len() as u32, regions: left_regions };
    let right_body = DataPageBody { num_rows: right.len() as u32, regions: right_regions };

    // Left page keeps pid with compacted slots.
    let left_slots: Vec<SlotEntry> = left
        .iter()
        .enumerate()
        .map(|(i, s)| SlotEntry { row_id: s.row_id, off: i as u32, len: s.len, live: true })
        .collect();
    let right_slots: Vec<SlotEntry> = right
        .iter()
        .enumerate()
        .map(|(i, s)| SlotEntry { row_id: s.row_id, off: i as u32, len: s.len, live: true })
        .collect();

    let new_pid = db.pager.alloc(crate::pager::PageKind::Data, encode_body(&right_body).into_boxed_slice());
    if let Some(page) = db.pager.get_mut(new_pid) {
        page.replace_slots(right_slots.clone());
    }
    {
        let page = db.pager.get_mut(pid).unwrap();
        page.buf = encode_body(&left_body).into_boxed_slice();
        page.replace_slots(left_slots.clone());
    }

    // Update the row-id map for both halves.
    for (i, s) in left_slots.iter().enumerate() {
        db.rowid_map.relocate(s.row_id, pid, i);
    }
    for (i, s) in right_slots.iter().enumerate() {
        db.rowid_map.relocate(s.row_id, new_pid, i);
    }
    db.fsm.set_space(pid, crate::mutation::PAGE_DATA_CAPACITY as u32);
    db.fsm.set_space(new_pid, crate::mutation::PAGE_DATA_CAPACITY as u32);
    db.page_rows.insert(pid, left_slots.iter().map(|s| s.row_id).collect());
    db.page_rows.insert(new_pid, right_slots.iter().map(|s| s.row_id).collect());
    Ok(())
}

fn clone_regions(regions: &[ColumnRegion]) -> Vec<ColumnRegion> {
    regions
        .iter()
        .map(|r| ColumnRegion { kind: r.kind, encoding: r.encoding, data: Vec::new() })
        .collect()
}

fn truncate_region(dest: &mut ColumnRegion, src: &ColumnRegion, indices: &[usize]) {
    match src.encoding {
        Encoding::Plain | Encoding::Dictionary => {
            let w = cell_width(src.kind).max(1);
            for &idx in indices {
                let off = idx * w;
                if off + w <= src.data.len() {
                    dest.data.extend_from_slice(&src.data[off..off + w]);
                }
            }
        }
        Encoding::Rle => {
            let runs = decode_runs(&src.data, src.kind.name()).unwrap_or_default();
            let flat = materialize(&runs);
            let kept: Vec<Run> = indices
                .iter()
                .map(|&i| Run::new(flat.get(i).copied().unwrap_or(Value::Null), 1))
                .collect();
            dest.data = encode_runs(&coalesce(&kept));
        }
    }
}

/// Merge an underfull page into a neighbor and free the emptied page.
pub fn merge(db: &mut Database, pid: PageId) -> Result<()> {
    let neighbor = db
        .pager
        .data_pages()
        .find(|p| p.id != pid)
        .map(|p| p.id);
    let neighbor = match neighbor {
        Some(n) => n,
        None => return Ok(()),
    };
    let (body, slots) = {
        let page = db.pager.get(pid).ok_or_else(|| Error::Internal("page vanished".into()))?;
        (decode_body(&page.buf)?, page.slot_dir.as_ref().map_or(Vec::new(), |d| d.to_vec()))
    };
    let (nbody, nslots) = {
        let page = db.pager.get(neighbor).ok_or_else(|| Error::Internal("page vanished".into()))?;
        (decode_body(&page.buf)?, page.slot_dir.as_ref().map_or(Vec::new(), |d| d.to_vec()))
    };
    let live: Vec<SlotEntry> = slots.iter().filter(|s| s.live).copied().collect();
    let mut merged = nbody;
    for (ci, region) in body.regions.iter().enumerate() {
        if ci >= merged.regions.len() {
            break;
        }
        match region.encoding {
            Encoding::Plain | Encoding::Dictionary => {
                let w = cell_width(region.kind).max(1);
                for s in &live {
                    let off = s.off as usize * w;
                    if off + w <= region.data.len() {
                        merged.regions[ci].data.extend_from_slice(&region.data[off..off + w]);
                    }
                }
            }
            Encoding::Rle => {
                let runs = decode_runs(&region.data, region.kind.name()).unwrap_or_default();
                let flat = materialize(&runs);
                let mut existing = materialize(&decode_runs(&merged.regions[ci].data, region.kind.name()).unwrap_or_default());
                for s in &live {
                    existing.push(flat.get(s.off as usize).copied().unwrap_or(Value::Null));
                }
                let new_runs: Vec<Run> = existing.iter().map(|&v| Run::new(v, 1)).collect();
                merged.regions[ci].data = encode_runs(&coalesce(&new_runs));
            }
        }
    }
    let mut new_slots = nslots;
    let start = new_slots.len();
    for (i, s) in live.iter().enumerate() {
        new_slots.push(SlotEntry { row_id: s.row_id, off: (start + i) as u32, len: s.len, live: true });
    }
    merged.num_rows = new_slots.len() as u32;
    let body_bytes = encode_body(&merged);
    let page = db.pager.get_mut(neighbor).unwrap();
    page.buf = body_bytes.into_boxed_slice();
    page.replace_slots(new_slots.clone());
    // Update the row-id map for moved rows.
    for (i, s) in new_slots.iter().enumerate().skip(start) {
        db.rowid_map.relocate(s.row_id, neighbor, i);
    }
    db.fsm.set_space(neighbor, crate::mutation::PAGE_DATA_CAPACITY as u32);
    db.pager.free(pid);
    db.fsm.forget(pid);
    db.fsm.return_page(pid);
    db.page_rows.remove(&pid);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutation::{insert, read_row_cell};
    use crate::schema::{ColKind, Column, Encoding, Schema};
    use crate::value::Value;

    fn int_db() -> Database {
        let mut cols = vec![Column::row_id()];
        cols.push(Column::new("x", ColKind::Int, Encoding::Plain));
        Database::new(Schema::new(cols))
    }

    #[test]
    fn repack_removes_tombstones_and_preserves_live_rows() {
        let mut db = int_db();
        let r1 = insert(&mut db, &[Value::Int(0), Value::Int(10)]).unwrap();
        let _r2 = insert(&mut db, &[Value::Int(0), Value::Int(20)]).unwrap();
        let _r3 = insert(&mut db, &[Value::Int(0), Value::Int(30)]).unwrap();
        // Delete the middle row to create a tombstone.
        crate::mutation::delete_where(&mut db, 1, crate::value::CmpOp::Eq, Value::Int(20)).unwrap();
        let page = db.pager.get(0).unwrap();
        assert_eq!(page.slot_count(), 3);
        drop(page);
        repack(&mut db, 0).unwrap();
        let page = db.pager.get(0).unwrap();
        assert_eq!(page.slot_count(), 2, "tombstone should be removed");
        // r1 was at slot 0 and stays at slot 0, so the safe read still works.
        assert_eq!(read_row_cell(&db, r1, 1), Some(Value::Int(10)));
    }
}
