//! The cross-page consistency checker.
//!
//! `verify` runs once, right after decode, before any script statement
//! execute. It checks the invariants that hold *at decode time*: that every
//! page referenced by the schema (dictionary pages, index pages) exists and is
//! of the right kind, that every data page's slot directory agrees with its
//! chunk row count, that every row-id map entry points at a real slot, that
//! every free-space-map page is live, and that every zone-map entry references
//! a live data page.
//!
//! It deliberately does not re-run after each script statement: the invariants
//! it checks are static. The two invariants the engine violates are dynamic —
//! they only come into being after a sequence of mutations — so they are
//! invisible to this checker by construction.

use crate::error::{Error, Result, VerifyError};
use crate::pager::PageKind;
use crate::Database;

/// Run every decode-time check. Returns the first failure encountered.
pub fn verify(db: &Database) -> Result<()> {
    check_schema_pages(db)?;
    check_data_pages(db)?;
    check_rowid_map(db)?;
    check_fsm_pages(db)?;
    check_zone_maps(db)?;
    check_index_gens(db)?;
    Ok(())
}

fn check_schema_pages(db: &Database) -> Result<()> {
    for col in &db.schema.columns {
        if let Some(dict_pid) = col.dict_page {
            match db.pager.get(dict_pid) {
                Some(page) if page.kind == PageKind::Dict => {}
                Some(_) => {
                    return Err(Error::Verify(VerifyError::Other(format!(
                        "column {} dict_page {} is not a dict page",
                        col.name, dict_pid
                    ))))
                }
                None => {
                    return Err(Error::Verify(VerifyError::OrphanPage { page: dict_pid }))
                }
            }
        }
    }
    Ok(())
}

fn check_data_pages(db: &Database) -> Result<()> {
    for page in db.pager.data_pages() {
        let slots = page.slot_count();
        // The chunk's declared row count must match the slot directory length.
        if let Some(body) = parse_body_len(&page.buf) {
            if body.num_rows as usize != slots {
                return Err(Error::Verify(VerifyError::Other(format!(
                    "data page {} has {} slots but body declares {} rows",
                    page.id, slots, body.num_rows
                ))));
            }
        }
        // Every live slot's row id must be present in the row-id map and point
        // back at this page.
        if let Some(dir) = page.slot_dir.as_ref() {
            for (i, s) in dir.iter().enumerate() {
                if !s.live {
                    continue;
                }
                match db.rowid_map.get(s.row_id) {
                    Some((p, slot)) if p == page.id && slot == i => {}
                    _ => {
                        return Err(Error::Verify(VerifyError::RowIdBounds {
                            row: s.row_id,
                        }))
                    }
                }
            }
        }
    }
    Ok(())
}

fn check_rowid_map(db: &Database) -> Result<()> {
    for row_id in db.rowid_map.row_ids() {
        let (pid, slot) = match db.rowid_map.get(row_id) {
            Some(v) => v,
            None => {
                return Err(Error::Verify(VerifyError::RowIdBounds { row: row_id }))
            }
        };
        let page = match db.pager.get(pid) {
            Some(p) => p,
            None => return Err(Error::Verify(VerifyError::OrphanPage { page: pid })),
        };
        if page.kind != PageKind::Data {
            return Err(Error::Verify(VerifyError::OrphanPage { page: pid }));
        }
        if slot >= page.slot_count() {
            return Err(Error::Verify(VerifyError::RowIdBounds { row: row_id }));
        }
    }
    Ok(())
}

fn check_fsm_pages(db: &Database) -> Result<()> {
    // Every data page the fsm tracks must exist; the fsm's totals are
    // informational and not strictly re-derived here.
    for (pid, _free) in db.fsm.iter_space() {
        if db.pager.get(pid).is_none() {
            return Err(Error::Verify(VerifyError::OrphanPage { page: pid }));
        }
    }
    Ok(())
}

fn check_zone_maps(db: &Database) -> Result<()> {
    for (ci, zm) in db.zone_maps.iter().enumerate() {
        for zone in zm.iter() {
            match db.pager.get(zone.page) {
                Some(page) if page.kind == PageKind::Data => {}
                Some(_) => {
                    return Err(Error::Verify(VerifyError::ZoneMapLiveness {
                        page: zone.page,
                    }))
                }
                None => {
                    return Err(Error::Verify(VerifyError::ZoneMapLiveness {
                        page: zone.page,
                    }))
                }
            }
        }
        let _ = ci;
    }
    Ok(())
}

fn check_index_gens(db: &Database) -> Result<()> {
    // At decode time each index cache entry's generation must match the current
    // generation of its dictionary page. (Divergence only arises later, during
    // mutation, which is exactly when this checker is no longer run.)
    for idx in db.indexes.iter().flatten() {
        let page = match db.pager.get(idx.dict_page) {
            Some(p) => p,
            None => continue,
        };
        for entry in &idx.entries {
            if entry.gen != page.gen {
                return Err(Error::Verify(VerifyError::Other(format!(
                    "index for {} has stale generation {} vs dict page gen {}",
                    idx.column, entry.gen, page.gen
                ))));
            }
        }
    }
    Ok(())
}

fn parse_body_len(buf: &[u8]) -> Option<crate::format::DataPageBody> {
    crate::format::decode_body(buf).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{decode, encode_database};
    use crate::schema::{ColKind, Column, Encoding, Schema};

    #[test]
    fn fresh_database_verifies() {
        let mut cols = vec![Column::row_id()];
        cols.push(Column::new("x", ColKind::Int, Encoding::Plain));
        let db = Database::new(Schema::new(cols));
        assert!(verify(&db).is_ok());
        let bytes = encode_database(&db);
        let db2 = decode(&bytes).unwrap();
        assert!(verify(&db2).is_ok());
    }
}
