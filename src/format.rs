//! The `.sht` container format: decoder, encoder, and data-page body helpers.
//!
//! On-disk layout
//! --------------
//!
//! ```text
//!   [8 bytes magic: "SCHIST1" + 0x01]
//!   [u32 version = 1]
//!   [u32 num_records]
//!   for each record:
//!     [u8 kind][u32 id][u32 off][u32 len][u32 checksum]
//!   ... record bodies, concatenated in directory order ...
//! ```
//!
//! Record kinds: 0 schema, 1 data, 2 dict, 3 index, 4 fsm, 5 rowid,
//! 6 zonemap. The `id` is the page id for page records and 0 for the singleton
//! schema/fsm/rowid/zonemap records. `checksum` is a simple additive checksum
//! over the body, used by the verifier.
//!
//! A *data record* body is itself structured:
//!
//! ```text
//!   [u32 body_len][body bytes][u32 slot_count][slot entries]
//! ```
//!
//! where `body` is the columnar chunk:
//!
//! ```text
//!   [u32 num_rows][u32 num_cols]
//!   for each column: [u8 kind][u8 encoding][u32 data_len][data bytes]
//! ```
//!
//! and a slot entry is `[u64 row_id][u32 off][u32 len][u8 live]`.

use crate::dict::Dict;
use crate::error::{DecodeError, Result};
use crate::fsm::Fsm;
use crate::index::IndexCache;
use crate::pager::{PageKind, Pager, SlotEntry};
use crate::rowid::RowIdMap;
use crate::schema::{ColKind, Encoding, Schema};
use crate::value::Value;
use crate::zonemap::ZoneMap;
use crate::Database;

pub const MAGIC: &[u8; 8] = b"SCHIST1\x01";
pub const VERSION: u32 = 1;

/// A directory entry.
#[derive(Debug, Clone, Copy)]
struct DirEntry {
    kind: u8,
    id: u32,
    off: u32,
    len: u32,
    checksum: u32,
}

/// The columnar chunk stored inside a data page body.
#[derive(Debug, Clone)]
pub struct DataPageBody {
    pub num_rows: u32,
    pub regions: Vec<ColumnRegion>,
}

/// One column's encoded region within a chunk.
#[derive(Debug, Clone)]
pub struct ColumnRegion {
    pub kind: ColKind,
    pub encoding: Encoding,
    pub data: Vec<u8>,
}

impl ColumnRegion {
    pub fn new(kind: ColKind, encoding: Encoding) -> Self {
        ColumnRegion {
            kind,
            encoding,
            data: Vec::new(),
        }
    }
}

/// The fixed byte width of one cell for a plain/dictionary column.
pub fn cell_width(kind: ColKind) -> usize {
    match kind {
        ColKind::Bool => 1,
        ColKind::Int => 8,
        ColKind::Real => 8,
        ColKind::Text => 4,
    }
}

/// Encode a single value into a cell for a plain/dictionary region.
pub fn encode_cell(kind: ColKind, value: &Value) -> Vec<u8> {
    match (kind, value) {
        (ColKind::Bool, Value::Bool(b)) => vec![*b as u8],
        (ColKind::Bool, Value::Null) => vec![0],
        (ColKind::Int, Value::Int(i)) => i.to_le_bytes().to_vec(),
        (ColKind::Int, Value::Null) => 0i64.to_le_bytes().to_vec(),
        (ColKind::Real, Value::Real(r)) => r.to_le_bytes().to_vec(),
        (ColKind::Real, Value::Int(i)) => (*i as f64).to_le_bytes().to_vec(),
        (ColKind::Real, Value::Null) => 0f64.to_le_bytes().to_vec(),
        (ColKind::Text, Value::Text(id)) => id.to_le_bytes().to_vec(),
        (ColKind::Text, Value::Null) => 0u32.to_le_bytes().to_vec(),
        _ => vec![0; cell_width(kind)],
    }
}

/// Read the cell at `slot_index` from a region, resolving dictionary ids
/// against the supplied dictionary mirror.
pub fn read_cell(
    region: &ColumnRegion,
    slot_index: usize,
    _dict: Option<&Dict>,
    _pager: &Pager,
) -> Value {
    match region.encoding {
        Encoding::Plain | Encoding::Dictionary => {
            let w = cell_width(region.kind);
            let off = slot_index * w;
            if off + w > region.data.len() {
                return Value::Null;
            }
            match region.kind {
                ColKind::Bool => Value::Bool(region.data[off] != 0),
                ColKind::Int => Value::Int(i64::from_le_bytes(
                    region.data[off..off + 8].try_into().unwrap(),
                )),
                ColKind::Real => Value::Real(f64::from_le_bytes(
                    region.data[off..off + 8].try_into().unwrap(),
                )),
                ColKind::Text => {
                    let id = u32::from_le_bytes(region.data[off..off + 4].try_into().unwrap());
                    Value::Text(id)
                }
            }
        }
        Encoding::Rle => {
            // The region stores runs; find the run covering `slot_index`.
            let runs = crate::rle::decode_runs(&region.data, region.kind.name()).unwrap_or_default();
            let mut pos = 0u64;
            for r in runs {
                if (slot_index as u64) < pos + r.count as u64 {
                    return r.value;
                }
                pos += r.count as u64;
            }
            Value::Null
        }
    }
}

/// Serialize a data page body (the columnar chunk) into bytes.
pub fn encode_body(body: &DataPageBody) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&body.num_rows.to_le_bytes());
    out.extend_from_slice(&(body.regions.len() as u32).to_le_bytes());
    for r in &body.regions {
        out.push(r.kind.as_u8());
        out.push(r.encoding.as_u8());
        out.extend_from_slice(&(r.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&r.data);
    }
    out
}

/// Parse a data page body from bytes.
pub fn decode_body(buf: &[u8]) -> Result<DataPageBody> {
    if buf.len() < 8 {
        return Err(DecodeError::Truncated.into());
    }
    let num_rows = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let num_cols = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let mut pos = 8;
    let mut regions = Vec::with_capacity(num_cols);
    for _ in 0..num_cols {
        if pos + 6 > buf.len() {
            return Err(DecodeError::Truncated.into());
        }
        let kind = ColKind::from_u8(buf[pos]).ok_or(DecodeError::BadSchema)?;
        let encoding = Encoding::from_u8(buf[pos + 1]).ok_or(DecodeError::BadEncoding)?;
        let data_len = u32::from_le_bytes(buf[pos + 2..pos + 6].try_into().unwrap()) as usize;
        pos += 6;
        if pos + data_len > buf.len() {
            return Err(DecodeError::Truncated.into());
        }
        let data = buf[pos..pos + data_len].to_vec();
        pos += data_len;
        regions.push(ColumnRegion { kind, encoding, data });
    }
    Ok(DataPageBody { num_rows, regions })
}

/// Serialize a slot directory.
pub fn encode_slots(slots: &[SlotEntry]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(slots.len() as u32).to_le_bytes());
    for s in slots {
        out.extend_from_slice(&s.row_id.to_le_bytes());
        out.extend_from_slice(&s.off.to_le_bytes());
        out.extend_from_slice(&s.len.to_le_bytes());
        out.push(s.live as u8);
    }
    out
}

/// Parse a slot directory.
pub fn decode_slots(buf: &[u8]) -> Result<Vec<SlotEntry>> {
    if buf.len() < 4 {
        return Err(DecodeError::Truncated.into());
    }
    let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut pos = 4;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if pos + 17 > buf.len() {
            return Err(DecodeError::Truncated.into());
        }
        let row_id = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        let off = u32::from_le_bytes(buf[pos + 8..pos + 12].try_into().unwrap());
        let len = u32::from_le_bytes(buf[pos + 12..pos + 16].try_into().unwrap());
        let live = buf[pos + 16] != 0;
        pos += 17;
        out.push(SlotEntry { row_id, off, len, live });
    }
    Ok(out)
}

fn checksum(data: &[u8]) -> u32 {
    let mut s: u32 = 0x811c9dc5;
    for &b in data {
        s = s.wrapping_mul(0x01000193).wrapping_add(b as u32);
    }
    s
}

/// Serialize a whole database (used by `checkpoint`).
pub fn encode_database(db: &Database) -> Vec<u8> {
    let mut entries: Vec<(u8, u32, Vec<u8>)> = Vec::new();
    // Schema record.
    entries.push((0, 0, db.schema.encode()));
    // Dict pages.
    // We locate dict pages via the schema's dict_page pointers.
    for (ci, col) in db.schema.columns.iter().enumerate() {
        if let Some(dict_pid) = col.dict_page {
            if let Some(page) = db.pager.get(dict_pid) {
                entries.push((2, dict_pid, page.buf.to_vec()));
            }
            let _ = ci;
        }
    }
    // Data pages.
    for page in db.pager.data_pages() {
        let mut body = Vec::new();
        let body_bytes = page.buf.to_vec();
        let slot_bytes = page
            .slot_dir
            .as_ref()
            .map_or(Vec::new(), |d| encode_slots(d));
        body.extend_from_slice(&(body_bytes.len() as u32).to_le_bytes());
        body.extend_from_slice(&body_bytes);
        body.extend_from_slice(&slot_bytes);
        entries.push((1, page.id, body));
    }
    // Index records (one per indexed column).
    for (ci, idx) in db.indexes.iter().enumerate() {
        if let Some(idx) = idx {
            entries.push((3, ci as u32, encode_index(idx, &db.pager)));
        }
    }
    // Fsm, rowid, zonemap records.
    entries.push((4, 0, db.fsm.encode()));
    entries.push((5, 0, db.rowid_map.encode()));
    let mut zm_bytes = Vec::new();
    zm_bytes.extend_from_slice(&(db.zone_maps.len() as u32).to_le_bytes());
    for zm in &db.zone_maps {
        let enc = zm.encode();
        zm_bytes.extend_from_slice(&(enc.len() as u32).to_le_bytes());
        zm_bytes.extend_from_slice(&enc);
    }
    entries.push((6, 0, zm_bytes));
    // next_row_id record (kind 7 reuse: store as a tiny record kind 0xff).
    let mut nrid = Vec::new();
    nrid.extend_from_slice(&db.next_row_id.to_le_bytes());
    entries.push((0xfe, 0, nrid));

    // Now lay out the directory + bodies.
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    let dir_start = out.len();
    let dir_len = entries.len() * 17;
    out.resize(dir_start + dir_len, 0);
    let mut body_off = 0u32;
    for (i, (kind, id, body)) in entries.iter().enumerate() {
        let base = dir_start + i * 17;
        out[base] = *kind;
        out[base + 1..base + 5].copy_from_slice(&id.to_le_bytes());
        out[base + 5..base + 9].copy_from_slice(&body_off.to_le_bytes());
        out[base + 9..base + 13].copy_from_slice(&(body.len() as u32).to_le_bytes());
        out[base + 13..base + 17].copy_from_slice(&checksum(body).to_le_bytes());
        body_off += body.len() as u32;
    }
    for (_, _, body) in &entries {
        out.extend_from_slice(body);
    }
    out
}

fn encode_index(idx: &IndexCache, pager: &Pager) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&idx.dict_page.to_le_bytes());
    out.extend_from_slice(&(idx.entries.len() as u32).to_le_bytes());
    for e in &idx.entries {
        out.extend_from_slice(&e.value_id.to_le_bytes());
        out.extend_from_slice(&e.gen.to_le_bytes());
        // Store the byte offset within the dict page (ptr - base), so decode can
        // re-capture the pointer into the freshly decoded dict page.
        if let Some(page) = pager.get(idx.dict_page) {
            let base = page.raw_ptr() as usize;
            let off = (e.ptr as usize).saturating_sub(base) as u32;
            out.extend_from_slice(&off.to_le_bytes());
        } else {
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        out.extend_from_slice(&(e.row_ids.len() as u32).to_le_bytes());
        for &r in &e.row_ids {
            out.extend_from_slice(&r.to_le_bytes());
        }
    }
    out
}

/// Decode a `.sht` blob into a database.
pub fn decode(data: &[u8]) -> Result<Database> {
    if data.len() < 8 + 4 + 4 {
        return Err(DecodeError::Truncated.into());
    }
    if &data[0..8] != MAGIC {
        return Err(DecodeError::BadMagic.into());
    }
    let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(DecodeError::UnsupportedVersion(version).into());
    }
    let num_records = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let dir_start = 16;
    let dir_len = num_records * 17;
    if data.len() < dir_start + dir_len {
        return Err(DecodeError::Truncated.into());
    }
    let mut dir = Vec::with_capacity(num_records);
    for i in 0..num_records {
        let base = dir_start + i * 17;
        let kind = data[base];
        let id = u32::from_le_bytes(data[base + 1..base + 5].try_into().unwrap());
        let off = u32::from_le_bytes(data[base + 5..base + 9].try_into().unwrap());
        let len = u32::from_le_bytes(data[base + 9..base + 13].try_into().unwrap());
        let cs = u32::from_le_bytes(data[base + 13..base + 17].try_into().unwrap());
        dir.push(DirEntry { kind, id, off, len, checksum: cs });
    }
    let bodies_start = dir_start + dir_len;

    // First pass: collect bodies and validate checksums.
    let mut bodies: Vec<&[u8]> = Vec::with_capacity(num_records);
    for e in &dir {
        let start = bodies_start + e.off as usize;
        let end = start + e.len as usize;
        if end > data.len() {
            return Err(DecodeError::Truncated.into());
        }
        let body = &data[start..end];
        if checksum(body) != e.checksum {
            return Err(DecodeError::BadChecksum { page: e.id }.into());
        }
        bodies.push(body);
    }

    // Second pass: build the database. Schema first.
    let mut schema: Option<Schema> = None;
    let mut next_row_id = 1u64;
    let mut fsm = Fsm::new();
    let mut rowid_map = RowIdMap::new();
    let mut zone_maps: Vec<ZoneMap> = Vec::new();
    let mut pager = Pager::new();
    let mut index_records: Vec<(usize, &[u8])> = Vec::new();

    for (i, e) in dir.iter().enumerate() {
        let body = bodies[i];
        match e.kind {
            0 => {
                schema = Some(Schema::decode(body)?);
            }
            0xfe => {
                if body.len() >= 8 {
                    next_row_id = u64::from_le_bytes(body[0..8].try_into().unwrap());
                }
            }
            2 => {
                // dict page
                let _ = pager.alloc(PageKind::Dict, body.to_vec().into_boxed_slice());
            }
            1 => {
                // data page
                if body.len() < 4 {
                    return Err(DecodeError::Truncated.into());
                }
                let body_len = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
                if 4 + body_len > body.len() {
                    return Err(DecodeError::Truncated.into());
                }
                let body_bytes = &body[4..4 + body_len];
                let slot_bytes = &body[4 + body_len..];
                let slots = decode_slots(slot_bytes)?;
                let pid = pager.alloc(PageKind::Data, body_bytes.to_vec().into_boxed_slice());
                if let Some(page) = pager.get_mut(pid) {
                    page.slot_dir = Some(slots.into_boxed_slice());
                }
            }
            3 => {
                // index record; defer until dict pages exist
                index_records.push((e.id as usize, body));
            }
            4 => fsm = Fsm::decode(body),
            5 => rowid_map = RowIdMap::decode(body),
            6 => {
                // zonemap records: [u32 num_cols][for each: u32 len + bytes].
                if let Some(s) = &schema {
                    if body.len() >= 4 {
                        let ncol = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
                        let mut pos = 4;
                        for _ in 0..ncol {
                            if pos + 4 > body.len() {
                                break;
                            }
                            let l = u32::from_le_bytes(body[pos..pos + 4].try_into().unwrap()) as usize;
                            pos += 4;
                            if pos + l > body.len() {
                                break;
                            }
                            zone_maps.push(ZoneMap::decode(&body[pos..pos + l]));
                            pos += l;
                        }
                    }
                    let _ = s;
                }
            }
            _ => {}
        }
    }

    let schema = schema.ok_or(DecodeError::BadSchema)?;
    // Ensure zone_maps has one entry per column.
    while zone_maps.len() < schema.arity() {
        zone_maps.push(ZoneMap::default());
    }
    let mut db = Database::new(schema);
    db.pager = pager;
    db.fsm = fsm;
    db.rowid_map = rowid_map;
    db.zone_maps = zone_maps;
    db.next_row_id = next_row_id;

    // Build index caches now that dict pages are decoded, capturing pointers.
    for (ci, body) in index_records {
        let idx = decode_index(body, &db);
        if ci < db.indexes.len() {
            db.indexes[ci] = Some(idx);
        }
    }

    // Build dictionary mirrors from the decoded dict pages.
    for (ci, col) in db.schema.columns.iter().enumerate() {
        if let Some(dict_pid) = col.dict_page {
            if let Some(page) = db.pager.get(dict_pid) {
                if let Ok(dict) = Dict::from_page(dict_pid, &page.buf, page.gen) {
                    db.dicts[ci] = Some(dict);
                }
            }
        }
    }

    Ok(db)
}

fn decode_index(body: &[u8], db: &Database) -> IndexCache {
    if body.len() < 8 {
        return IndexCache::empty(String::new());
    }
    let dict_page = u32::from_le_bytes(body[0..4].try_into().unwrap());
    let n = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    let column = db
        .schema
        .columns
        .get(0)
        .map(|c| c.name.clone())
        .unwrap_or_default();
    let mut idx = IndexCache::with_dict(column, dict_page);
    let mut pos = 8;
    for _ in 0..n {
        if pos + 20 > body.len() {
            break;
        }
        let value_id = u32::from_le_bytes(body[pos..pos + 4].try_into().unwrap());
        let gen = u64::from_le_bytes(body[pos + 4..pos + 12].try_into().unwrap());
        let off = u32::from_le_bytes(body[pos + 12..pos + 16].try_into().unwrap());
        let rc = u32::from_le_bytes(body[pos + 16..pos + 20].try_into().unwrap()) as usize;
        pos += 20;
        let mut row_ids = Vec::with_capacity(rc);
        for _ in 0..rc {
            if pos + 8 > body.len() {
                break;
            }
            row_ids.push(u64::from_le_bytes(body[pos..pos + 8].try_into().unwrap()));
            pos += 8;
        }
        let ptr = if let Some(page) = db.pager.get(dict_page) {
            // Safety: `off` is a byte offset recorded when the index was
            // serialized, and the dict page has just been decoded.
            unsafe { page.raw_ptr().add(off as usize) }
        } else {
            std::ptr::null()
        };
        idx.entries.push(crate::index::IndexEntry {
            value_id,
            ptr,
            gen,
            row_ids,
        });
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ColKind, Column, Encoding, Schema};

    #[test]
    fn empty_database_round_trips() {
        let mut cols = vec![Column::row_id()];
        cols.push(Column::new("x", ColKind::Int, Encoding::Plain));
        let db = Database::new(Schema::new(cols));
        let bytes = encode_database(&db);
        let db2 = decode(&bytes).unwrap();
        assert_eq!(db2.schema, db.schema);
        assert_eq!(db2.pager.count(), 0);
    }

    #[test]
    fn body_round_trips() {
        let mut body = DataPageBody {
            num_rows: 2,
            regions: vec![ColumnRegion::new(ColKind::Int, Encoding::Plain)],
        };
        body.regions[0].data.extend_from_slice(&1i64.to_le_bytes());
        body.regions[0].data.extend_from_slice(&2i64.to_le_bytes());
        let bytes = encode_body(&body);
        let back = decode_body(&bytes).unwrap();
        assert_eq!(back.num_rows, 2);
        assert_eq!(back.regions.len(), 1);
        assert_eq!(back.regions[0].data.len(), 16);
    }
}
