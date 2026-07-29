//! Diagnostics and pretty-printing.
//!
//! Small helpers used by the CLI and by tests to inspect a database's shape:
//! the page set, the row count, the index/dictionary state, and a tabular
//! dump of the rows.

use crate::query::{scan_all, Row};
use crate::Database;

/// A one-line summary of the database.
pub fn summary(db: &Database) -> String {
    format!(
        "schist db: {} pages, {} rows, next_row_id={}, inserts={} updates={} deletes={} compactions={} scans={} index_scans={}",
        db.pager.count(),
        db.rowid_map.len(),
        db.next_row_id,
        db.stats.inserts,
        db.stats.updates,
        db.stats.deletes,
        db.stats.compactions,
        db.stats.scans,
        db.stats.index_scans,
    )
}

/// List every page with its kind, generation, buffer size, and slot count.
pub fn page_list(db: &Database) -> Vec<String> {
    let mut out = Vec::new();
    for page in db.pager.iter() {
        out.push(format!(
            "page {:>3} {:>8} gen={:<3} buf={:<5} slots={}",
            page.id,
            page.kind.name(),
            page.gen,
            page.buf.len(),
            page.slot_count(),
        ));
    }
    out
}

/// List each dictionary mirror's page, generation, and entry count.
pub fn dict_list(db: &Database) -> Vec<String> {
    let mut out = Vec::new();
    for (ci, d) in db.dicts.iter().enumerate() {
        if let Some(d) = d {
            out.push(format!(
                "col {} dict page={} gen={} entries={}",
                ci, d.page_id, d.gen, d.len()
            ));
        }
    }
    out
}

/// List each index cache's column, dict page, and entry count.
pub fn index_list(db: &Database) -> Vec<String> {
    let mut out = Vec::new();
    for idx in db.indexes.iter().flatten() {
        out.push(format!(
            "index col={} dict_page={} entries={}",
            idx.column,
            idx.dict_page,
            idx.len(),
        ));
    }
    out
}

/// Render a row as a tab-separated line.
pub fn render_row(row: &Row) -> String {
    let mut s = format!("{}", row.id);
    for v in &row.values {
        s.push('\t');
        s.push_str(&render_value(*v));
    }
    s
}

pub fn render_value(v: crate::value::Value) -> String {
    use crate::value::Value;
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Real(r) => r.to_string(),
        Value::Text(id) => format!("#{}", id),
    }
}

/// Dump all rows (via a full scan) as text.
pub fn dump_rows(db: &mut Database) -> Vec<String> {
    scan_all(db).iter().map(render_row).collect()
}

/// Full diagnostic text.
pub fn full_report(db: &mut Database) -> String {
    let mut out = String::new();
    out.push_str(&summary(db));
    out.push('\n');
    for line in page_list(db) {
        out.push_str(&line);
        out.push('\n');
    }
    for line in dict_list(db) {
        out.push_str(&line);
        out.push('\n');
    }
    for line in index_list(db) {
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str("--- rows ---\n");
    for line in dump_rows(db) {
        out.push_str(&line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutation::insert;
    use crate::schema::{ColKind, Column, Encoding, Schema};
    use crate::value::Value;

    #[test]
    fn summary_and_dump() {
        let mut db = Database::new(Schema::new(vec![
            Column::row_id(),
            Column::new("x", ColKind::Int, Encoding::Plain),
        ]));
        insert(&mut db, &[Value::Int(0), Value::Int(5)]).unwrap();
        let s = summary(&db);
        assert!(s.contains("1 rows"));
        let rows = dump_rows(&mut db);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("5"));
    }
}
