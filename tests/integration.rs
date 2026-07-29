//! End-to-end integration: build a database, round-trip it through the
//! container format, run scripts, and exercise compaction.

use schist::format::{decode, encode_database};
use schist::schema::{ColKind, Column, Encoding, Schema};
use schist::value::Value;
use schist::{run_combined, Database};

fn two_col_db() -> Database {
    let mut cols = vec![Column::row_id()];
    cols.push(Column::new("x", ColKind::Int, Encoding::Plain));
    Database::new(Schema::new(cols))
}

#[test]
fn round_trip_empty_database() {
    let db = two_col_db();
    let bytes = encode_database(&db);
    let db2 = decode(&bytes).unwrap();
    assert_eq!(db2.schema, db.schema);
    assert!(schist::verify::verify(&db2).is_ok());
}

#[test]
fn run_combined_with_script() {
    let mut db = two_col_db();
    schist::mutation::insert(&mut db, &[Value::Int(0), Value::Int(1)]).unwrap();
    schist::mutation::insert(&mut db, &[Value::Int(0), Value::Int(2)]).unwrap();
    let db_bytes = encode_database(&db);
    let script = b"scan; scan where x = 2";
    let mut blob = (script.len() as u32).to_le_bytes().to_vec();
    blob.extend_from_slice(script);
    blob.extend_from_slice(&db_bytes);
    // Should not panic.
    run_combined(&blob);
}

#[test]
fn compaction_preserves_live_rows_via_safe_scan() {
    let mut db = two_col_db();
    for i in 1..=5 {
        schist::mutation::insert(&mut db, &[Value::Int(0), Value::Int(i)]).unwrap();
    }
    schist::mutation::delete_where(&mut db, 1, schist::value::CmpOp::Eq, Value::Int(3)).unwrap();
    schist::compact::compact(&mut db).unwrap();
    let rows = schist::query::scan_all(&mut db);
    let xs: Vec<i64> = rows.iter().filter_map(|r| r.values[1].as_int()).collect();
    assert_eq!(xs, vec![1, 2, 4, 5]);
}

#[test]
fn text_column_dictionary_round_trip() {
    let mut cols = vec![Column::row_id()];
    let mut t = Column::new("name", ColKind::Text, Encoding::Dictionary);
    t.index_page = Some(0); // mark indexed; page allocated lazily
    cols.push(t);
    let mut db = Database::new(Schema::new(cols));
    let id = schist::mutation::intern_text(&mut db, 1, b"alice").unwrap();
    assert_eq!(id, 0);
    let bytes = encode_database(&db);
    let db2 = decode(&bytes).unwrap();
    assert!(schist::verify::verify(&db2).is_ok());
}
