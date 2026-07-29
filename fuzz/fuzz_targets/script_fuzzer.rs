#![no_main]
//! Secondary harness: decode a tiny fixed empty database, then run the input
//! bytes as an operation script. This exercises the script interpreter and the
//! mutation/compaction/query layers without requiring a structured container.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let db = schist::Database::new(schist::schema::Schema::new(vec![
        schist::schema::Column::row_id(),
        schist::schema::Column::new("x", schist::schema::ColKind::Int, schist::schema::Encoding::Plain),
    ]));
    // The fuzz input is treated as the script source.
    let mut db = db;
    if let Ok(src) = std::str::from_utf8(data) {
        let _ = schist::script::run_script(&mut db, src);
    }
});
