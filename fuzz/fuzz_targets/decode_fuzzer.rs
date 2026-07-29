#![no_main]
//! Secondary harness: decode + verify only. This reaches the container decoder
//! and the cross-page verifier but does not run any script. It is bonus
//! coverage of the decode path; the primary `query_fuzzer` drives the full
//! pipeline.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(db) = schist::format::decode(data) {
        let _ = schist::verify::verify(&db);
    }
});
