#![no_main]
//! Primary/canonical harness: the input bytes are a combined blob of a `.sht`
//! database and an operation script (a 4-byte little-endian script length,
//! then the script, then the database). This drives the whole engine end to
//! end — decode the container, run the cross-page verifier, then execute the
//! script (insert / update / delete / scan / index_scan / join / compact /
//! checkpoint / agg) against the decoded database.
//!
//! Checksum verification is performed by the decoder (it is part of the
//! container format), and the cross-page verifier gates what reaches the
//! script runner.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    schist::run_combined(data);
});
