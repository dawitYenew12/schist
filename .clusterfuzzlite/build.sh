#!/bin/bash -eu
# ClusterFuzzLite build entry point for schist.
#
# Builds every fuzz target into $OUT with the requested sanitizer, and packages
# each target's seed corpus and the shared dictionary alongside it.

SRC="${SRC:-$(pwd)}"
OUT="${OUT:-${SRC}/out}"
mkdir -p "$OUT"
cd "${SRC}"

# The OSS-Fuzz/ClusterFuzzLite Rust base image already provides cargo-fuzz.
# (No install step here: the build environment has no network access.)

FUZZ_TARGETS="query_fuzzer decode_fuzzer script_fuzzer"

CARGO_FUZZ_BUILD_FLAGS="${CARGO_FUZZ_BUILD_FLAGS:---release}"
for target in $FUZZ_TARGETS; do
  echo "[build] ${target}"
  cargo +nightly fuzz build $CARGO_FUZZ_BUILD_FLAGS "$target"
done

# Locate the built binaries (triple dir varies by toolchain) and stage them.
BUILD_DIR="${SRC}/fuzz/target"
TRIPLE_DIR=$(find "$BUILD_DIR" -maxdepth 1 -type d -name "*-unknown-linux-gnu" | head -n1)
TRIPLE_DIR="${TRIPLE_DIR:-${BUILD_DIR}/x86_64-unknown-linux-gnu}"

for target in $FUZZ_TARGETS; do
  bin_path="${TRIPLE_DIR}/release/${target}"
  if [ -f "$bin_path" ]; then
    cp "$bin_path" "${OUT}/${target}"
    echo "[stage] ${target} -> \$OUT/${target}"
  else
    echo "[warn] built binary not found for ${target} at ${bin_path}" >&2
  fi

  # Package the per-target seed corpus, if present.
  corpus_dir="${SRC}/fuzz/corpus/${target}"
  if [ -d "$corpus_dir" ]; then
    (cd "$corpus_dir" && zip -q -r "${OUT}/${target}_seed_corpus.zip" . || true)
  fi

  # Copy the shared dictionary for each target.
  if [ -f "${SRC}/fuzz/dictionary.txt" ]; then
    cp "${SRC}/fuzz/dictionary.txt" "${OUT}/${target}.dict"
  fi
done
