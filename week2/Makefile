# Reproducibility harness for the md crate (week2/md/).
#
# `make reproduce` runs the crate in release mode with its default flags,
# from this directory, writing its output files (run.json, traj.jsonl)
# under week2/artifacts/.
#
# Usage:
#     make reproduce

ARTIFACTS := artifacts

.PHONY: reproduce

reproduce:
	mkdir -p $(ARTIFACTS)
	cargo run --quiet --release --manifest-path md/Cargo.toml -- run --out $(ARTIFACTS)
