# Reproducibility harness for the md crate (week2/md/).
#
# `make reproduce` runs the crate in release mode with its default flags,
# from this directory, writing its output under week2/artifacts/.
#
# Usage:
#     make reproduce

ARTIFACTS := artifacts
RUN_LOG   := $(ARTIFACTS)/run.log

.PHONY: reproduce

reproduce:
	mkdir -p $(ARTIFACTS)
	cargo run --quiet --release --manifest-path md/Cargo.toml > $(RUN_LOG) 2>&1
	@echo "wrote $(RUN_LOG)"
