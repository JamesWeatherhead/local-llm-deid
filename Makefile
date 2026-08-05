# Reproduce the offline self-test, the manuscript tables, and the tests with one command.
#
#   make selftest run the offline test stub over the synthetic fixture and score it
#   make tables   rebuild the corpus, checkpoint, and results tables
#   make ci       reproduce the 95% CI for PHI removed
#   make test     run the offline test suite
#   make configs  regenerate model_configs/<model-id>.json
#   make figures  render Figures 2-6 (SVG + CSV) from metrics_long.csv
#   make all      selftest + tables + ci + figures
#   make clean    remove the out/ directory
#
# Everything here is standard-library Python and runs offline. Override the
# interpreter with `make PY=python3.11 ...` if your python3 is elsewhere.

PY ?= python3
NOTES ?= fixtures/synthetic/notes
GOLD ?= fixtures/synthetic/gold
OUT ?= out

.PHONY: all selftest tables results checkpoints corpus ci test configs figures clean help

help:
	@echo "Targets: selftest, tables, results, checkpoints, corpus, ci, test, configs, figures, clean, all"

selftest:
	PYTHONPATH=src $(PY) -m deid.run_model --model-id offline-stub \
	    --notes-dir $(NOTES) --out-dir $(OUT)/predictions \
	    --protocol-dir protocol --offline-stub --name-file fixtures/synthetic/name_gazetteer.txt
	PYTHONPATH=src $(PY) -m deid.metrics \
	    --pred-dir $(OUT)/predictions --gold-dir $(GOLD) --out-dir $(OUT)
	$(PY) scripts/make_results_table.py --metrics $(OUT)/metrics_long.csv

checkpoints:
	$(PY) scripts/make_checkpoints_table.py

corpus:
	$(PY) scripts/make_corpus_table.py --gold-dir $(GOLD)

results:
	@test -f $(OUT)/metrics_long.csv || $(MAKE) selftest
	$(PY) scripts/make_results_table.py --metrics $(OUT)/metrics_long.csv

ci:
	@test -f $(OUT)/per_doc_long.csv || $(MAKE) selftest
	$(PY) scripts/bootstrap_ci.py --per-doc $(OUT)/per_doc_long.csv

figures:
	@test -f $(OUT)/metrics_long.csv || $(MAKE) selftest
	$(PY) scripts/make_figures.py --metrics $(OUT)/metrics_long.csv --out-dir $(OUT)/figures

tables: checkpoints corpus results

test:
	$(PY) -m unittest discover -s tests -v

configs:
	$(PY) model_configs/build_configs.py

clean:
	rm -rf $(OUT)

all: selftest tables ci figures
