PY ?= python

.PHONY: all collect ingest index ask eval eval-offline eval-fixture examples experiment test

all: collect ingest index

collect:            ## arXiv search + curation + PDF download -> data/papers.jsonl
	$(PY) -m sciagent.collect --max-per-query 25 --keep 40

ingest:             ## PDFs -> text/figure/table chunks + figure PNGs
	$(PY) -m sciagent.ingest

index:              ## hybrid BM25 + dense index
	$(PY) -m sciagent.index

ask:                ## make ask Q="your question"
	$(PY) -m sciagent.agent "$(Q)" --trace

eval:               ## full evaluation with the LLM agent (+ judge)
	$(PY) eval/run_eval.py --judge

eval-offline:       ## deterministic baseline agent on the real corpus
	$(PY) eval/run_eval.py --provider offline

eval-fixture:       ## CI: synthetic corpus, no network, no key
	$(PY) eval/run_eval.py --fixture --provider offline --lsa

examples:
	$(PY) examples/run_examples.py

experiment:         ## the Burgers surrogate-vs-solver experiment on its own
	$(PY) -m sciagent.ml.burgers

test:
	$(PY) -m pytest -q tests
