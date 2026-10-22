# SciAgent — an agentic research assistant for ML in CFD/FEM simulation

A small agentic AI system that collects papers on **machine learning for computational fluid
dynamics and finite element simulation** (PINNs, neural operators, graph-network simulators,
learned turbulence closures), indexes their text, figures and tables, and answers research
questions with cited evidence. The agent chooses its own tools — corpus search, context reading,
figure inspection with a vision model, numeric-claim extraction, live arXiv search, Python, and
a built-in ML experiment — and reports when evidence is missing or conflicting.

Why this topic: the literature genuinely disagrees. Papers report 10–1000× speed-ups for learned
solvers, while others argue that these comparisons use weak baselines, or that PINNs do not beat
FEM. That makes it a good test of whether a RAG agent can surface conflicts instead of averaging them away.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY=...; neither -> offline baseline agent

make collect      # ~5 min: arXiv search, relevance curation, PDF download  -> data/papers.jsonl, data/pdfs/
make ingest       # ~2-5 min: chunks, figure PNGs, tables                   -> data/chunks.jsonl, data/figures/
make index        # hybrid BM25 + dense index                               -> data/index.pkl
make ask Q="Do PINNs beat the finite element method?"
make eval         # retrieval + end-to-end + robustness evaluation        -> results/eval_*.md/json
make examples     # example Q&A with tool traces                          -> examples/
make test         # unit/integration tests (offline, no key needed)
```

**Google Colab:** open `sciagent_colab.ipynb` in Colab (upload it, or File → Upload notebook),
upload `sci-agent.zip` when prompted, and run the cells top to bottom. The API key goes into Colab
*Secrets*. TensorFlow, scikit-learn and sentence-transformers are preinstalled there.

**TensorFlow:** the ML component also trains a Keras MLP surrogate on the same POD coefficients
when you pass `--tf` (`python -m sciagent.ml.burgers --tf`) or set `SCIAGENT_TF=1`. The agent can
request it with `use_tensorflow=true`. Without TensorFlow it is skipped.

No API key? Everything still runs: `make eval-offline` uses a deterministic heuristic agent,
and `make eval-fixture` runs the whole pipeline on a small synthetic corpus, without network access.

Configuration lives in `sciagent/config.py`. Environment variables: `SCIAGENT_LLM`
(`auto|anthropic|openai|offline`), `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `SCIAGENT_EMBED`
(sentence-transformers model, default `BAAI/bge-small-en-v1.5`), `SCIAGENT_RERANK`
(optional cross-encoder).

## Repository layout

```
sciagent/
  collect.py        arXiv queries + seed IDs -> dedupe -> TF-IDF relevance curation -> PDF download
  ingest.py         section-aware chunking, figure captions + rendered figure PNGs, tables (pdfplumber)
  index.py          own BM25 + dense (bge / LSA fallback) + RRF fusion, filters, per-paper diversity cap
  tools.py          the agent's tools (JSON schemas + implementations), evidence bookkeeping
  agent.py          LLM tool-use loop, offline baseline agent, citation verifier
  llm.py            thin Anthropic / OpenAI adapter (tool use + vision)
  ml/burgers.py     ML component: spectral Burgers solver + POD-ridge/POD-MLP surrogates
eval/
  questions.jsonl          22 questions: factual, conflict, unanswerable, figure, analysis
  fixture_questions.jsonl  10 questions for the synthetic corpus
  run_eval.py              retrieval ablation, end-to-end metrics, robustness suite, LLM judge
tests/                     16 tests (pytest), synthetic fixture corpus
examples/                  example questions and outputs
docs/ARCHITECTURE.md       architecture and design choices
docs/PRESENTATION.md       slide plan + likely questions; docs/SciAgent_presentation.pdf = slides
docs/AI_USAGE.md           how coding agents were used to build this
```

## Example

```
$ make ask Q="Are learned neural operators really faster than numerical solvers? Compare the claims."
  -> extract_numeric_claims({"topic": "..."})
  -> search_papers({"query": "...", "k": 6})

Evidence from the corpus points in different directions:
- Most studies compare against weak baselines: when the numerical solver is run at a resolution
  that matches the learned model's accuracy, reported speedups shrink by one to three orders of
  magnitude, and several learned solvers are slower. [fx-crit#abs]
- On two-dimensional turbulent flow at viscosity 1e-4 the operator is about 1000 times faster than
  the pseudo-spectral solver used to generate training data ... [fx-fno#abs]
STATUS: CONFLICTING
CONFIDENCE: low
```

(Offline policy on the synthetic fixture corpus. See `examples/` for more, including the figure,
abstention and own-experiment cases.)

## Evaluation results

See `results/`. Two kinds of result are included:

1. **ML component — own experiment** (`results/burgers_surrogate.*`, real numbers, seed 0):

| method | rel. L2 in-dist | rel. L2 OOD (lower viscosity) | ms/query | offline cost (s) |
|---|---|---|---|---|
| reference solver (256 modes) | 0 (ground truth) | 0 | ~23 | 0 |
| coarse solver (32 modes) | 0.49 % | 5.9 % | ~13 | 0 |
| POD + ridge (poly-2, 5-fold CV) | 20.3 % | 45.7 % | 0.02 | 9 |
| POD + MLP (sklearn, 5-fold CV) | 16.0 % | 35.6 % | 0.02 | 25 |
| POD + MLP (Keras, `--tf`) | 21.7 % | 50.6 % | 4.4 | 74 |

   The sklearn MLP is ~650× faster per query than the coarse classical solver, but that solver is ~33× more
   accurate in-distribution and does not collapse out of distribution. The surrogate only breaks
   even after ~400–1,100 queries (the Keras MLP is slower per query due to framework call overhead). A small-scale reproduction of the "weak baselines" argument.
   (Caveat: the solver timings include Python loop overhead, which flatters the coarse solver's
   cost less than a compiled code would.)

2. **Agent evaluation** on the real corpus (40 papers, 22 questions). Same tools and verifier, two policies:

| metric | rule-based baseline | local LLM agent (Qwen2.5-7B, Ollama) |
|---|---|---|
| status accuracy | 77 % | 86 % |
| factual questions correct (12) | 8 | 12 |
| key-fact coverage | 0.53 | 0.74 |
| citation validity | 100 % | 97 % (1 fabricated citation removed) |
| analysis questions (2) | 2 | 0 (answers cited nothing, so marked UNVERIFIED) |
| seconds per question | ~2 | ~10 |

   Retrieval (hybrid): Recall@5 0.97, MRR 0.963 (BM25 alone: 0.863).
   Robustness (LLM): hedged in 5 of 12 cases after the gold paper was deleted; reported 2 of 3 injected
   contradictions. The baseline over-abstains (coverage threshold set on the fixture); the LLM under-explores
   (one search, misses counter-evidence) and misreads numbers. Faithfulness 0.97 was judged by the same 7B
   model, so treat it with caution. `results/eval_fixture_offline.md` is only a CI smoke test.

## Limitations

- `run_python` runs in a subprocess with a timeout, not a security sandbox. Use a container for untrusted use.
- Figure crops use a caption-position heuristic and can include body text or miss multi-panel figures.
- Equations are not parsed (text extraction flattens LaTeX). Table extraction is best-effort.
- The evaluation set is small (22 questions) and hand-written. Keyword matching is a coarse
  correctness proxy, which is why the optional LLM-as-judge faithfulness score is included.
- The offline agent's abstention threshold (0.6 IDF-weighted query coverage) was set by hand on the
  fixture set. With more labelled questions it should be tuned on a held-out split.
