# Evaluation report (arxiv, policy=llm:local, model=qwen2.5:7b)

## Retrieval (paper-level)

| mode | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| bm25 | 0.898 | 0.926 | 0.954 | 0.863 |
| dense | 0.898 | 0.972 | 1.0 | 0.921 |
| hybrid | 0.926 | 0.972 | 0.972 | 0.963 |

## End-to-end

| metric | value |
|---|---|
| status_ok | 0.864 |
| keyword_score | 0.735 |
| citation_validity | 0.974 |
| gold_cited | 0.833 |
| citation_precision | 0.78 |
| tools_ok | 0.864 |
| faithfulness | 0.965 |
| steps | 1.955 |
| seconds | 9.588 |
| n | 22 |

| type | n | status_ok | keyword_score | gold_cited | tools_ok |
|---|---|---|---|---|---|
| analysis | 2 | 0.0 | 1.0 | 0.0 | 0.5 |
| conflict | 3 | 0.667 | 0.667 | 0.667 | 1.0 |
| factual | 12 | 1.0 | 0.708 | 1.0 | 1.0 |
| figure | 2 | 1.0 | None | 0.5 | 0.5 |
| unanswerable | 3 | 1.0 | None | None | 0.667 |

### Per question

| id | status | ok | kw | tools |
|---|---|---|---|---|
| f1 | PARTIAL | ✓ | 0.0 | search_papers |
| f2 | SUPPORTED | ✓ | 1.0 | search_papers |
| f3 | SUPPORTED | ✓ | 0.5 | search_papers |
| f4 | SUPPORTED | ✓ | 1.0 | search_papers |
| f5 | PARTIAL | ✓ | 1.0 | search_papers |
| f6 | SUPPORTED | ✓ | 1.0 | search_papers |
| f7 | PARTIAL | ✓ | 0.0 | search_papers |
| f8 | PARTIAL | ✓ | 0.0 | search_papers |
| f9 | SUPPORTED | ✓ | 1.0 | search_papers |
| f10 | SUPPORTED | ✓ | 1.0 | search_papers |
| f11 | SUPPORTED | ✓ | 1.0 | search_papers |
| f12 | SUPPORTED | ✓ | 1.0 | search_papers |
| c1 | SUPPORTED | ✗ | 0.0 | search_papers |
| c2 | PARTIAL | ✓ | 1.0 | search_papers |
| c3 | CONFLICTING | ✓ | 1.0 | search_papers |
| u1 | INSUFFICIENT | ✓ | - | search_papers |
| u2 | INSUFFICIENT | ✓ | - | search_papers |
| u3 | INSUFFICIENT | ✓ | - |  |
| g1 | SUPPORTED | ✓ | - | find_figures |
| g2 | SUPPORTED | ✓ | - | search_papers |
| a1 | UNVERIFIED | ✗ | 1.0 | run_surrogate_experiment |
| a2 | UNVERIFIED | ✗ | 1.0 | run_surrogate_experiment |

## Robustness

- **abstain_or_hedge_when_gold_removed**: 0.417
- **conflict_detected_when_injected**: 0.667
