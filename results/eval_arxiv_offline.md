# Evaluation report (arxiv, policy=offline)

## Retrieval (paper-level)

| mode | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| bm25 | 0.898 | 0.926 | 0.954 | 0.863 |
| dense | 0.898 | 0.972 | 1.0 | 0.921 |
| hybrid | 0.926 | 0.972 | 0.972 | 0.963 |

## End-to-end

| metric | value |
|---|---|
| status_ok | 0.773 |
| keyword_score | 0.529 |
| citation_validity | 1.0 |
| gold_cited | 0.889 |
| citation_precision | 0.789 |
| tools_ok | 1.0 |
| faithfulness | None |
| steps | 1.591 |
| seconds | 2.304 |
| n | 22 |

| type | n | status_ok | keyword_score | gold_cited | tools_ok |
|---|---|---|---|---|---|
| analysis | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| conflict | 3 | 0.667 | 0.667 | 0.667 | 1.0 |
| factual | 12 | 0.667 | 0.417 | 0.917 | 1.0 |
| figure | 2 | 1.0 | None | 1.0 | 1.0 |
| unanswerable | 3 | 1.0 | None | None | 1.0 |

### Per question

| id | status | ok | kw | tools |
|---|---|---|---|---|
| f1 | INSUFFICIENT | ✗ | 0.0 | extract_numeric_claims, search_papers |
| f2 | INSUFFICIENT | ✗ | 0.5 | extract_numeric_claims, search_papers |
| f3 | INSUFFICIENT | ✗ | 0.0 | search_papers |
| f4 | PARTIAL | ✓ | 0.5 | search_papers |
| f5 | SUPPORTED | ✓ | 0.0 | extract_numeric_claims, search_papers |
| f6 | INSUFFICIENT | ✗ | 0.0 | search_papers |
| f7 | SUPPORTED | ✓ | 0.0 | extract_numeric_claims, search_papers |
| f8 | SUPPORTED | ✓ | 1.0 | search_papers |
| f9 | PARTIAL | ✓ | 0.0 | search_papers |
| f10 | SUPPORTED | ✓ | 1.0 | search_papers |
| f11 | SUPPORTED | ✓ | 1.0 | search_papers |
| f12 | PARTIAL | ✓ | 1.0 | extract_numeric_claims, search_papers |
| c1 | INSUFFICIENT | ✗ | 1.0 | extract_numeric_claims, search_papers |
| c2 | PARTIAL | ✓ | 1.0 | search_papers |
| c3 | PARTIAL | ✓ | 0.0 | search_papers |
| u1 | INSUFFICIENT | ✓ | - | search_papers |
| u2 | INSUFFICIENT | ✓ | - | search_papers |
| u3 | INSUFFICIENT | ✓ | - | search_papers |
| g1 | PARTIAL | ✓ | - | find_figures, inspect_figure, search_papers |
| g2 | PARTIAL | ✓ | - | find_figures, inspect_figure, search_papers |
| a1 | PARTIAL | ✓ | 1.0 | extract_numeric_claims, run_surrogate_experiment, search_papers |
| a2 | PARTIAL | ✓ | 1.0 | run_surrogate_experiment, search_papers |
