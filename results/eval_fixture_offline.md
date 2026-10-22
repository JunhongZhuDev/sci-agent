# Evaluation report (fixture, policy=offline)

## Retrieval (paper-level)

| mode | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| bm25 | 1.0 | 1.0 | 1.0 | 1.0 |
| dense | 1.0 | 1.0 | 1.0 | 1.0 |
| hybrid | 1.0 | 1.0 | 1.0 | 1.0 |

## End-to-end

| metric | value |
|---|---|
| status_ok | 1.0 |
| keyword_score | 1.0 |
| citation_validity | 1.0 |
| gold_cited | 1.0 |
| citation_precision | 0.617 |
| tools_ok | 1.0 |
| faithfulness | None |
| steps | 2.0 |
| seconds | 0.006 |
| n | 10 |

| type | n | status_ok | keyword_score | gold_cited | tools_ok |
|---|---|---|---|---|---|
| conflict | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| factual | 4 | 1.0 | 1.0 | 1.0 | 1.0 |
| figure | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| table | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| unanswerable | 2 | 1.0 | None | None | 1.0 |

### Per question

| id | status | ok | kw | tools |
|---|---|---|---|---|
| xf1 | SUPPORTED | ✓ | 1.0 | extract_numeric_claims, search_papers |
| xf2 | PARTIAL | ✓ | 1.0 | search_papers |
| xf3 | CONFLICTING | ✓ | 1.0 | extract_numeric_claims, search_papers |
| xf4 | SUPPORTED | ✓ | 1.0 | search_papers |
| xc1 | CONFLICTING | ✓ | 1.0 | extract_numeric_claims, search_papers |
| xc2 | SUPPORTED | ✓ | 1.0 | extract_numeric_claims, search_papers |
| xu1 | INSUFFICIENT | ✓ | - | search_papers |
| xu2 | INSUFFICIENT | ✓ | - | search_papers |
| xg1 | SUPPORTED | ✓ | 1.0 | extract_numeric_claims, find_figures, inspect_figure, search_papers |
| xt1 | SUPPORTED | ✓ | 1.0 | extract_numeric_claims, find_figures, inspect_figure, search_papers |

## Robustness

- **abstain_or_hedge_when_gold_removed**: 1.0
- **conflict_detected_when_injected**: 1.0
