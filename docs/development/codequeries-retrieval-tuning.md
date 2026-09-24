# CodeQueries retrieval tuning

The current defaults were calibrated with live `BAAI/bge-m3` embeddings and live
`BAAI/bge-reranker-v2-m3` cross-encoder scores. Weighted RRF selects up to 20 candidates, the
reranker determines their final order, and its normalized score supplies the absolute relevance
cutoff. The objective is maximum positive-class F1 while keeping recall above 80%.

## Experiment

- Validation: 600 balanced examples, seed `20260914`, fingerprint
  `52426d13119636d6330f0a8658960d3dfddfba6ff306d4eb6bf75898c610861e`.
- Held-out test: 100 balanced examples, seed `20260915`, fingerprint
  `a733315c5c6a724f55e137549e9af1b7548195c7d7c92fa5a2e1a9644ce3057d`.
- Dense-to-keyword ratios: 0.001 through 100, including 1:1.
- Cutoffs: quantile search followed by exact observed reranker scores.
- Candidate budgets: 32 per retrieval channel, 20 for reranking, five final chunks.

The ignored cache and sweep outputs are under `apps/api/.data/retrieval-tuning`. From `apps/api`,
the reproducible stages are:

```bash
python -m scripts.tune_codequeries collect --split validation --seed 20260914 --count 600
python -m scripts.tune_codequeries rerank --split validation --seed 20260914 --count 600
python -m scripts.tune_codequeries sweep --split validation --seed 20260914 \
  --thresholds exact --relative-thresholds 0.8,0.85,0.855,0.86,0.9 --min-recall 0.8
```

## Selected calibration

The validation-only maximum under the recall constraint was threshold `0.0004991053`, with 31.51%
precision, 83.38% recall, and 45.74% F1. It produced 78% recall on the held-out sample, so it did not
meet the required recall floor consistently.

Further calibration showed that most false positives were weak followers of a much stronger first
result. The final gate combines an absolute floor of **0.00017189** with a query-relative floor of
**0.855**: a chunk must score at least 85.5% as highly as the query's best candidate. This preserves
the five-chunk maximum and returns several chunks when their reranker evidence is genuinely close.

This combination maximizes pooled F1 among the tested absolute and relative cutoffs while keeping
recall above 80% on both samples:

| Split      | Precision | Recall |     F1 |
| ---------- | --------: | -----: | -----: |
| Validation |    47.25% | 87.38% | 61.34% |
| Held-out   |    47.00% | 94.00% | 62.67% |
| Pooled     |    47.22% | 88.27% | 61.52% |

The reranker produces one-label normalized classifier scores concentrated near zero. The small
cutoff is therefore expected and must not be interpreted as a percentage or probability.

## Weight calibration

Every tested ratio from 0.001:1 through 100:1 produced the same metrics at the selected cutoff.
Most CodeQueries examples have fewer than the 20 rerank candidates, so RRF weights rarely remove a
chunk before cross-encoder scoring. Equal **1.0 vector / 1.0 keyword** weights remain the default
because they sit on the stable plateau and preserve both retrieval channels.

The previous absolute-only reranker gate reached 45.55% validation F1 and 47.80% held-out F1. The
relative gate improves those results by 15.79 and 14.87 percentage points while retaining 87.38%
and 94% recall. The earlier dense-cosine gate at `0.3893` reached only 74.46% validation recall and
74% held-out recall.
