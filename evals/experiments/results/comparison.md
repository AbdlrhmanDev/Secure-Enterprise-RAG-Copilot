# Experiment comparison

| Experiment | Recall@K | Precision@K | MRR | nDCG@K | ACL leaks | Correct | Grounded | Citation | Refusal | Retr p95 ms | Rerank p95 ms | p50 ms | p95 ms | USD/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_bm25 | 0.6633 | 0.0663 | 0.5758 | 0.597 | 0 | 0.6327 | 1.0 | 1.0 | 1.0 | 14.8 | 0.0 | 1441.1 | 2582.9 | 0.000401 |
| 02_dense | 0.949 | 0.0949 | 0.8893 | 0.9039 | 0 | 0.9286 | 1.0 | 0.989 | 1.0 | 65.5 | 0.0 | 1572.8 | 2816.3 | 0.000494 |
| 03_hybrid_rrf | 0.949 | 0.0949 | 0.775 | 0.8156 | 0 | 0.8571 | 1.0 | 0.9709 | 1.0 | 71.3 | 0.0 | 1577.8 | 2719.9 | 0.000483 |
| 04_hybrid_weighted | 0.9592 | 0.0959 | 0.8601 | 0.8845 | 0 | 0.949 | 0.9785 | 1.0 | 1.0 | 1058.5 | 0.0 | 1681.8 | 2706.1 | 0.000479 |
| 05_dense_rerank | 0.9592 | 0.0959 | 0.8967 | 0.9119 | 0 | 0.9184 | 1.0 | 0.9889 | 1.0 | 86.9 | 1017.2 | 2266.2 | 4220.0 | 0.000503 |
| 06_hybrid_rrf_rerank | 0.9592 | 0.0959 | 0.9061 | 0.9193 | 0 | 0.949 | 1.0 | 0.9946 | 1.0 | 63.4 | 704.7 | 1987.3 | 2976.3 | 0.000479 |

K = 10. 136 cases (98 answerable, 38 that must be refused). Embedder: BAAI/bge-small-en-v1.5. Generator: gpt-5-mini. Judge: gpt-5-mini. Latency scope: end_to_end.

Measured 2026-10-09 with `OPENAI_REASONING_EFFORT=low` for generation and judging. All six runs use the full 136-case dataset (no limit). Query latency and cost exclude judge calls. Cost estimates use $0.25 / million input tokens and $2.00 / million output tokens and exclude infrastructure.
