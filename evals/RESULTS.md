# Eval results

Notes:
- Agent vs pipeline (iter 04): the agent fixes out-of-scope refusals (0 → 1.00 with the search
  nudge) and handles comparisons, but costs 2–4 LLM calls per question; the fixed pipeline is
  better value for single-fact questions. qwen3:4b-instruct is 6× faster than qwen3:8b with a
  similar judge score.

| date | label | target | model | k | hit@k | mrr | citation_correct | keywords | answered | refusal_correct | judge | failures | avg s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23 23:59 | retrieval-k5 | retrieval | qwen3:8b | 5 | 1.00 | 0.66 | – | – | – | – | – | – | 0.0 |
| 2026-10-23 23:59 | retrieval-k1 | retrieval | qwen3:8b | 1 | 0.38 | 0.38 | – | – | – | – | – | – | 0.0 |
| 2026-10-23 23:59 | retrieval-k10 | retrieval | qwen3:8b | 10 | 1.00 | 0.66 | – | – | – | – | – | – | 0.0 |
| 2026-10-24 00:06 | baseline | pipeline | qwen3:8b | 5 | 1.00 | 0.66 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | – | 33.7 |
| 2026-10-24 00:20 | pipeline-11 | pipeline | qwen3:8b | 5 | 1.00 | 0.62 | 0.78 | 0.94 | 1.00 | 1.00 | 0.91 | – | 21.8 |
| 2026-10-24 00:39 | agent-11 | agent | qwen3:8b | 5 | 0.86 | 0.62 | 0.86 | 1.00 | 1.00 | 0.00 | 0.67 | 2 | 94.2 |
| 2026-10-24 07:01 | agent-11-4b | agent | qwen3:4b-instruct | 5 | 0.94 | 0.67 | 0.78 | 0.78 | 0.78 | 1.00 | 0.64 | 0 | 15.2 |
| 2026-10-24 07:56 | regression-iter08 | agent | qwen3:4b-instruct | 5 | 0.89 | 0.58 | 0.89 | 0.89 | 0.89 | 1.00 | 0.73 | 0 | 22.1 |
