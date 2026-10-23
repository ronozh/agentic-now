# HR eval results

Notes (iter 05, text-to-SQL): easy questions pass; every trap fails because the LLM can't know
the business definitions: headcount counts all 500 rows (ignores status history, contractors,
interns, exits), FTE ignores part-time weighting rules, "unpaid leave" uses a wrong status value
('on unpaid leave' vs the real code), year-end ignores planned exits and approved requisitions.

Notes (iter 06, semantic layer): traps 0/5 → 5/5 and faster (fewer, simpler tool calls). The
LLM no longer needs to know the definitions: it only picks a metric, and the YAML encodes the
policy as tested, point-in-time SQL. Raw SQL stays as a fallback for questions no metric covers.

| date | label | model | value_correct | easy | trap | failures | avg s |
|---|---|---|---|---|---|---|---|
| 2026-10-24 07:07 | sql-4b | qwen3:4b-instruct | 0.38 | 1.00 | 0.00 | 0 | 19.5 |
| 2026-10-24 07:13 | semantic-4b | qwen3:4b-instruct | 1.00 | 1.00 | 1.00 | 0 | 10.3 |
| 2026-10-24 07:15 | sql-4b-v2 | qwen3:4b-instruct | 0.38 | 1.00 | 0.00 | 0 | 13.3 |
| 2026-10-24 07:57 | regression-iter08 | qwen3:4b-instruct | 0.88 | 0.67 | 1.00 | 0 | 12.4 |
