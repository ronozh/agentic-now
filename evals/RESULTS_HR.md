# HR eval results

Notes (iter 05, text-to-SQL): easy questions pass; every trap fails because the LLM can't know
the business definitions: headcount counts all 500 rows (ignores status history, contractors,
interns, exits), FTE ignores part-time weighting rules, "unpaid leave" uses a wrong status value
('on unpaid leave' vs the real code), year-end ignores planned exits and approved requisitions.

| date | label | model | value_correct | easy | trap | failures | avg s |
|---|---|---|---|---|---|---|---|
| 2026-10-24 07:07 | sql-4b | qwen3:4b-instruct | 0.38 | 1.00 | 0.00 | 0 | 19.5 |
