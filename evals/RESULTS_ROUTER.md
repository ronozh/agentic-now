# Router eval results

Notes (iter 07): same accuracy local vs A2A/HTTP. Remote is faster here (31 vs 39 s/case)
because stdio MCP spawns a server subprocess per HR call, while HTTP reuses one running server.
`specialists_ok` was added after a run where routing scored 1.00 although every A2A call failed.

| date | label | books | hr mcp | model | routing | specialists_ok | hr_value | failures | avg s |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 07:31 | a2a-http-4b | a2a | http | qwen3:4b-instruct | 1.00 | 1.00 | 1.00 | 0 | 30.8 |
| 2026-10-24 07:36 | local-4b | local | stdio | qwen3:4b-instruct | 1.00 | 1.00 | 1.00 | 0 | 40.0 |
| 2026-10-24 07:43 | local-4b-v2 | local | stdio | qwen3:4b-instruct | 1.00 | 1.00 | 1.00 | 0 | 39.0 |
