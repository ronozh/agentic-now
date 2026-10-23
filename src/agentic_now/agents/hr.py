"""HR agent over the HR MCP server, in two modes:

- sql (iteration 05): text-to-SQL; the LLM must guess business definitions.
- semantic (iteration 06): the LLM picks a governed metric; the semantic layer writes the SQL.

Run: uv run python -m agentic_now.agents.hr [--mode sql|semantic] "What is our headcount today?"
"""

import argparse
import asyncio
import logging
import re
import sys
import time
import uuid
from datetime import date
from typing import Literal

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langfuse import get_client, observe, propagate_attributes
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, TextOutput, ToolOutput, UsageLimits
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.toolsets import AbstractToolset

from agentic_now.config import settings
from agentic_now.llm import chat_model
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability

logger = logging.getLogger(__name__)

Mode = Literal["sql", "semantic"]
MAX_REQUESTS = 12

SQL_INSTRUCTIONS = """\
You answer questions about the company's workforce using the HR database tools.
Follow these steps:
1. Call list_tables, then describe_table for the tables you need.
2. Call run_readonly_sql. Tables are in schema hr, e.g. hr.employees. On an error, fix the
   SQL and try again.
3. Call final_answer: the main number in `value`, the SQL you used in `sql_used`.
"""

SEMANTIC_INSTRUCTIONS = """\
You answer questions about the company's workforce. Governed metrics come first.
Follow these steps:
1. Call list_metrics. If a metric matches the question (headcount, FTE, leave, projections),
   call query_metric. Use filters or group_by for subsets. as_of is a date YYYY-MM-DD
   (default today); for a future date such as year end, use headcount_projected. If it
   returns an error, read it and fix the call. Never compute such numbers with SQL.
2. Only if no metric matches (e.g. departments, hires, requisitions), call describe_table and
   run_readonly_sql. Tables are in schema hr, e.g. hr.employees.
3. Call final_answer with the number in `value` and any SQL you ran in `sql_used`. For a
   metric, briefly state its definition (e.g. "excludes contractors and interns"); for
   projections, say it assumes approved requisitions are hired.
"""

TOOLS = {
    "sql": {"list_tables", "describe_table", "run_readonly_sql"},
    # Metrics first, raw SQL as the fallback for questions no metric covers.
    "semantic": {"list_metrics", "query_metric", "describe_table", "run_readonly_sql"},
}


class HRAnswer(BaseModel):
    answer: str = Field(description="Short answer in plain English.")
    value: float | None = Field(description="The main number in the answer, if any.")
    sql_used: str | None = Field(
        default=None, description="The final SQL query, if any."
    )


def hr_mcp_toolset(tool_names: set[str]) -> AbstractToolset:
    """The HR MCP server, exposing only `tool_names` to the agent.

    stdio: a subprocess per agent (iteration 05). http: a shared service (iteration 07).
    """
    if settings.hr_mcp_transport == "http":
        toolset = MCPToolset(settings.hr_mcp_url)
    else:
        transport = StdioTransport(
            command=sys.executable, args=["-m", "agentic_now.mcp_servers.hr_server"]
        )
        toolset = MCPToolset(Client(transport))
    return toolset.filtered(lambda _ctx, tool: tool.name in tool_names)


def today() -> str:
    return f"Today's date is {date.today().isoformat()}."


def text_answer(ctx: RunContext, text: str) -> HRAnswer:
    """Small models often reply in plain text instead of calling `final_answer` and ignore
    the retry asking for a tool call. Accept the text: the value is its first number (dates
    removed), the SQL is the last query the agent ran."""
    numbers = re.findall(
        r"-?\d[\d,]*(?:\.\d+)?", re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
    )
    queries = [
        part.args_as_dict().get("sql")
        for message in ctx.messages
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == "run_readonly_sql"
    ]
    return HRAnswer(
        answer=text,
        value=float(numbers[0].replace(",", "")) if numbers else None,
        sql_used=queries[-1] if queries else None,
    )


def _build(mode: Mode) -> Agent[None, HRAnswer]:
    """A fresh agent per call: an MCP session is bound to one event loop, and callers
    (router tools, API requests) run on different threads with their own loops."""
    agent = Agent(
        chat_model(),
        toolsets=[hr_mcp_toolset(TOOLS[mode])],
        output_type=[
            ToolOutput(HRAnswer, name="final_answer"),
            TextOutput(text_answer),
        ],
        instructions=SQL_INSTRUCTIONS if mode == "sql" else SEMANTIC_INSTRUCTIONS,
        name=f"hr_{mode}",
    )
    agent.instructions(today)
    return agent


@observe(name="hr_agent")
async def ask_async(question: str, mode: Mode = "semantic") -> HRAnswer:
    """Async entry point: callers inside another agent's tool can't use `run_sync`."""
    get_client().update_current_span(
        metadata={"model": settings.chat_model, "mode": mode}
    )
    started = time.perf_counter()
    result = await _build(mode).run(
        question, usage_limits=UsageLimits(request_limit=MAX_REQUESTS)
    )
    tools = [
        part.tool_name
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    logger.info(
        "hr agent (%s) answered in %.1fs (requests=%d, tools=%s, sql=%r)",
        mode,
        time.perf_counter() - started,
        result.usage.requests,
        tools,
        result.output.sql_used,
    )
    return result.output


def ask(question: str, mode: Mode = "semantic") -> HRAnswer:
    return asyncio.run(ask_async(question, mode))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", choices=list(TOOLS), default="semantic")
    args = parser.parse_args()
    setup_logging()
    init_observability()
    with propagate_attributes(
        session_id=str(uuid.uuid4()), user_id="dev", tags=["hr-agent", args.mode, "cli"]
    ):
        result = ask(args.question, args.mode)
    flush()
    print(result.answer)
    print(f"  value: {result.value}")
    print(f"  sql:   {result.sql_used}")


if __name__ == "__main__":
    main()
