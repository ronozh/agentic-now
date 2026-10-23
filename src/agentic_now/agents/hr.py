"""Iteration 05: HR agent. Answers HR questions by writing SQL through the HR MCP server.

Run: uv run python -m agentic_now.agents.hr "How many departments are there?"
"""

import logging
import re
import sys
import time
import uuid
from datetime import date

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langfuse import get_client, observe, propagate_attributes
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, TextOutput, ToolOutput, UsageLimits
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ToolCallPart

from agentic_now.config import settings
from agentic_now.llm import chat_model
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability

logger = logging.getLogger(__name__)

MAX_REQUESTS = 12

SQL_INSTRUCTIONS = """\
You answer questions about the company's workforce using the HR database tools.
Follow these steps:
1. Call list_tables, then describe_table for the tables you need.
2. Call run_readonly_sql. Tables are in schema hr, e.g. hr.employees. On an error, fix the
   SQL and try again.
3. Call final_answer: the main number in `value`, the SQL you used in `sql_used`.
"""


class HRAnswer(BaseModel):
    answer: str = Field(description="Short answer in plain English.")
    value: float | None = Field(description="The main number in the answer, if any.")
    sql_used: str | None = Field(
        default=None, description="The final SQL query, if any."
    )


def hr_mcp_toolset() -> MCPToolset:
    """The HR MCP server as a subprocess over stdio (iteration 07 adds HTTP)."""
    transport = StdioTransport(
        command=sys.executable, args=["-m", "agentic_now.mcp_servers.hr_server"]
    )
    return MCPToolset(Client(transport))


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


agent = Agent(
    chat_model(),
    toolsets=[hr_mcp_toolset()],
    output_type=[ToolOutput(HRAnswer, name="final_answer"), TextOutput(text_answer)],
    instructions=SQL_INSTRUCTIONS,
    name="hr",
)


@agent.instructions
def today() -> str:
    return f"Today's date is {date.today().isoformat()}."


@observe(name="hr_agent")
def ask(question: str) -> HRAnswer:
    get_client().update_current_span(metadata={"model": settings.chat_model})
    started = time.perf_counter()
    result = agent.run_sync(
        question, usage_limits=UsageLimits(request_limit=MAX_REQUESTS)
    )
    tools = [
        part.tool_name
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    logger.info(
        "hr agent answered in %.1fs (requests=%d, tools=%s, sql=%r)",
        time.perf_counter() - started,
        result.usage.requests,
        tools,
        result.output.sql_used,
    )
    return result.output


def main() -> None:
    setup_logging()
    init_observability()
    with propagate_attributes(
        session_id=str(uuid.uuid4()), user_id="dev", tags=["hr-agent", "cli"]
    ):
        result = ask(sys.argv[1])
    flush()
    print(result.answer)
    print(f"  value: {result.value}")
    print(f"  sql:   {result.sql_used}")


if __name__ == "__main__":
    main()
