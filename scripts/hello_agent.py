"""Iteration 00: smallest possible agent — one local LLM, one tool.

Run: uv run python scripts/hello_agent.py ["question"]
"""

import sys
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from agentic_now.config import settings
from agentic_now.llm import chat_model

agent = Agent(
    chat_model(),
    instructions="You are a helpful assistant. Use tools for facts you cannot know, such as today's date.",
)


@agent.tool_plain
def get_current_date() -> str:
    """Return today's date in ISO format (YYYY-MM-DD)."""
    return date.today().isoformat()


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What day of the week is it today?"
    result = agent.run_sync(question)

    print(f"model:    {settings.chat_model}")
    print(f"question: {question}")
    print(f"answer:   {result.output}\n")
    print("tool activity:")
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print(f"  -> ToolCallPart   {part.tool_name}({part.args or ''})")
            elif isinstance(part, ToolReturnPart):
                print(f"  <- ToolReturnPart {part.tool_name} = {part.content}")


if __name__ == "__main__":
    main()
