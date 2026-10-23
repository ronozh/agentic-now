"""HR MCP server, connecting as the read-only `hr_readonly` role.

Tools: schema discovery and read-only SQL (iteration 05).

Run (stdio):      uv run python -m agentic_now.mcp_servers.hr_server
Run (HTTP):       uv run python -m agentic_now.mcp_servers.hr_server --transport http --port 8002
Inspect:          uv run fastmcp dev inspector src/agentic_now/mcp_servers/hr_server.py
"""

import argparse
import logging
import re
from typing import Any

import psycopg
from fastmcp import FastMCP
from psycopg import sql
from psycopg.rows import dict_row

from agentic_now.config import settings
from agentic_now.log import setup_logging

logger = logging.getLogger(__name__)

MAX_ROWS = 200
SINGLE_SELECT = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

mcp = FastMCP(
    "hr",
    instructions="Read-only access to the company HR database (PostgreSQL schema `hr`).",
)


def _query(
    query: str | sql.Composed, params: tuple | dict = ()
) -> list[dict[str, Any]]:
    # The role itself is read-only with a 5s statement timeout (sql/hr_roles.sql); the checks
    # in run_readonly_sql are a friendlier first line of defense, not the security boundary.
    with psycopg.connect(settings.hr_database_url, row_factory=dict_row) as conn:
        return conn.execute(query, params).fetchall()


@mcp.tool
def list_tables() -> list[dict[str, Any]]:
    """List the HR tables with a one-line description of each."""
    return _query(
        """
        SELECT c.relname AS table, obj_description(c.oid) AS description
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'hr' AND c.relkind = 'r' ORDER BY c.relname
        """
    )


@mcp.tool
def describe_table(table: str) -> dict[str, Any]:
    """Columns (name, type, description) and 3 sample rows of one HR table.

    Args:
        table: Table name without schema, e.g. "employees".
    """
    # Resolve the name against the catalog (only real tables in schema hr), then quote it as
    # an identifier: user input never becomes SQL text.
    found = _query(
        """
        SELECT c.oid, c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'hr' AND c.relkind = 'r' AND c.relname = %s
        """,
        (table,),
    )
    if not found:
        return {"error": f"Unknown table {table!r}. Use list_tables()."}
    columns = _query(
        """
        SELECT attname AS column, format_type(atttypid, atttypmod) AS type,
               col_description(attrelid, attnum) AS description
        FROM pg_attribute WHERE attrelid = %s AND attnum > 0 AND NOT attisdropped
        ORDER BY attnum
        """,
        (found[0]["oid"],),
    )
    samples = _query(
        sql.SQL("SELECT * FROM {} ORDER BY 1 LIMIT 3").format(
            sql.Identifier("hr", found[0]["relname"])
        )
    )
    return {"table": f"hr.{table}", "columns": columns, "sample_rows": samples}


@mcp.tool
def run_readonly_sql(sql: str) -> dict[str, Any]:
    """Run one read-only SQL query (SELECT or WITH) against the HR database.

    Tables live in schema `hr`. Returns at most 200 rows. On error, returns the database
    error message so the query can be fixed and retried.

    Args:
        sql: A single SELECT/WITH statement.
    """
    statement = sql.strip().rstrip(";")
    if not SINGLE_SELECT.match(statement) or ";" in statement:
        return {"error": "Only a single SELECT or WITH statement is allowed."}
    try:
        # Limit inside the database: never pull a huge result into memory.
        rows = _query(f"SELECT * FROM ({statement}) q LIMIT {MAX_ROWS + 1}")
    except psycopg.Error as e:
        logger.info("SQL error: %s", e)
        return {"error": str(e).strip()}
    return {
        "row_count": len(rows),
        "truncated": len(rows) > MAX_ROWS,
        "rows": rows[:MAX_ROWS],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    setup_logging()
    if args.transport == "http":
        mcp.run(transport="http", host="127.0.0.1", port=args.port, show_banner=False)
    else:
        mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
