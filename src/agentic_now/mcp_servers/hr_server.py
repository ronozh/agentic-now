"""HR MCP server, connecting as the read-only `hr_readonly` role.

Tools: raw SQL (iteration 05) and governed metrics from the semantic layer (iteration 06).

Run (stdio):      uv run python -m agentic_now.mcp_servers.hr_server
Run (HTTP):       uv run python -m agentic_now.mcp_servers.hr_server --transport http --port 8002
Inspect:          uv run fastmcp dev inspector src/agentic_now/mcp_servers/hr_server.py
"""

import argparse
import logging
import re
from datetime import date
from typing import Any

import psycopg
from fastmcp import FastMCP
from psycopg import sql
from psycopg.rows import dict_row

from agentic_now.config import settings
from agentic_now.log import setup_logging
from agentic_now.semantic.layer import SemanticError, compile_metric, load

logger = logging.getLogger(__name__)

MAX_ROWS = 200
SINGLE_SELECT = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

mcp = FastMCP(
    "hr",
    instructions="Read-only access to the company HR database (PostgreSQL schema `hr`).",
)


def _query(
    query: str | sql.Composed, params: tuple | dict | None = None
) -> list[dict[str, Any]]:
    # The role itself is read-only with a 5s statement timeout (sql/hr_roles.sql); the checks
    # in run_readonly_sql are a friendlier first line of defense, not the security boundary.
    with psycopg.connect(settings.hr_database_url, row_factory=dict_row) as conn:
        # params=None: without parameters psycopg leaves `%` alone (e.g. LIKE 'Eng%').
        return conn.execute(query, params or None).fetchall()


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
    # Small models count the samples as if they were the table; say what they are.
    return {
        "table": f"hr.{table}",
        "columns": columns,
        "examples_only_3_of_many_rows": samples,
        "note": "To count or aggregate, call run_readonly_sql.",
    }


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
        # Newline before ")": a trailing `-- comment` must not swallow the wrapper.
        rows = _query(f"SELECT * FROM ({statement}\n) q LIMIT {MAX_ROWS + 1}")
    except psycopg.Error as e:
        logger.info("SQL error: %s", e)
        return {"error": str(e).strip()}
    return {
        "row_count": len(rows),
        "truncated": len(rows) > MAX_ROWS,
        "rows": rows[:MAX_ROWS],
    }


# --- Semantic layer tools (iteration 06): the agent picks a metric, never writes SQL ------


@mcp.tool
def list_metrics() -> dict[str, Any]:
    """List governed HR metrics with their definitions, synonyms and the allowed dimensions.

    Use these instead of writing SQL: they encode the official business definitions.
    """
    layer = load()
    return {
        "metrics": {
            name: {"definition": m.description, "synonyms": m.synonyms}
            for name, m in layer.metrics.items()
        },
        "dimensions": {name: d.description for name, d in layer.dimensions.items()},
    }


@mcp.tool
def query_metric(
    metric: str,
    as_of: str | None = None,
    group_by: list[str] | None = None,
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute a governed HR metric.

    Args:
        metric: Metric name from list_metrics, e.g. "headcount" or "fte".
        as_of: Date as YYYY-MM-DD. Defaults to today. Future dates need "headcount_projected".
        group_by: Dimensions to break the result down by, e.g. ["department"].
        filters: Dimension values to filter on, e.g. {"employment_type": "part_time"}.
    """
    try:
        as_of_date = date.fromisoformat(as_of) if as_of else date.today()
        layer = load()
        query, params = compile_metric(layer, metric, as_of_date, group_by, filters)
    except (SemanticError, ValueError) as e:
        return {"error": str(e)}
    try:
        rows = _query(query, params)
    except psycopg.Error as e:
        return {"error": str(e).strip()}
    return {
        "metric": metric,
        "as_of": as_of_date.isoformat(),
        "definition": layer.metric(metric).description,
        "rows": rows,
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
