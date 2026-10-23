"""Helpers shared by the eval scripts."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from langfuse import get_client
from pydantic_evals.reporting import EvaluationReport

from agentic_now.obs import init_observability

EVALS_DIR = Path(__file__).parent
RESULTS_DIR = EVALS_DIR / "results"


def case_results(case) -> dict[str, float]:
    """All scores and assertions of one report case as floats (assertions: 1.0 / 0.0)."""
    return {
        name: float(r.value) for name, r in {**case.scores, **case.assertions}.items()
    }


def summarize(report: EvaluationReport) -> dict[str, float]:
    """Mean of every metric over the cases that produced it, plus mean task duration.

    A case that crashed (`report.failures`) scores 0 on every metric, so failures can't make
    an agent look better than it is.
    """
    values: dict[str, list[float]] = defaultdict(list)
    for case in report.cases:
        for name, value in case_results(case).items():
            values[name].append(value)
    for name in values:
        values[name] += [0.0] * len(report.failures)
    summary = {name: mean(v) for name, v in values.items()}
    summary["avg_seconds"] = (
        mean(c.task_duration for c in report.cases) if report.cases else 0.0
    )
    summary["failures"] = float(len(report.failures))
    return summary


def push_scores(report: EvaluationReport) -> None:
    """Attach every metric to its case's Langfuse trace (`output.trace_id`)."""
    if init_observability() is None:  # tracing off: no traces to attach to
        return
    langfuse = get_client()
    for case in report.cases:
        for name, result in {**case.scores, **case.assertions}.items():
            langfuse.create_score(
                trace_id=case.output.trace_id,
                name=name,
                value=float(result.value),
                comment=getattr(result, "reason", None),
            )


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def append_row(md_path: Path, title: str, columns: list[str], cells: list[str]) -> None:
    """Append one row to a markdown results table, creating the table on first use."""
    if not md_path.exists():
        md_path.write_text(
            f"# {title}\n\n| {' | '.join(columns)} |\n|{'---|' * len(columns)}\n"
        )
    with md_path.open("a") as f:
        f.write(f"| {' | '.join(cells)} |\n")
