"""Router evaluation: right specialists called, and HR numbers correct.

Run:
  uv run python -m evals.router_eval
  BOOKS_AGENT_MODE=a2a HR_MCP_TRANSPORT=http uv run python -m evals.router_eval --label remote
"""

import argparse
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml
from langfuse import get_client, propagate_attributes
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from agentic_now import db
from agentic_now.agents import router
from agentic_now.config import settings
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability
from evals.common import EVALS_DIR, append_row, push_scores, summarize, timestamp

logger = logging.getLogger(__name__)

DATASET = EVALS_DIR / "datasets" / "router_golden.yaml"
RESULTS_MD = EVALS_DIR / "RESULTS_ROUTER.md"


class RouterExpected(BaseModel):
    agents: list[str]
    hr_truth_sql: str | None = None


class RouterOutput(router.RouterAnswer):
    trace_id: str | None = None


@dataclass
class RoutingCorrect(Evaluator[str, RouterOutput, RouterExpected]):
    def evaluate(
        self, ctx: EvaluatorContext[str, RouterOutput, RouterExpected]
    ) -> EvaluationReason:
        expected, got = sorted(ctx.metadata.agents), ctx.output.agents_used
        return EvaluationReason(
            value=got == expected, reason=f"expected {expected}, got {got}"
        )


@dataclass
class SpecialistsOk(Evaluator[str, RouterOutput, RouterExpected]):
    """Routing alone can't see a specialist that was called but failed."""

    def evaluate(
        self, ctx: EvaluatorContext[str, RouterOutput, RouterExpected]
    ) -> EvaluationReason:
        return EvaluationReason(
            value=not ctx.output.errors, reason=", ".join(ctx.output.errors) or "ok"
        )


@dataclass
class HRValueCorrect(Evaluator[str, RouterOutput, RouterExpected]):
    """The HR specialist returned the true number AND the router's final answer quotes it
    (the router could drop or garble it when merging answers)."""

    def evaluate(
        self, ctx: EvaluatorContext[str, RouterOutput, RouterExpected]
    ) -> dict[str, EvaluationReason]:
        if not ctx.metadata.hr_truth_sql:
            return {}
        with db.connect() as conn:
            truth = float(conn.execute(ctx.metadata.hr_truth_sql).fetchone()[0])
        values = [a.value for a in ctx.output.hr_answers]
        specialist_ok = any(v is not None and abs(v - truth) < 1e-6 for v in values)
        quoted = f"{truth:g}" in ctx.output.answer.replace(",", "")
        return {
            "hr_value_correct": EvaluationReason(
                value=specialist_ok and quoted,
                reason=f"expected {truth:g}, specialist {values}, quoted in answer: {quoted}",
            )
        }


def load_dataset(
    path: Path = DATASET, limit: int | None = None
) -> Dataset[str, RouterOutput, RouterExpected]:
    cases = [
        Case(
            name=raw.pop("name"),
            inputs=raw.pop("question"),
            metadata=RouterExpected(**raw),
        )
        for raw in yaml.safe_load(path.read_text())["cases"][:limit]
    ]
    return Dataset(
        name="router_golden",
        cases=cases,
        evaluators=[RoutingCorrect(), SpecialistsOk(), HRValueCorrect()],
    )


def task(question: str) -> RouterOutput:
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        name="eval_case", input=question
    ) as span:
        answer, _ = router.ask(question)
        span.update(output=answer.answer)
        return RouterOutput(
            **answer.model_dump(), trace_id=langfuse.get_current_trace_id()
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", default="local")
    parser.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    args = parser.parse_args()

    setup_logging()
    init_observability()
    with propagate_attributes(
        session_id=f"eval-{uuid.uuid4().hex[:8]}", tags=["eval", "router", args.label]
    ):
        report = load_dataset(limit=args.limit).evaluate_sync(
            task, max_concurrency=1, progress=False
        )
    report.print(include_reasons=True)
    if report.failures:
        logger.error(
            "%d cases failed: %s",
            len(report.failures),
            [f.name for f in report.failures],
        )

    summary = summarize(report)
    push_scores(report)
    append_row(
        RESULTS_MD,
        "Router eval results",
        [
            "date",
            "label",
            "books",
            "hr mcp",
            "model",
            "routing",
            "specialists_ok",
            "hr_value",
            "failures",
            "avg s",
        ],
        [
            timestamp(),
            args.label,
            settings.books_agent_mode,
            settings.hr_mcp_transport,
            settings.chat_model,
            f"{summary.get('RoutingCorrect', 0.0):.2f}",
            f"{summary.get('SpecialistsOk', 0.0):.2f}",
            f"{summary.get('hr_value_correct', 0.0):.2f}",
            str(len(report.failures)),
            f"{summary['avg_seconds']:.1f}",
        ],
    )
    flush()


if __name__ == "__main__":
    main()
