"""HR agent evaluation: is the number in the answer correct?

Run:
  uv run python -m evals.hr_eval                    # text-to-SQL agent (iteration 05)
  uv run python -m evals.hr_eval --label my-run
"""

import argparse
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from langfuse import get_client, propagate_attributes
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from agentic_now import db
from agentic_now.agents.hr import ask
from agentic_now.config import settings
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability
from evals.common import (
    EVALS_DIR,
    RESULTS_DIR,
    append_row,
    case_results,
    push_scores,
    summarize,
    timestamp,
)

logger = logging.getLogger(__name__)

DATASET = EVALS_DIR / "datasets" / "hr_golden.yaml"
RESULTS_MD = EVALS_DIR / "RESULTS_HR.md"


class HRExpected(BaseModel):
    kind: Literal["easy", "trap"]
    ground_truth_sql: str
    tolerance: float = 0.0


class HROutput(BaseModel):
    answer: str
    value: float | None
    sql_used: str | None
    trace_id: str | None = None


@dataclass
class ValueCorrect(Evaluator[str, HROutput, HRExpected]):
    """Compare the agent's number with the ground-truth SQL, computed now (dates move)."""

    def evaluate(
        self, ctx: EvaluatorContext[str, HROutput, HRExpected]
    ) -> dict[str, EvaluationReason]:
        with db.connect() as conn:
            truth = float(conn.execute(ctx.metadata.ground_truth_sql).fetchone()[0])
        got = ctx.output.value
        ok = got is not None and abs(got - truth) <= ctx.metadata.tolerance
        return {
            "value_correct": EvaluationReason(
                value=ok, reason=f"expected {truth:g}, got {got}"
            )
        }


def load_dataset(
    path: Path = DATASET, limit: int | None = None
) -> Dataset[str, HROutput, HRExpected]:
    cases = [
        Case(
            name=raw.pop("name"), inputs=raw.pop("question"), metadata=HRExpected(**raw)
        )
        for raw in yaml.safe_load(path.read_text())["cases"][:limit]
    ]
    return Dataset(name="hr_golden", cases=cases, evaluators=[ValueCorrect()])


def task(question: str) -> HROutput:
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        name="eval_case", input=question
    ) as span:
        result = ask(question)
        output = HROutput(
            **result.model_dump(), trace_id=langfuse.get_current_trace_id()
        )
        span.update(output=result.model_dump())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", default="sql")
    parser.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    args = parser.parse_args()

    setup_logging()
    init_observability()
    with propagate_attributes(
        session_id=f"eval-{uuid.uuid4().hex[:8]}", tags=["eval", "hr", args.label]
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
    by_kind = {
        kind: [
            case_results(c).get("value_correct", 0.0)
            for c in report.cases
            if c.metadata.kind == kind
        ]
        for kind in ("easy", "trap")
    }
    accuracy = {kind: sum(v) / len(v) if v else 0.0 for kind, v in by_kind.items()}
    push_scores(report)

    stamp = timestamp()
    RESULTS_DIR.mkdir(exist_ok=True)
    (
        RESULTS_DIR / f"{stamp.replace(' ', '_').replace(':', '')}-hr-{args.label}.json"
    ).write_text(
        json.dumps(
            {
                "label": args.label,
                "model": settings.chat_model,
                "summary": summary,
                "accuracy": accuracy,
                "cases": [
                    {
                        "name": c.name,
                        "output": c.output.model_dump(exclude={"trace_id"}),
                        "results": case_results(c),
                    }
                    for c in report.cases
                ],
                "failures": [f.name for f in report.failures],
            },
            indent=2,
            default=str,
        )
    )
    append_row(
        RESULTS_MD,
        "HR eval results",
        [
            "date",
            "label",
            "model",
            "value_correct",
            "easy",
            "trap",
            "failures",
            "avg s",
        ],
        [
            stamp,
            args.label,
            settings.chat_model,
            f"{summary.get('value_correct', 0.0):.2f}",
            f"{accuracy['easy']:.2f}",
            f"{accuracy['trap']:.2f}",
            str(len(report.failures)),
            f"{summary['avg_seconds']:.1f}",
        ],
    )
    flush()


if __name__ == "__main__":
    main()
