"""Iteration 03: RAG evaluation with Pydantic Evals.

Run:
  uv run python -m evals.rag_eval                          # retrieval + LLM answer + judge
  uv run python -m evals.rag_eval --target retrieval       # retrieval only: fast, deterministic
  uv run python -m evals.rag_eval --target agent           # agentic RAG (iteration 04)
  uv run python -m evals.rag_eval --k 10 --label k10       # an experiment, compared in RESULTS.md
  uv run python -m evals.rag_eval --limit 3                # smoke test on the first 3 cases
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
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from agentic_now import db
from agentic_now.agents.books import ask
from agentic_now.config import settings
from agentic_now.llm import chat_model
from agentic_now.log import setup_logging
from agentic_now.obs import flush, init_observability
from agentic_now.rag.answer import answer
from agentic_now.rag.retrieve import Source, retrieve
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

DATASET = EVALS_DIR / "datasets" / "rag_golden.yaml"
RESULTS_MD = EVALS_DIR / "RESULTS.md"
METRICS = [
    "hit@k",
    "mrr",
    "citation_correct",
    "keywords",
    "answered",
    "refusal_correct",
    "judge",
]

Target = Literal["retrieval", "pipeline", "agent"]


class Expected(BaseModel):
    """Case metadata: what a correct run looks like."""

    sources: list[
        dict[str, list[int]]
    ] = []  # fact groups: {book title: [printed pages]}
    keywords: list[str] = []
    out_of_scope: bool = False


class EvalOutput(BaseModel):
    answer: str | None  # None when only retrieval is evaluated
    citations: list[Source] = []
    retrieved: list[Source]
    trace_id: str | None = None


def matches(source: Source, group: dict[str, list[int]]) -> bool:
    return source.printed_page in group.get(source.title, [])


def is_refusal(text: str) -> bool:
    text = text.lower().replace("’", "'")
    return "don't know" in text or "do not know" in text


# --- Evaluators: deterministic first, LLM judge last -------------------------------------


@dataclass
class RetrievalMetrics(Evaluator[str, EvalOutput, Expected]):
    """hit@k: share of expected facts found in the retrieved chunks. MRR: 1/rank of first relevant chunk."""

    def evaluate(
        self, ctx: EvaluatorContext[str, EvalOutput, Expected]
    ) -> dict[str, float]:
        groups = ctx.metadata.sources
        if not groups:
            return {}
        retrieved = ctx.output.retrieved
        found = sum(any(matches(s, g) for s in retrieved) for g in groups)
        ranks = [
            i for i, s in enumerate(retrieved, 1) if any(matches(s, g) for g in groups)
        ]
        return {"hit@k": found / len(groups), "mrr": 1 / ranks[0] if ranks else 0.0}


@dataclass
class AnswerChecks(Evaluator[str, EvalOutput, Expected]):
    """Cheap string checks on the answer and its citations."""

    def evaluate(
        self, ctx: EvaluatorContext[str, EvalOutput, Expected]
    ) -> dict[str, float | bool]:
        out, expected = ctx.output, ctx.metadata
        if out.answer is None:
            return {}
        if expected.out_of_scope:
            return {"refusal_correct": is_refusal(out.answer) and not out.citations}
        text = out.answer.lower()
        checks: dict[str, float | bool] = {"answered": not is_refusal(out.answer)}
        if expected.keywords:
            checks["keywords"] = sum(
                k.lower() in text for k in expected.keywords
            ) / len(expected.keywords)
        if expected.sources:
            checks["citation_correct"] = all(
                any(matches(c, g) for c in out.citations) for g in expected.sources
            )
        return checks


class Grade(BaseModel):
    reason: str = Field(description="One sentence explaining the grade.")
    passed: bool


JUDGE_INSTRUCTIONS = """\
You grade answers from an assistant that answers questions using management textbooks.
- Pass if the answer is correct, specific and addresses the whole question.
- The expected key terms are a hint about what a good answer covers, not a strict checklist.
- If the question is marked out of scope, ONLY a refusal such as "I don't know" passes.
"""

judge_agent = Agent(
    chat_model(),
    output_type=NativeOutput(Grade),
    instructions=JUDGE_INSTRUCTIONS,
    model_settings={"temperature": 0},
    name="judge",
)


@dataclass
class AnswerJudge(Evaluator[str, EvalOutput, Expected]):
    """LLM-as-judge with the local model. Useful but noisy: trust deterministic metrics more."""

    async def evaluate(
        self, ctx: EvaluatorContext[str, EvalOutput, Expected]
    ) -> dict[str, EvaluationReason]:
        if ctx.output.answer is None:
            return {}
        expected = (
            "out of scope: must refuse"
            if ctx.metadata.out_of_scope
            else f"key terms: {', '.join(ctx.metadata.keywords)}"
        )
        prompt = (
            f"Question: {ctx.inputs}\nExpected: {expected}\nAnswer: {ctx.output.answer}"
        )
        grade = (await judge_agent.run(prompt)).output
        return {"judge": EvaluationReason(value=grade.passed, reason=grade.reason)}


# --- Dataset, task, reporting ------------------------------------------------------------


def load_dataset(
    path: Path = DATASET, limit: int | None = None
) -> Dataset[str, EvalOutput, Expected]:
    cases = [
        Case(
            name=raw.pop("name"),
            inputs=raw.pop("question"),
            metadata=Expected(**raw),
        )
        for raw in yaml.safe_load(path.read_text())["cases"][:limit]
    ]
    return Dataset(
        name="rag_golden",
        cases=cases,
        evaluators=[RetrievalMetrics(), AnswerChecks(), AnswerJudge()],
    )


def make_task(target: Target, k: int):
    def task(question: str) -> EvalOutput:
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            name="eval_case", input=question
        ) as span:
            trace_id = langfuse.get_current_trace_id()
            if target == "retrieval":
                with db.connect() as conn:
                    hits = retrieve(conn, question, k)
                output = EvalOutput(
                    answer=None, retrieved=[h.source for h in hits], trace_id=trace_id
                )
            else:
                result = (
                    answer(question, k) if target == "pipeline" else ask(question, k)
                )
                output = EvalOutput(**result.model_dump(), trace_id=trace_id)
            span.update(output=output.model_dump(exclude={"trace_id"}))
        return output

    return task


def record(
    summary: dict[str, float], report: EvaluationReport, args: argparse.Namespace
) -> None:
    stamp = timestamp()
    RESULTS_DIR.mkdir(exist_ok=True)
    result_file = (
        RESULTS_DIR / f"{stamp.replace(' ', '_').replace(':', '')}-{args.label}.json"
    )
    result_file.write_text(
        json.dumps(
            {
                "label": args.label,
                "target": args.target,
                "model": settings.chat_model,
                "k": args.k,
                "summary": summary,
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
    label = args.label if args.limit is None else f"{args.label} (first {args.limit})"
    append_row(
        RESULTS_MD,
        "Eval results",
        ["date", "label", "target", "model", "k", *METRICS, "failures", "avg s"],
        [
            stamp,
            label,
            args.target,
            settings.chat_model,
            str(args.k),
            *(f"{summary[m]:.2f}" if m in summary else "–" for m in METRICS),
            f"{summary['failures']:.0f}",
            f"{summary['avg_seconds']:.1f}",
        ],
    )
    logger.info("saved %s and appended %s", result_file.name, RESULTS_MD.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target", choices=["retrieval", "pipeline", "agent"], default="pipeline"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    args = parser.parse_args()

    setup_logging()
    init_observability()
    dataset = load_dataset(limit=args.limit)
    with propagate_attributes(
        session_id=f"eval-{uuid.uuid4().hex[:8]}",
        tags=["eval", args.target, args.label],
    ):
        report = dataset.evaluate_sync(
            make_task(args.target, args.k), max_concurrency=1, progress=False
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
    record(summary, report, args)
    flush()


if __name__ == "__main__":
    main()
