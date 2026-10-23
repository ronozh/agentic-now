"""Semantic layer tests: compiled metrics must equal the hand-written ground truth (no LLM).

Needs the seeded HR database: uv run python -m agentic_now.hr.seed
"""

from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from agentic_now import db
from agentic_now.semantic.layer import SemanticError, compile_metric, load

GOLDEN = {
    c["name"]: c["ground_truth_sql"]
    for c in yaml.safe_load(Path("evals/datasets/hr_golden.yaml").read_text())["cases"]
}
TODAY = date.today()
YEAR_END = date(TODAY.year, 12, 31)


@pytest.fixture(scope="module")
def conn():
    with db.connect() as connection:
        yield connection


def run(conn, metric, as_of=TODAY, group_by=None, filters=None):
    sql, params = compile_metric(load(), metric, as_of, group_by, filters, today=TODAY)
    return conn.execute(sql, params).fetchall()


def truth(conn, case):
    return float(conn.execute(GOLDEN[case]).fetchone()[0])


@pytest.mark.parametrize(
    ("case", "metric", "filters"),
    [
        ("headcount_today", "headcount", None),
        ("fte_today", "fte", None),
        ("unpaid_leave_today", "headcount_on_leave", {"status": "unpaid_leave"}),
        ("part_time_headcount", "headcount", {"employment_type": "part_time"}),
    ],
)
def test_metric_matches_ground_truth(conn, case, metric, filters):
    assert float(run(conn, metric, filters=filters)[0][0]) == truth(conn, case)


@pytest.mark.skipif(TODAY >= YEAR_END, reason="no future date left this year")
def test_projected_year_end_matches_ground_truth(conn):
    assert float(run(conn, "headcount_projected", YEAR_END)[0][0]) == truth(
        conn, "headcount_year_end"
    )


@pytest.mark.parametrize(
    "dimension", ["department", "location", "employment_type", "status"]
)
def test_group_by_sums_to_total(conn, dimension):
    rows = run(conn, "headcount", group_by=[dimension])
    assert len(rows) > 1
    assert sum(row[-1] for row in rows) == run(conn, "headcount")[0][0]


def test_point_in_time_excludes_people_who_left(conn):
    # Headcount a year ago differs from today: history is effective-dated, not "current flags".
    assert run(conn, "headcount", TODAY - timedelta(days=365)) != run(conn, "headcount")


def test_fte_weight_is_policy_not_code(conn):
    layer = load()
    changed = layer.model_copy(
        update={"fte_weights": {**layer.fte_weights, "part_time": 0.6}}
    )
    sql, params = compile_metric(changed, "fte", TODAY, today=TODAY)
    part_timers = run(conn, "fte", filters={"employment_type": "part_time"})[0][0] / 0.5
    assert float(conn.execute(sql, params).fetchone()[0]) == pytest.approx(
        truth(conn, "fte_today") + 0.1 * float(part_timers)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metric_name": "salary_total"}, "Unknown metric"),
        ({"group_by": ["salary; DROP TABLE x"]}, "Unknown dimension"),
        ({"filters": {"1=1 OR department": "x"}}, "Unknown dimension"),
        ({"as_of": YEAR_END + timedelta(days=1)}, "headcount_projected"),
        ({"metric_name": "headcount_projected", "as_of": TODAY}, "future dates"),
        (
            {
                "metric_name": "headcount_projected",
                "as_of": YEAR_END + timedelta(days=1),
                "group_by": ["status"],
            },
            "not available",
        ),
    ],
)
def test_rejects_invalid_requests(kwargs, message):
    args = {"metric_name": "headcount", "as_of": TODAY, **kwargs}
    with pytest.raises(SemanticError, match=message):
        compile_metric(load(), today=TODAY, **args)


def test_yaml_typos_fail_fast(tmp_path):
    bad = yaml.safe_load(Path("src/agentic_now/semantic/metrics.yaml").read_text())
    bad["metrics"]["headcount"]["statuses"].append("on_vacation")
    path = tmp_path / "metrics.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="on_vacation"):
        load(path)
