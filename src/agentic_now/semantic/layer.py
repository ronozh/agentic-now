"""Iteration 06: semantic layer. The LLM picks metric + dimensions + filters + date;
this module turns that into SQL deterministically from the definitions in metrics.yaml.
"""

from datetime import date
from functools import cache
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

METRICS_FILE = Path(__file__).with_name("metrics.yaml")

EmploymentType = Literal["full_time", "part_time", "contractor", "intern"]
Status = Literal["active", "paid_leave", "unpaid_leave"]


class SemanticError(ValueError):
    """A request the semantic layer cannot answer; the message tells the caller how to fix it."""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)  # typos in YAML fail fast


class Dimension(Strict):
    description: str
    sql: str
    requisition_sql: str | None = None  # None: not available for projected metrics


class Metric(Strict):
    description: str
    statuses: list[Status]
    measure: Literal["count", "fte"]
    projected: bool = False
    synonyms: list[str] = []


class SemanticLayer(Strict):
    employee_types: list[EmploymentType]
    fte_weights: dict[EmploymentType, float]
    dimensions: dict[str, Dimension]
    metrics: dict[str, Metric]

    @model_validator(mode="after")
    def _weights_cover_employee_types(self) -> Self:
        missing = set(self.employee_types) - set(self.fte_weights)
        if missing:
            raise ValueError(f"fte_weights missing for {sorted(missing)}")
        return self

    def metric(self, name: str) -> Metric:
        if name not in self.metrics:
            raise SemanticError(
                f"Unknown metric {name!r}. Available: {sorted(self.metrics)}"
            )
        return self.metrics[name]

    def check_dimensions(self, names: list[str], projected: bool) -> None:
        for name in names:
            if name not in self.dimensions:
                raise SemanticError(
                    f"Unknown dimension {name!r}. Available: {sorted(self.dimensions)}"
                )
            if projected and self.dimensions[name].requisition_sql is None:
                raise SemanticError(
                    f"Dimension {name!r} is not available for projected metrics."
                )


@cache
def load(path: Path = METRICS_FILE) -> SemanticLayer:
    return SemanticLayer.model_validate(yaml.safe_load(path.read_text()))


def compile_metric(
    layer: SemanticLayer,
    metric_name: str,
    as_of: date,
    group_by: list[str] | None = None,
    filters: dict[str, str] | None = None,
    today: date | None = None,
) -> tuple[str, dict[str, Any]]:
    """SQL + params for one metric. Identifiers come only from the YAML; values are params."""
    metric = layer.metric(metric_name)
    group_by, filters, today = group_by or [], filters or {}, today or date.today()
    layer.check_dimensions([*group_by, *filters], metric.projected)
    if metric.projected and as_of <= today:
        raise SemanticError(
            f"{metric_name} is for future dates; use 'headcount' for {as_of}."
        )
    if not metric.projected and as_of > today:
        raise SemanticError(
            f"{as_of} is in the future. Use 'headcount_projected' (assumes approved requisitions "
            "are hired) or ask the user whether they mean today's actual headcount."
        )

    params: dict[str, Any] = {
        "as_of": as_of,
        "today": today,
        "employee_types": list(layer.employee_types),
        "statuses": list(metric.statuses),
    }
    filter_sql, req_filter_sql = [], []
    for i, (name, value) in enumerate(filters.items()):
        params[f"f{i}"] = value
        filter_sql.append(f"AND {layer.dimensions[name].sql} = %(f{i})s")
        if metric.projected:
            req_filter_sql.append(
                f"AND {layer.dimensions[name].requisition_sql} = %(f{i})s"
            )

    if metric.measure == "fte":
        cases = []
        for i, (employment_type, weight) in enumerate(layer.fte_weights.items()):
            params[f"type{i}"], params[f"weight{i}"] = employment_type, weight
            cases.append(f"WHEN %(type{i})s THEN %(weight{i})s")
        # numeric avoids float noise (323.60000000000036); float8 keeps it a JSON number.
        weighted = f"sum(CASE e.employment_type {' '.join(cases)} END)"
        measure = f"round({weighted}::numeric, 2)::float8"
    else:
        measure = "count(DISTINCT e.id)"

    dims = [f"{layer.dimensions[g].sql} AS {g}" for g in group_by]
    # GROUP BY position, not alias: an alias like "status" would bind to the h.status column.
    group_sql = (
        "GROUP BY " + ", ".join(str(i) for i in range(1, len(group_by) + 1))
        if group_by
        else ""
    )
    employees_sql = f"""
        SELECT {", ".join([*dims, f"{measure} AS value"])}
        FROM hr.employees e
        JOIN hr.departments d ON d.id = e.department_id
        JOIN hr.employment_status_history h ON h.employee_id = e.id
         AND h.valid_from <= %(as_of)s AND (h.valid_to IS NULL OR h.valid_to > %(as_of)s)
        WHERE e.employment_type = ANY(%(employee_types)s) AND h.status = ANY(%(statuses)s)
        {" ".join(filter_sql)}
        {group_sql}"""
    if not metric.projected:
        return employees_sql, params

    req_dims = [f"{layer.dimensions[g].requisition_sql} AS {g}" for g in group_by]
    requisitions_sql = f"""
        SELECT {", ".join([*req_dims, "count(*) AS value"])}
        FROM hr.requisitions r JOIN hr.departments d ON d.id = r.department_id
        WHERE r.status = 'approved' AND r.employment_type = ANY(%(employee_types)s)
          AND r.target_start_date > %(today)s AND r.target_start_date <= %(as_of)s
        {" ".join(req_filter_sql)}
        {group_sql}"""
    keys = ", ".join(group_by)
    sql = f"""
        SELECT {keys + ", " if keys else ""}sum(value)::bigint AS value  -- projected metrics are counts
        FROM ({employees_sql} UNION ALL {requisitions_sql}) parts
        {group_sql}"""
    return sql, params
