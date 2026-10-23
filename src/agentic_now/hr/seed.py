"""Iteration 05: deterministic synthetic HR data with deliberate edge cases.

Edge cases that make "headcount" hard: part-time, contractors, interns, paid and unpaid
leave, past exits, planned (future) exits, and approved requisitions starting later.

Run: uv run python -m agentic_now.hr.seed
"""

import logging
import random
from datetime import date, timedelta

from agentic_now import db
from agentic_now.db import SQL_DIR
from agentic_now.log import setup_logging

logger = logging.getLogger(__name__)

SEED = 42
SNAPSHOT = date(
    2026, 10, 1
)  # data reflects this date; planned exits/hires come after it
FIRST_HIRE = date(2023, 1, 1)
YEAR_END = date(2026, 12, 31)
N_EMPLOYEES = 500

DEPARTMENTS = {  # name: (location, base salary, titles)
    "Engineering": (
        "San Francisco",
        150_000,
        ["Software Engineer", "Senior Engineer", "QA Engineer"],
    ),
    "Product": ("San Francisco", 140_000, ["Product Manager", "Designer"]),
    "Sales": ("New York", 95_000, ["Account Executive", "Sales Development Rep"]),
    "Marketing": ("New York", 100_000, ["Marketing Manager", "Content Strategist"]),
    "Customer Support": ("Austin", 60_000, ["Support Specialist", "Support Lead"]),
    "Operations": ("Austin", 80_000, ["Operations Analyst", "Office Manager"]),
    "Finance": ("Chicago", 110_000, ["Financial Analyst", "Accountant"]),
    "HR": ("Chicago", 90_000, ["HR Business Partner", "Recruiter"]),
}
EMPLOYMENT_TYPES = {
    "full_time": 0.72,
    "part_time": 0.15,
    "contractor": 0.10,
    "intern": 0.03,
}
FIRST_NAMES = [
    "Ava",
    "Ben",
    "Chloe",
    "Dan",
    "Ella",
    "Finn",
    "Grace",
    "Hugo",
    "Ivy",
    "Jack",
    "Kara",
    "Liam",
    "Maya",
    "Noah",
    "Olive",
    "Paul",
    "Quinn",
    "Rosa",
    "Sam",
    "Tara",
    "Uma",
    "Vik",
    "Wren",
    "Xavi",
    "Yara",
    "Zane",
    "Aria",
    "Leo",
    "Mila",
    "Omar",
    "Nina",
    "Raj",
    "Sara",
    "Tom",
    "Lena",
    "Kai",
    "Iris",
    "Eli",
]
LAST_NAMES = [
    "Adams",
    "Brown",
    "Chen",
    "Diaz",
    "Evans",
    "Ford",
    "Garcia",
    "Hill",
    "Ito",
    "Jones",
    "Khan",
    "Lee",
    "Moore",
    "Nguyen",
    "Ortiz",
    "Patel",
    "Quinn",
    "Reed",
    "Smith",
    "Tan",
    "Ueda",
    "Vargas",
    "Wong",
    "Young",
    "Zhang",
    "Silva",
    "Kim",
    "Novak",
    "Rossi",
    "Dubois",
]


def _day(rng: random.Random, start: date, end: date) -> date:
    """Random date in [start, end)."""
    return start + timedelta(days=rng.randrange(max((end - start).days, 1)))


def _status_history(
    rng: random.Random, hire: date, termination: date | None
) -> list[tuple]:
    """Contiguous (status, valid_from, valid_to) segments from hire date onwards."""
    segments: list[list] = [["active", hire, None]]
    leave_draw = rng.random()
    earliest_leave = hire + timedelta(days=60)
    still_here = termination is None or termination > SNAPSHOT
    if earliest_leave < SNAPSHOT - timedelta(days=30):
        if still_here and leave_draw < 0.08:  # currently on leave (open-ended)
            status = "paid_leave" if leave_draw < 0.05 else "unpaid_leave"
            start = _day(
                rng, max(earliest_leave, SNAPSHOT - timedelta(days=120)), SNAPSHOT
            )
            segments[-1][2] = start
            segments.append([status, start, None])
        elif leave_draw < 0.18:  # a past leave that ended
            start = _day(rng, earliest_leave, SNAPSHOT - timedelta(days=30))
            end = min(
                start + timedelta(days=rng.randint(14, 90)),
                SNAPSHOT - timedelta(days=1),
            )
            if termination is None or end < termination:
                segments[-1][2] = start
                segments += [["paid_leave", start, end], ["active", end, None]]
    if termination:
        while segments[-1][1] >= termination:  # drop segments starting after the exit
            segments.pop()
        segments[-1][2] = termination
        segments.append(["terminated", termination, None])
    return [tuple(s) for s in segments]


def build(rng: random.Random) -> tuple[list[tuple], list[tuple], list[tuple]]:
    employees, history = [], []
    dept_names = list(DEPARTMENTS)
    for emp_id in range(1, N_EMPLOYEES + 1):
        dept = rng.choice(dept_names)
        _, base_salary, titles = DEPARTMENTS[dept]
        employment_type = rng.choices(
            list(EMPLOYMENT_TYPES), weights=EMPLOYMENT_TYPES.values()
        )[0]
        fte = 0.5 if employment_type == "part_time" else 1.0
        hire = _day(rng, FIRST_HIRE, SNAPSHOT)
        exit_draw = rng.random()
        termination = None
        if exit_draw < 0.15 and hire + timedelta(days=45) < SNAPSHOT:  # already left
            termination = _day(rng, hire + timedelta(days=45), SNAPSHOT)
        elif exit_draw < 0.19:  # planned exit before year end
            termination = _day(rng, SNAPSHOT + timedelta(days=10), YEAR_END)
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        employees.append(
            (
                emp_id,
                first,
                last,
                f"{first}.{last}{emp_id}@example.com".lower(),
                dept_names.index(dept) + 1,
                rng.choice(titles),
                employment_type,
                fte,
                round(base_salary * rng.uniform(0.8, 1.4) * fte, -2),
                hire,
                termination,
            )
        )
        history += [
            (emp_id, *segment) for segment in _status_history(rng, hire, termination)
        ]

    requisitions = []
    for req_id, status in enumerate(
        ["approved"] * 18 + ["open"] * 6 + ["filled"] * 4 + ["cancelled"] * 2, start=1
    ):
        dept = rng.choice(dept_names)
        start = (
            _day(rng, SNAPSHOT, date(2027, 3, 31))
            if status in ("approved", "open")
            else _day(rng, date(2026, 1, 1), SNAPSHOT)
        )
        requisitions.append(
            (
                req_id,
                dept_names.index(dept) + 1,
                rng.choice(DEPARTMENTS[dept][2]),
                "full_time",
                status,
                start,
            )
        )
    return employees, history, requisitions


def seed() -> None:
    employees, history, requisitions = build(random.Random(SEED))
    with db.connect() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute((SQL_DIR / "hr_schema.sql").read_text())
        cur.executemany(
            "INSERT INTO hr.departments (id, name, location) VALUES (%s, %s, %s)",
            [
                (i, name, loc)
                for i, (name, (loc, _, _)) in enumerate(DEPARTMENTS.items(), start=1)
            ],
        )
        cur.executemany(
            "INSERT INTO hr.employees (id, first_name, last_name, email, department_id, job_title,"
            " employment_type, fte_ratio, salary, hire_date, termination_date)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            employees,
        )
        # Each department's earliest full-time hire manages everyone else in it.
        cur.execute(
            """
            UPDATE hr.employees e SET manager_id = head.id
            FROM (SELECT DISTINCT ON (department_id) id, department_id FROM hr.employees
                  WHERE employment_type = 'full_time' ORDER BY department_id, hire_date, id) head
            WHERE e.department_id = head.department_id AND e.id <> head.id
            """
        )
        cur.executemany(
            "INSERT INTO hr.employment_status_history (employee_id, status, valid_from, valid_to)"
            " VALUES (%s, %s, %s, %s)",
            history,
        )
        cur.executemany(
            "INSERT INTO hr.requisitions (id, department_id, job_title, employment_type, status,"
            " target_start_date) VALUES (%s, %s, %s, %s, %s, %s)",
            requisitions,
        )
        for table in ("departments", "employees", "requisitions"):
            cur.execute(
                f"SELECT setval('hr.{table}_id_seq', (SELECT max(id) FROM hr.{table}))"
            )
        cur.execute((SQL_DIR / "hr_roles.sql").read_text())
    logger.info(
        "seeded %d employees, %d status rows, %d requisitions",
        len(employees),
        len(history),
        len(requisitions),
    )


def main() -> None:
    setup_logging()
    seed()


if __name__ == "__main__":
    main()
