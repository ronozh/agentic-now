-- HR data model (iteration 05). Point-in-time questions use effective-dated status rows.
CREATE EXTENSION IF NOT EXISTS btree_gist;  -- lets an EXCLUDE constraint mix = and && operators
CREATE SCHEMA IF NOT EXISTS hr;

DROP TABLE IF EXISTS hr.requisitions, hr.employment_status_history, hr.employees, hr.departments CASCADE;

CREATE TABLE hr.departments (
    id       serial PRIMARY KEY,
    name     text UNIQUE NOT NULL,
    location text NOT NULL
);
COMMENT ON TABLE hr.departments IS 'Company departments.';

CREATE TABLE hr.employees (
    id               serial PRIMARY KEY,
    first_name       text NOT NULL,
    last_name        text NOT NULL,
    email            text UNIQUE NOT NULL,
    department_id    int NOT NULL REFERENCES hr.departments,
    manager_id       int REFERENCES hr.employees,
    job_title        text NOT NULL,
    employment_type  text NOT NULL CHECK (employment_type IN ('full_time', 'part_time', 'contractor', 'intern')),
    fte_ratio        numeric(3, 2) NOT NULL CHECK (fte_ratio > 0 AND fte_ratio <= 1),
    salary           numeric(10, 2) NOT NULL,
    hire_date        date NOT NULL,
    termination_date date CHECK (termination_date IS NULL OR termination_date > hire_date)
);
COMMENT ON TABLE hr.employees IS 'Everyone who ever worked here, including contractors, interns and former employees.';
COMMENT ON COLUMN hr.employees.employment_type IS 'full_time | part_time | contractor | intern';
COMMENT ON COLUMN hr.employees.fte_ratio IS 'Contracted share of a full-time schedule: 1.0 full-time, 0.5 part-time.';
COMMENT ON COLUMN hr.employees.termination_date IS 'Last day + 1. NULL = no exit planned. May be in the future (planned exit).';

CREATE TABLE hr.employment_status_history (
    id          serial PRIMARY KEY,
    employee_id int NOT NULL REFERENCES hr.employees,
    status      text NOT NULL CHECK (status IN ('active', 'paid_leave', 'unpaid_leave', 'terminated')),
    valid_from  date NOT NULL,
    valid_to    date CHECK (valid_to IS NULL OR valid_to > valid_from),
    -- The database itself guarantees one status per employee per day.
    EXCLUDE USING gist (employee_id WITH =, daterange(valid_from, valid_to) WITH &&)
);
COMMENT ON TABLE hr.employment_status_history IS
    'Effective-dated status per employee. A row applies when valid_from <= date < valid_to (valid_to NULL = still current).';

CREATE TABLE hr.requisitions (
    id                serial PRIMARY KEY,
    department_id     int NOT NULL REFERENCES hr.departments,
    job_title         text NOT NULL,
    employment_type   text NOT NULL CHECK (employment_type IN ('full_time', 'part_time')),
    status            text NOT NULL CHECK (status IN ('open', 'approved', 'filled', 'cancelled')),
    target_start_date date NOT NULL
);
COMMENT ON TABLE hr.requisitions IS 'Hiring requests. approved = budgeted hire expected to start on target_start_date.';
