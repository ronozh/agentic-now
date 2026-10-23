-- Least privilege for the agent: read-only, HR schema only, short statement timeout.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'hr_readonly') THEN
        CREATE ROLE hr_readonly LOGIN PASSWORD 'hr_readonly';
    END IF;
END $$;

REVOKE ALL ON SCHEMA public FROM hr_readonly;
GRANT USAGE ON SCHEMA hr TO hr_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA hr TO hr_readonly;
ALTER ROLE hr_readonly SET statement_timeout = '5s';
ALTER ROLE hr_readonly SET default_transaction_read_only = on;
ALTER ROLE hr_readonly SET search_path = hr;
