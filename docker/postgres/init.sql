-- Um único servidor PostgreSQL, três bancos isolados por responsabilidade:
--   warehouse -> dados analíticos (raw / staging / marts / ml)
--   airflow   -> metadados do orquestrador
--   mlflow    -> backend store do tracking server e do model registry
CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow OWNER airflow;

CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;

\connect warehouse
CREATE SCHEMA IF NOT EXISTS raw AUTHORIZATION analytics;
CREATE SCHEMA IF NOT EXISTS staging AUTHORIZATION analytics;
CREATE SCHEMA IF NOT EXISTS marts AUTHORIZATION analytics;
CREATE SCHEMA IF NOT EXISTS ml AUTHORIZATION analytics;

-- Usuário somente leitura para ferramentas de BI (Power BI, Streamlit): princípio do menor privilégio
CREATE USER bi_reader WITH PASSWORD 'bi_reader';
GRANT CONNECT ON DATABASE warehouse TO bi_reader;
GRANT USAGE ON SCHEMA marts, ml TO bi_reader;
ALTER DEFAULT PRIVILEGES FOR USER analytics IN SCHEMA marts GRANT SELECT ON TABLES TO bi_reader;
ALTER DEFAULT PRIVILEGES FOR USER analytics IN SCHEMA ml GRANT SELECT ON TABLES TO bi_reader;
