-- ──────────────────────────────────────────────
-- SentinelAI — PostgreSQL Schema
-- ──────────────────────────────────────────────
-- Executed automatically by PostgreSQL on first container startup
-- via docker-entrypoint-initdb.d mount.

-- Scored transactions table
CREATE TABLE IF NOT EXISTS scored_transactions (
    id              BIGSERIAL PRIMARY KEY,
    transaction_id  BIGINT NOT NULL,
    amount          DOUBLE PRECISION,
    anomaly_score   DOUBLE PRECISION NOT NULL,
    fraud_probability DOUBLE PRECISION NOT NULL,
    decision        VARCHAR(16) NOT NULL,
    expected_cost   DOUBLE PRECISION NOT NULL,
    is_fraud        BOOLEAN,                          -- ground truth label (for FNR measurement)
    worker_id       VARCHAR(64),
    scored_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    ingested_at     TIMESTAMP WITH TIME ZONE,         -- when the producer sent it
    CONSTRAINT uq_transaction_id UNIQUE (transaction_id)
);

-- Indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_scored_decision ON scored_transactions (decision);
CREATE INDEX IF NOT EXISTS idx_scored_at ON scored_transactions (scored_at);
CREATE INDEX IF NOT EXISTS idx_scored_fraud_prob ON scored_transactions (fraud_probability);

-- Metrics summary view
CREATE OR REPLACE VIEW scoring_metrics AS
SELECT
    COUNT(*)                                          AS total_scored,
    COUNT(DISTINCT transaction_id)                    AS unique_transactions,
    AVG(fraud_probability)                            AS avg_fraud_probability,
    COUNT(*) FILTER (WHERE decision = 'block')        AS blocked_count,
    COUNT(*) FILTER (WHERE decision = 'approve')      AS approved_count,
    COUNT(*) FILTER (WHERE decision = 'review')       AS review_count,
    -- False negative rate: transactions that ARE fraud but were APPROVED
    CASE
        WHEN COUNT(*) FILTER (WHERE is_fraud = TRUE) > 0
        THEN COUNT(*) FILTER (WHERE is_fraud = TRUE AND decision = 'approve')::DOUBLE PRECISION
             / COUNT(*) FILTER (WHERE is_fraud = TRUE)
        ELSE 0.0
    END                                               AS false_negative_rate,
    -- False positive rate: transactions that are NOT fraud but were BLOCKED
    CASE
        WHEN COUNT(*) FILTER (WHERE is_fraud = FALSE) > 0
        THEN COUNT(*) FILTER (WHERE is_fraud = FALSE AND decision = 'block')::DOUBLE PRECISION
             / COUNT(*) FILTER (WHERE is_fraud = FALSE)
        ELSE 0.0
    END                                               AS false_positive_rate,
    MIN(scored_at)                                    AS first_scored,
    MAX(scored_at)                                    AS last_scored
FROM scored_transactions;

-- Duplicate detection query (used by resilience tests)
CREATE OR REPLACE VIEW duplicate_check AS
SELECT
    transaction_id,
    COUNT(*) AS score_count
FROM scored_transactions
GROUP BY transaction_id
HAVING COUNT(*) > 1;
