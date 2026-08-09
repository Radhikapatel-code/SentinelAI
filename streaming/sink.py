"""
SentinelAI — PostgreSQL and Redis Sinks.

Writes scored transaction results to persistent storage (PostgreSQL)
and real-time cache (Redis) for dashboard consumption.

Design decisions:
    - PostgreSQL uses UPSERT (ON CONFLICT DO UPDATE) for idempotent writes,
      which is critical for at-least-once delivery semantics. If a worker
      retries a batch, no duplicates are created.
    - Redis pub/sub enables real-time dashboard without polling.
    - Both sinks use connection pooling for efficiency under load.
"""

import json
import logging
import time
from typing import Optional

from streaming.models import ScoringResult

logger = logging.getLogger(__name__)


class PostgresSink:
    """Batch-inserts scored results into PostgreSQL.

    Uses UPSERT (INSERT ON CONFLICT UPDATE) to guarantee idempotent
    writes even under at-least-once delivery semantics.

    Attributes:
        dsn: PostgreSQL connection string.
        batch_size: Number of results to buffer before flushing.
    """

    def __init__(self, dsn: str, batch_size: int = 50) -> None:
        """Initialize PostgreSQL sink with connection pool.

        Args:
            dsn: PostgreSQL connection string.
            batch_size: Flush threshold.
        """
        import psycopg2
        from psycopg2 import pool as pg_pool
        from psycopg2.extras import execute_values

        self.dsn = dsn
        self.batch_size = batch_size
        self._buffer: list[ScoringResult] = []

        self._pool = pg_pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=dsn,
        )
        self._execute_values = execute_values

        logger.info("PostgreSQL sink initialized (dsn=%s)", dsn[:40] + "...")

    def write(self, result: ScoringResult) -> None:
        """Buffer a single result; flush if buffer is full.

        Args:
            result: Scoring result to write.
        """
        self._buffer.append(result)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def write_batch(self, results: list[ScoringResult]) -> None:
        """Write a batch of results directly (bypasses buffer).

        Args:
            results: List of scoring results to write.
        """
        self._buffer.extend(results)
        self.flush()

    def flush(self) -> int:
        """Flush buffered results to PostgreSQL.

        Returns:
            int: Number of rows written.

        Raises:
            psycopg2.Error: On database errors (connection, constraint, etc.)
        """
        if not self._buffer:
            return 0

        batch = self._buffer.copy()
        self._buffer.clear()

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # UPSERT: idempotent write for at-least-once semantics
                sql = """
                    INSERT INTO scored_transactions (
                        transaction_id, amount, anomaly_score,
                        fraud_probability, decision, expected_cost,
                        is_fraud, worker_id, scored_at, ingested_at
                    ) VALUES %s
                    ON CONFLICT (transaction_id) DO UPDATE SET
                        anomaly_score = EXCLUDED.anomaly_score,
                        fraud_probability = EXCLUDED.fraud_probability,
                        decision = EXCLUDED.decision,
                        expected_cost = EXCLUDED.expected_cost,
                        worker_id = EXCLUDED.worker_id,
                        scored_at = EXCLUDED.scored_at
                """

                values = []
                for r in batch:
                    from datetime import datetime, timezone
                    scored_dt = datetime.fromtimestamp(
                        r.scored_at, tz=timezone.utc
                    )
                    ingested_dt = datetime.fromtimestamp(
                        r.ingested_at, tz=timezone.utc
                    ) if r.ingested_at else None

                    values.append((
                        r.transaction_id,
                        r.amount,
                        r.anomaly_score,
                        r.fraud_probability,
                        r.decision,
                        r.expected_cost,
                        r.is_fraud,
                        r.worker_id,
                        scored_dt,
                        ingested_dt,
                    ))

                self._execute_values(cur, sql, values)
                conn.commit()

            logger.debug("Flushed %d results to PostgreSQL", len(batch))
            return len(batch)

        except Exception:
            conn.rollback()
            logger.exception("PostgreSQL flush failed, %d results lost", len(batch))
            raise
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        """Flush remaining buffer and close connection pool."""
        self.flush()
        self._pool.closeall()
        logger.info("PostgreSQL sink closed")


class RedisSink:
    """Publishes scored results to Redis for real-time consumption.

    Uses pub/sub for live dashboard feeds and sorted sets for
    recent result caching.

    Attributes:
        url: Redis connection URL.
        channel: Pub/sub channel name.
    """

    def __init__(
        self,
        url: str,
        channel: str = "sentinel:scored",
        cache_ttl: int = 3600,
    ) -> None:
        """Initialize Redis sink.

        Args:
            url: Redis connection URL.
            channel: Pub/sub channel for live results.
            cache_ttl: TTL in seconds for cached results.
        """
        import redis as redis_lib

        self._redis = redis_lib.from_url(url, decode_responses=True)
        self.channel = channel
        self.cache_ttl = cache_ttl

        # Verify connection
        self._redis.ping()
        logger.info("Redis sink initialized (channel=%s)", channel)

    def write(self, result: ScoringResult) -> None:
        """Publish a scoring result to Redis.

        Args:
            result: Scoring result to publish.
        """
        payload = result.to_json()

        # Pub/sub for live consumers (dashboard)
        self._redis.publish(self.channel, payload)

        # Cache in a sorted set (by scored_at) for recent results
        self._redis.zadd(
            "sentinel:recent_scores",
            {payload: result.scored_at},
        )

        # Trim to keep only the latest N results
        self._redis.zremrangebyrank("sentinel:recent_scores", 0, -10001)

    def write_batch(self, results: list[ScoringResult]) -> None:
        """Write a batch of results to Redis.

        Uses pipelining for efficiency.

        Args:
            results: List of scoring results.
        """
        pipe = self._redis.pipeline()

        for result in results:
            payload = result.to_json()
            pipe.publish(self.channel, payload)
            pipe.zadd(
                "sentinel:recent_scores",
                {payload: result.scored_at},
            )

        pipe.zremrangebyrank("sentinel:recent_scores", 0, -10001)
        pipe.execute()

        logger.debug("Published %d results to Redis", len(results))

    def get_recent(self, count: int = 100) -> list[dict]:
        """Get the most recent scored results from cache.

        Args:
            count: Number of recent results to retrieve.

        Returns:
            list[dict]: Recent scoring results, newest first.
        """
        raw = self._redis.zrevrange(
            "sentinel:recent_scores", 0, count - 1
        )
        return [json.loads(r) for r in raw]

    def close(self) -> None:
        """Close Redis connection."""
        self._redis.close()
        logger.info("Redis sink closed")


class CompositeSink:
    """Writes to both PostgreSQL and Redis.

    Wraps both sinks behind a single interface. PostgreSQL write
    failures are considered fatal (raised); Redis failures are
    logged but not fatal (dashboard degradation is acceptable).
    """

    def __init__(
        self,
        postgres_sink: Optional[PostgresSink] = None,
        redis_sink: Optional[RedisSink] = None,
    ) -> None:
        """Initialize composite sink.

        Args:
            postgres_sink: PostgreSQL sink (optional).
            redis_sink: Redis sink (optional).
        """
        self.postgres = postgres_sink
        self.redis = redis_sink

    def write_batch(self, results: list[ScoringResult]) -> None:
        """Write a batch to all configured sinks.

        Args:
            results: Scoring results to write.

        Raises:
            Exception: If PostgreSQL write fails (critical path).
        """
        # PostgreSQL: critical path — failure is fatal
        if self.postgres:
            self.postgres.write_batch(results)

        # Redis: best-effort — failure degrades dashboard, not scoring
        if self.redis:
            try:
                self.redis.write_batch(results)
            except Exception:
                logger.warning(
                    "Redis write failed for %d results (non-fatal)",
                    len(results),
                    exc_info=True,
                )

    def flush(self) -> None:
        """Flush all sinks."""
        if self.postgres:
            self.postgres.flush()

    def close(self) -> None:
        """Close all sinks."""
        if self.postgres:
            self.postgres.close()
        if self.redis:
            self.redis.close()
