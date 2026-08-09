"""
SentinelAI Streaming Configuration.

Centralized configuration for the real-time scoring pipeline.
All values can be overridden via environment variables.

Usage:
    from streaming.config import get_streaming_config
    config = get_streaming_config()
    print(config.kafka.bootstrap_servers)
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KafkaConfig:
    """Kafka/Redpanda broker configuration.

    Attributes:
        bootstrap_servers: Comma-separated broker addresses.
        topic: Topic name for incoming transactions.
        scored_topic: Topic name for scored results.
        consumer_group: Consumer group ID for scoring workers.
        partition_count: Number of partitions (for documentation;
            actual count is set at topic creation time).
        auto_offset_reset: Where to start consuming if no committed offset.
        enable_auto_commit: Whether to auto-commit offsets (disabled for
            at-least-once semantics — we commit manually after sink write).
        max_poll_records: Max records per poll batch.
        session_timeout_ms: Consumer session timeout for rebalancing.
    """

    bootstrap_servers: str = "localhost:19092"
    topic: str = "transactions"
    scored_topic: str = "scored-transactions"
    consumer_group: str = "sentinel-scorers"
    partition_count: int = 12
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    max_poll_records: int = 100
    session_timeout_ms: int = 30000


@dataclass(frozen=True)
class PostgresConfig:
    """PostgreSQL sink configuration.

    Attributes:
        dsn: Full connection string.
        pool_min: Minimum connections in the pool.
        pool_max: Maximum connections in the pool.
        batch_size: Number of results to batch before inserting.
    """

    dsn: str = "postgresql://sentinel:sentinel_dev@localhost:5432/sentinel"
    pool_min: int = 2
    pool_max: int = 10
    batch_size: int = 50


@dataclass(frozen=True)
class RedisConfig:
    """Redis cache/pub-sub configuration.

    Attributes:
        url: Redis connection URL.
        channel: Pub/sub channel for scored results.
        cache_ttl: TTL in seconds for cached results.
        max_cache_size: Max number of recent results to cache.
    """

    url: str = "redis://localhost:6379/0"
    channel: str = "sentinel:scored"
    cache_ttl: int = 3600
    max_cache_size: int = 10000


@dataclass(frozen=True)
class ProducerConfig:
    """Transaction producer configuration.

    Attributes:
        rate: Target transactions per second.
        burst_multiplier: Spike multiplier (e.g., 5 = 5x burst).
        burst_duration: Duration of each burst in seconds.
        duration: Total run duration in seconds (0 = unlimited).
        data_path: Path to the transaction data CSV for replay.
    """

    rate: int = 500
    burst_multiplier: int = 1
    burst_duration: int = 10
    duration: int = 60
    data_path: str = "data/stream_transactions.csv"


@dataclass(frozen=True)
class RateLimiterConfig:
    """Token-bucket rate limiter configuration.

    Attributes:
        rate: Tokens per second (refill rate).
        capacity: Maximum bucket capacity (burst size).
    """

    rate: float = 1000.0
    capacity: float = 2500.0


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Circuit breaker configuration.

    Attributes:
        failure_threshold: Consecutive failures before opening.
        recovery_timeout: Seconds to wait before half-open test.
        success_threshold: Successes in half-open before closing.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3


@dataclass(frozen=True)
class BackpressureConfig:
    """Backpressure monitoring configuration.

    Attributes:
        lag_threshold: Consumer lag (messages) that triggers throttling.
        check_interval: Seconds between lag checks.
    """

    lag_threshold: int = 5000
    check_interval: float = 5.0


@dataclass(frozen=True)
class MetricsConfig:
    """Prometheus metrics configuration.

    Attributes:
        port: Port to expose metrics HTTP endpoint.
        prefix: Metric name prefix.
    """

    port: int = 8001
    prefix: str = "sentinel"


@dataclass(frozen=True)
class StreamingConfig:
    """Top-level streaming pipeline configuration.

    Aggregates all component configs.
    """

    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    producer: ProducerConfig = field(default_factory=ProducerConfig)
    rate_limiter: RateLimiterConfig = field(default_factory=RateLimiterConfig)
    circuit_breaker: CircuitBreakerConfig = field(
        default_factory=CircuitBreakerConfig
    )
    backpressure: BackpressureConfig = field(
        default_factory=BackpressureConfig
    )
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def get_streaming_config() -> StreamingConfig:
    """Load streaming configuration from environment variables.

    Environment variables override defaults. Variable names follow
    the pattern: COMPONENT_FIELD (e.g., KAFKA_BOOTSTRAP_SERVERS).

    Returns:
        StreamingConfig: Fully resolved streaming configuration.
    """
    return StreamingConfig(
        kafka=KafkaConfig(
            bootstrap_servers=os.environ.get(
                "KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"
            ),
            topic=os.environ.get("KAFKA_TOPIC", "transactions"),
            scored_topic=os.environ.get(
                "KAFKA_SCORED_TOPIC", "scored-transactions"
            ),
            consumer_group=os.environ.get(
                "KAFKA_CONSUMER_GROUP", "sentinel-scorers"
            ),
            partition_count=int(
                os.environ.get("KAFKA_PARTITION_COUNT", "12")
            ),
            max_poll_records=int(
                os.environ.get("KAFKA_MAX_POLL_RECORDS", "100")
            ),
            session_timeout_ms=int(
                os.environ.get("KAFKA_SESSION_TIMEOUT_MS", "30000")
            ),
        ),
        postgres=PostgresConfig(
            dsn=os.environ.get(
                "POSTGRES_DSN",
                "postgresql://sentinel:sentinel_dev@localhost:5432/sentinel",
            ),
            batch_size=int(os.environ.get("POSTGRES_BATCH_SIZE", "50")),
        ),
        redis=RedisConfig(
            url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        ),
        producer=ProducerConfig(
            rate=int(os.environ.get("PRODUCER_RATE", "500")),
            burst_multiplier=int(
                os.environ.get("PRODUCER_BURST_MULTIPLIER", "1")
            ),
            burst_duration=int(
                os.environ.get("PRODUCER_BURST_DURATION", "10")
            ),
            duration=int(os.environ.get("PRODUCER_DURATION", "60")),
            data_path=os.environ.get(
                "PRODUCER_DATA_PATH", "data/stream_transactions.csv"
            ),
        ),
        rate_limiter=RateLimiterConfig(
            rate=float(os.environ.get("RATE_LIMITER_RATE", "1000.0")),
            capacity=float(
                os.environ.get("RATE_LIMITER_CAPACITY", "2500.0")
            ),
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=int(
                os.environ.get("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
            ),
            recovery_timeout=float(
                os.environ.get("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30.0")
            ),
            success_threshold=int(
                os.environ.get("CIRCUIT_BREAKER_SUCCESS_THRESHOLD", "3")
            ),
        ),
        backpressure=BackpressureConfig(
            lag_threshold=int(
                os.environ.get("BACKPRESSURE_LAG_THRESHOLD", "5000")
            ),
            check_interval=float(
                os.environ.get("BACKPRESSURE_CHECK_INTERVAL", "5.0")
            ),
        ),
        metrics=MetricsConfig(
            port=int(os.environ.get("METRICS_PORT", "8001")),
        ),
    )
