"""
SentinelAI Streaming Module.

Provides real-time transaction scoring via a distributed worker pool.
Components:
    - producer: Replays transaction data as a Kafka/Redpanda stream
    - consumer: Consumer group workers that score transactions
    - scorer: Thread-safe wrapper around the ML scoring pipeline
    - sink: PostgreSQL and Redis result sinks
    - rate_limiter: Custom token-bucket rate limiter
    - circuit_breaker: Circuit breaker for degraded scoring
    - backpressure: Queue depth monitoring and throttling
    - metrics: Prometheus instrumentation
"""
