# SentinelAI — Target Load Profile

> Every scaling decision in this project is justified against these numbers.

## Target Throughput

| Metric | Value | Rationale |
|--------|-------|-----------|
| **Sustained throughput** | 500 tx/sec | Realistic for a mid-tier financial institution (~43M tx/day) |
| **Burst throughput** | 2,500 tx/sec (5x) | Black Friday / flash sale spike simulation |
| **Stress test ceiling** | 5,000 tx/sec (10x) | Find the breaking point, document actual bottleneck |

## Latency Targets

| Metric | Target |
|--------|--------|
| **p50 scoring latency** | < 5ms |
| **p95 scoring latency** | < 20ms |
| **p99 scoring latency** | < 50ms |
| **End-to-end (ingest → sink)** | < 100ms at sustained load |

## Data Source

- **Dataset**: Synthetic transactions generated with 5-feature schema matching production model
- **Replay strategy**: Controlled-rate replay of pre-generated CSV via Kafka producer
- **Honesty note**: This is simulated load from a static dataset, not live production traffic. The streaming infrastructure, partitioning, and failure handling are real; the data source is replayed. This is standard for portfolio projects and stated explicitly.

## Infrastructure (Single Machine)

| Component | Count | Notes |
|-----------|-------|-------|
| Redpanda partitions | 12 | One-way door — can increase, not decrease |
| Worker processes | 1–12 | Scale up to partition count |
| PostgreSQL | 1 instance | Result sink |
| Redis | 1 instance | Hot cache + pub/sub for dashboard |

## Why These Numbers

- **500 tx/sec sustained** is the number that makes the horizontal scaling story meaningful — it's high enough that a single worker can't handle it alone (single worker ≈ 150-200 tx/sec based on sklearn inference benchmarks), forcing you to actually scale.
- **12 partitions** allows demonstrating linear scaling from 1→4→8→12 workers without repartitioning.
- **Single machine, multi-process** is honest. Multi-machine distribution is acknowledged as a production step, not simulated.
