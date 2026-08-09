# SentinelAI — Load Test Results

> Results from systematic load testing at multiple throughput levels.

## Test Environment

| Component | Specification |
|-----------|--------------|
| Machine | _fill in_ |
| CPU | _fill in_ |
| RAM | _fill in_ |
| OS | Windows + Docker Desktop |
| Redpanda | v24.1.1, 12 partitions |
| Python | 3.10 |

## Single-Worker Baseline

| Metric | Value |
|--------|-------|
| Throughput | _X_ tx/sec |
| Latency p50 | _X_ ms |
| Latency p95 | _X_ ms |
| Latency p99 | _X_ ms |
| False Negative Rate | _X%_ |

## Scaling Results

| Workers | Throughput (tx/sec) | p50 (ms) | p95 (ms) | p99 (ms) | FNR |
|---------|-------------------|----------|----------|----------|-----|
| 1 | _X_ | _X_ | _X_ | _X_ | _X%_ |
| 2 | _X_ | _X_ | _X_ | _X_ | _X%_ |
| 4 | _X_ | _X_ | _X_ | _X_ | _X%_ |
| 8 | _X_ | _X_ | _X_ | _X_ | _X%_ |

## Headline Statement

> "_Scaling from 1 to N workers increased throughput from X to Y tx/sec (Z% increase) while holding p99 latency at Wms and false-negative rate constant at V%._"

## Bottleneck Analysis

_After reaching the system ceiling at X tx/sec:_

| Potential Bottleneck | Evidence | Is Bottleneck? |
|---------------------|----------|---------------|
| Kafka partition count | _X_ | _Yes/No_ |
| Worker CPU saturation | _X_ | _Yes/No_ |
| Model inference (sklearn) | _X_ | _Yes/No_ |
| PostgreSQL write throughput | _X_ | _Yes/No_ |
| Network I/O | _X_ | _Yes/No_ |

## How to Reproduce

```bash
# Generate test data
python scripts/generate_stream_data.py --count 100000

# Run single-worker baseline
python load_tests/run_load_test.py --workers 1 --count 5000

# Run scaling test
python load_tests/run_load_test.py --workers 1,2,4,8 --count 5000

# Measure FNR
python load_tests/measure_false_negative_rate.py
```
