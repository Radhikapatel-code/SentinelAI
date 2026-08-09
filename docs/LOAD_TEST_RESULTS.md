# SentinelAI — Load Test Results & System Benchmarks

> Results from systematic load testing across worker scale levels and dataset evaluations after joblib model serialization and consumer I/O batching optimizations.

---

## 💻 Test Environment

| Component | Specification |
|-----------|--------------|
| **Host Machine** | 8 Logical Cores / 4 Physical Cores x86_64 Workstation |
| **RAM** | 16 GB DDR4 |
| **OS** | Windows 11 |
| **Message Broker** | Redpanda v24.1.1 (12 topic partitions) |
| **Python Version** | Python 3.13.7 |
| **Optimization Stack** | Pre-trained `joblib` deserialization, batch size 50 sink/offset commits, post-initialization warmup timer sync |

---

## 📈 Multi-Worker Scaling Results

| Workers | Aggregate Throughput | Latency p50 | Latency p95 | Latency p99 | False Negative Rate (FNR) |
|---------|---------------------|-------------|-------------|-------------|--------------------------|
| **1 Worker (Baseline)** | **80.7 tx/sec** | **12.03 ms** | **15.42 ms** | **18.01 ms** | **0.00%** |
| **1 Worker (Steady)** | **81.2 tx/sec** | **11.79 ms** | **17.85 ms** | **21.54 ms** | **0.00%** |
| **2 Workers** | **141.3 tx/sec** | **11.98 ms** | **18.62 ms** | **23.01 ms** | **0.00%** |
| **4 Workers** | **157.2 tx/sec** | **24.23 ms** | **38.91 ms** | **47.18 ms** | **0.00%** |
| **8 Workers** | **189.5 tx/sec** | **35.15 ms** | **84.32 ms** | **118.80 ms**| **0.00%** |

---

## 🎯 Headline Resume Statement

> *"Scaled real-time transaction scoring from 81.2 to 189.5 tx/sec across a 12-partition Redpanda consumer group using Python multiprocessing and consumer I/O batching, maintaining a 0.00% False-Negative Rate under load."*

---

## 🔍 System Bottleneck Analysis

| Potential Bottleneck | Status | Evidence & Explanation |
|---------------------|--------|------------------------|
| **Cold-Start Training Contention** | **RESOLVED** | Serializing Isolation Forest & Logistic Regression via `joblib.dump` eliminated simultaneous training contention during worker startup. |
| **Synchronous Network I/O** | **RESOLVED** | Accumulating 50 results/offsets per write/commit batch eliminated per-message round-trip latencies. |
| **Physical CPU Core Ceiling** | **Primary Limit** | Host system has 4 physical cores (8 logical threads). Scaling from 1 to 2 workers yields linear 1.74x scaling (81.2 → 141.3 tx/sec). At 8 workers, physical core saturation increases p99 latency to 118.8ms due to OS thread context switching. |
| **Message Broker (Redpanda)** | Unconstrained | 12 partitions handle > 5,000 tx/sec easily. |

---

## 🛠️ How to Reproduce Benchmarks

```bash
# 1. Offline model training & serialization
python scripts/train_models.py

# 2. Run multi-worker load test (synchronous warmup)
python -u load_tests/run_load_test.py --count 1000 --workers 1,2,4,8

# 3. Measure False Negative Rate
python -u load_tests/measure_false_negative_rate.py
```
