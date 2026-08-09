# 🧠 SentinelAI

### A Distributed, Real-Time Fraud Scoring Pipeline with Explainable AI

[![CI](https://github.com/Radhikapatel-code/SentinelAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Radhikapatel-code/SentinelAI/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[🚀 Live Dashboard](https://huggingface.co/spaces/raddhika/SentinelAI) | [📚 API Docs](https://sentinelai-uzhn.onrender.com/docs)

![SentinelAI Banner](https://raw.githubusercontent.com/Radhikapatel-code/SentinelAI/main/assets/banner.png)

---

## 📌 Overview & Stated Performance Goal

**SentinelAI** is a horizontally-scalable fraud detection system maintaining **sub-25ms p99 scoring latency at up to 2 parallel worker processes (81.2 → 141.3 tx/sec)**, scaling up to **189.5 tx/sec** at higher worker concurrency — while holding the **false-negative rate constant at 0.00%**.

Unlike traditional batch classifiers, SentinelAI treats fraud detection as an online **cost-optimized decision search** powered by Isolation Forest anomaly scoring, Logistic Regression risk probabilities, and cost-aware A* search executed across a distributed worker pool.

> 💡 **Data Source Disclosure**: System evaluation replays the 284,807-row Kaggle Credit Card Fraud dataset at controlled and variable rates to simulate live production streaming traffic.

---

## 🏗️ System Architecture

```mermaid
graph LR
    subgraph "Ingestion Layer"
        P["Ingestion Producer<br/>(ingestion/producer.py)"]
    end

    subgraph "Partitioned Broker"
        R["Redpanda Broker<br/>12 partitions"]
    end

    subgraph "Worker Pool (Multiprocessing)"
        W1["Worker Process 1"]
        W2["Worker Process 2"]
        W3["Worker Process 3"]
        W4["Worker Process N"]
    end

    subgraph "Scoring Engine"
        IF["Isolation Forest"]
        LR["Logistic Regression"]
        AS["A* Decision Search"]
    end

    subgraph "Storage Sinks"
        PG["PostgreSQL (UPSERT)"]
        RD["Redis (Pub/Sub)"]
    end

    subgraph "Observability Layer"
        PR["Prometheus Metrics"]
        GR["Grafana Dashboard"]
    end

    P --> R
    R --> W1 & W2 & W3 & W4
    W1 & W2 & W3 & W4 --> IF & LR --> AS
    AS --> PG & RD
    W1 & W2 & W3 & W4 -->|Prometheus| PR --> GR
```

See [Architecture Details](docs/architecture.md) for data flow and component specifications.

---

## 🔑 Architectural Design Rationale

| Architectural Area | Choice | Interview Rationale & Trade-offs |
|-------------------|--------|----------------------------------|
| **Message Broker** | Redpanda | C++ Kafka-compatible broker; zero ZooKeeper/JVM overhead, faster local iteration. |
| **Partitioning Strategy** | `account_id % N` or `tx_id % N` | `account_id % N` maintains per-account strict sequence ordering; `tx_id % N` provides perfectly uniform worker load distribution. |
| **Scale Ceiling** | 12 Partitions | Partition count sets the maximum horizontal worker scaling ceiling (1 to 12 workers) without requiring repartitioning. |
| **Concurrency Model** | Multiprocessing | CPU-bound sklearn inference bypasses Python GIL restrictions; each worker process loads read-only model instances into isolated memory space. |
| **Delivery Guarantee** | At-least-once + Idempotence | Idempotent PostgreSQL UPSERT (`ON CONFLICT (transaction_id) DO UPDATE`) achieves effectively-once results without 2PC transaction overhead. |
| **Rate Limiter** | Token-Bucket | Custom hand-coded token-bucket algorithm supporting burst capacity and non-blocking acquire. |
| **Fault Tolerance** | Circuit Breaker | 3-state circuit breaker (`CLOSED` → `OPEN` → `HALF_OPEN`) fast-fails degraded calls (< 1ms) to prevent cascading queue buildup. |

See [Design Decisions](docs/DESIGN_DECISIONS.md) for complete technical breakdowns.

---

## 🛡️ Failure Modes Handled (Resilience Summary)

See [Resilience Test Results](docs/RESILIENCE.md) for full benchmarks.

| Failure Scenario | Defensive Mechanism | Measured Outcome | Data Loss |
|------------------|-------------------|------------------|-----------|
| **Worker Process Crash** | Kafka Consumer Group Rebalance | Recovered & rebalanced in **12.4s** | **0 tx lost** |
| **Degraded Scoring Path** | 3-State Circuit Breaker | Fast-rejected in **0.42ms** when `OPEN` | **0 tx lost** |
| **10x Traffic Burst (5k tx/s)**| Queue Lag Backpressure | Backpressure triggered at 10k lag; drained in **18.2s** | **0 tx lost** |
| **Concurrent Execution** | Thread-Safe Scorer Wrapper | 100% deterministic decision matching | **0 duplicates** |

---

## 📊 Benchmark & Hardware Scaling Results

See [Load Test Results](docs/LOAD_TEST_RESULTS.md) for full metrics.

> 💻 **Test Hardware**: **4 Physical Cores / 8 Logical Threads** x86_64 CPU, 16 GB RAM, Windows 11.

| Scale Configuration | Aggregate Throughput | Scaling Efficiency | p50 Latency | p99 Latency | Sub-25ms Budget Met? | False Negative Rate (FNR) |
|---------------------|---------------------|--------------------|-------------|-------------|----------------------|--------------------------|
| **1 Worker (Baseline)** | **81.2 tx/sec** | — | **11.79 ms** | **21.54 ms** | ✅ Yes | **0.00%** (0 / 12) |
| **2 Workers** | **141.3 tx/sec** | **1.74x** (Near-linear) | **11.98 ms** | **23.01 ms** | ✅ Yes | **0.00%** (0 / 12) |
| **4 Workers** | **157.2 tx/sec** | 1.11x (Core ceiling) | **24.23 ms** | **47.18 ms** | ⚠️ Exceeded (47ms) | **0.00%** (0 / 12) |
| **8 Workers** | **189.5 tx/sec** | 1.21x (Contention) | **35.15 ms** | **118.80 ms**| ❌ Exceeded (118ms)| **0.00%** (0 / 12) |

> 🎯 **Hardware Bottleneck & Model Quality**:
> - **1 → 2 Workers**: Scales near-linearly (**1.74x**) within sub-25ms p99 latency (**23.01ms**) on dedicated physical cores.
> - **4 → 8 Workers**: Saturates 4 physical CPU cores; OS thread scheduling context-switching increases p99 latency to **47.18ms** (4 workers) and **118.80ms** (8 workers).
> - **Model Quality**: **0.00% False Negative Rate** (0 / 12 fraud cases approved out of 1,000 test transactions; 0.173% base fraud rate).

---

## 📝 Resume Integration & Interview Prep

### Resume Bullet
> *"Scaled real-time transaction scoring from 81.2 to 189.5 tx/sec across a 12-partition Redpanda consumer group using Python multiprocessing and consumer I/O batching, maintaining a 0.00% False-Negative Rate under load."*

### 🎙️ Verbal Walkthrough
Refer to [docs/VERBAL_WALKTHROUGH.md](docs/VERBAL_WALKTHROUGH.md) for a 2-minute spoken interview script covering partitioning choices, GIL concurrency, and fault recovery.

---

## ▶️ Quick Start

### 1. Full Stack (Docker Compose)
```bash
# Clone repository
git clone https://github.com/Radhikapatel-code/SentinelAI.git
cd SentinelAI

# Spin up Redpanda, Postgres, Redis, Prometheus, Grafana, API, Workers
docker compose up -d

# Run stream producer
python -m ingestion.producer --rate 500 --burst 2500
```

### 2. Run Benchmarks & Resilience Tests
```bash
# Unit & thread safety tests
pytest tests/test_scorer_thread_safety.py tests/test_rate_limiter.py -v

# Multi-worker throughput load test
python -u load_tests/run_load_test.py --count 200 --workers 1,2,4,8

# Model False Negative Rate evaluation
python -u load_tests/measure_false_negative_rate.py
```

---

## 📁 Repository Structure

```
SentinelAI/
├── api/                     # FastAPI backend (/analyze, /ingest, /results)
├── ingestion/               # Transaction streaming producer package
├── workers/                 # Multiprocessing consumer group worker pool
├── streaming/               # Core pipeline (scorer, rate limiter, circuit breaker, backpressure)
├── models/                  # Isolation Forest & Logistic Regression ML models
├── decision_engine/         # A* cost-aware decision search engine
├── explainability/          # SHAP & LLM natural language explainers
├── load_tests/              # Throughput benchmark & FNR measurement tools
├── monitoring/              # Prometheus scrapers & Grafana dashboards
├── docs/                    # Design decisions, load results, resilience report, verbal guide
├── docker-compose.yml       # Full service deployment stack
├── Dockerfile.worker        # Consumer process container
└── Dockerfile.api           # API service container
```

---

## 👤 Author

**Radhika Sanagadhiya**  
Undergrad in Information and Communication Technology (ICT) with CS minor  
📧 rp773061@gmail.com
