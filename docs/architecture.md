# SentinelAI — System Architecture

## High-Level Pipeline

```mermaid
graph LR
    subgraph Ingestion
        P["Producer<br/>(replay / API)"]
    end

    subgraph "Message Broker"
        R["Redpanda<br/>12 partitions<br/>topic: transactions"]
    end

    subgraph "Worker Pool (Consumer Group)"
        W1["Worker 1<br/>partitions 0-2"]
        W2["Worker 2<br/>partitions 3-5"]
        W3["Worker 3<br/>partitions 6-8"]
        W4["Worker 4<br/>partitions 9-11"]
    end

    subgraph Scoring
        IF["Isolation Forest"]
        LR["Logistic Regression"]
        AS["A* Decision Engine"]
    end

    subgraph Sinks
        PG["PostgreSQL<br/>scored_transactions"]
        RD["Redis<br/>pub/sub + cache"]
    end

    subgraph Observability
        PR["Prometheus"]
        GR["Grafana"]
    end

    subgraph API
        FA["FastAPI<br/>/analyze /ingest /results"]
    end

    P --> R
    FA -->|/ingest| R
    R --> W1 & W2 & W3 & W4
    W1 & W2 & W3 & W4 --> IF & LR --> AS
    AS --> PG & RD
    W1 & W2 & W3 & W4 -->|metrics| PR
    FA -->|metrics| PR
    PR --> GR
    FA -->|query| PG
    RD -->|live feed| GR
```

## Component Responsibilities

### Producer (`streaming/producer.py`)
- Replays transaction dataset at configurable rate (tx/sec)
- Supports burst injection for spike simulation
- Idempotent writes (transaction ID as key)
- Partitions by `transaction_id % 12` for even distribution

### Redpanda (Message Broker)
- Kafka-compatible streaming platform
- 12 partitions on `transactions` topic
- Enables horizontal scaling up to 12 workers
- Provides consumer group coordination and offset management

### Worker Pool (`streaming/consumer.py`)
- Consumer group: `sentinel-scorers`
- Each worker is a separate **process** (multiprocessing, not threads)
- Owns a subset of partitions (auto-assigned by consumer group protocol)
- Loads models once at startup, scores in a tight loop
- Commits offsets only after successful sink write (at-least-once)

### Scorer (`streaming/scorer.py`)
- Wraps existing Isolation Forest + Logistic Regression + A* search
- Stateless after initialization → safe for concurrent use
- No SHAP/LLM in hot path (available on-demand via API)

### Sinks (`streaming/sink.py`)
- **PostgreSQL**: Persistent storage, batch inserts, queryable history
- **Redis**: Pub/sub for real-time dashboard, LRU cache for recent results

### FastAPI (`api/main.py`)
- `/ingest`: Alternative ingestion via HTTP (publishes to Redpanda)
- `/analyze`: Synchronous single-transaction scoring (existing)
- `/results/{id}`: Query scored results from PostgreSQL
- `/health`: System health including queue depth, circuit breaker state
- Protected by token-bucket rate limiter and circuit breaker

### Observability
- **Prometheus**: Scrapes worker + API metrics every 15s
- **Grafana**: Live dashboard with throughput, latency percentiles, queue depth, false negative rate

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Broker | Redpanda | No ZooKeeper, single binary, same Kafka protocol |
| Partitions | 12 | Scale 1→12 workers; one-way door, chosen deliberately high |
| Partition key | `transaction_id % N` | Even distribution; scoring is stateless per-tx, no ordering needed |
| Concurrency | Multiprocessing | CPU-bound scoring (sklearn); GIL prevents thread parallelism |
| Delivery | At-least-once | Simpler than exactly-once; sink is idempotent (UPSERT on tx_id) |
| Rate limiting | Custom token-bucket | Interviewable implementation; burst-tolerant |
| Failure handling | Circuit breaker | Fail fast under degraded scoring, don't queue indefinitely |

## Data Flow (Single Transaction)

```mermaid
sequenceDiagram
    participant P as Producer
    participant R as Redpanda
    participant W as Worker
    participant S as Scorer
    participant PG as PostgreSQL
    participant RD as Redis

    P->>R: publish(tx_json, key=tx_id)
    R->>W: poll() → batch of messages
    W->>S: score(tx_dict)
    S->>S: anomaly_score(IF)
    S->>S: fraud_probability(LR)
    S->>S: a_star_decision(cost_fn)
    S-->>W: ScoringResult
    W->>PG: batch_insert(results)
    W->>RD: publish(results)
    W->>R: commit_offsets()
```
