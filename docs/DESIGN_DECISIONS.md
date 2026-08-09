# SentinelAI — Design Decisions

> Every scaling decision is documented here with rationale, alternatives considered, and interview-ready explanations.

---

## 1. Redpanda over Kafka

**Decision:** Use Redpanda as the message broker instead of Apache Kafka.

**Why:**
- Single binary — no ZooKeeper dependency, simpler local dev
- Kafka-compatible protocol — same `confluent-kafka` client library works
- Faster startup (< 2s vs 15-30s for Kafka + ZooKeeper)
- Lower resource footprint for single-node development

**Interview answer:**
> "I chose Redpanda for local development because it's a single binary with no ZooKeeper dependency, which made iteration faster. It speaks the exact same Kafka protocol, so the same confluent-kafka client code would work unchanged against a production Kafka cluster. The architectural decision is about the streaming _pattern_, not the specific broker."

---

## 2. 12 Partitions

**Decision:** Create the `transactions` topic with 12 partitions.

**Why:**
- **Scaling ceiling:** Each partition can only be consumed by one worker in a consumer group. 12 partitions allows scaling from 1 to 12 workers.
- **One-way door:** Kafka partition count can be increased but not decreased cleanly. Starting at 12 (3x initial worker count of 4) provides room to demonstrate scaling.
- **Load distribution:** At 500 tx/sec target, each partition handles ~42 tx/sec, well within capacity.

**What I'd change in production:**
- Production would use 24-48 partitions for a team of workers
- Partition key would be `account_id` if we needed per-account ordering for sequential fraud detection

---

## 3. Multiprocessing over asyncio

**Decision:** Use multiprocessing (separate OS processes) for scoring workers, not asyncio or threading.

**Why:**
- **CPU-bound work:** Isolation Forest and Logistic Regression inference are CPU-bound operations (numpy/sklearn)
- **Python's GIL:** The Global Interpreter Lock prevents true parallelism with threads for CPU work
- **asyncio limitation:** `asyncio` is designed for I/O-bound work (network calls, disk I/O). Using it for CPU-bound scoring would block the event loop.
- **Process isolation:** Each worker process has its own memory space, model copy, and GIL — true parallel CPU utilization

**Trade-off:** Each process loads its own copy of the model (~few MB). This is acceptable given the throughput gain.

---

## 4. Token-Bucket over Leaky-Bucket Rate Limiting

**Decision:** Implement a token-bucket rate limiter (custom, no library).

**Why:**
- **Burst tolerance:** Token-bucket allows bursts up to `capacity` — matches our traffic pattern (sustained 500 tx/sec with bursts to 2,500)
- **Leaky-bucket alternative:** Enforces strict constant rate, would reject legitimate burst traffic
- **Custom implementation:** Built from scratch so the algorithm is defensible in an interview — it's a straightforward O(1) operation per request

**Algorithm:**
```
refill_tokens = elapsed_time × rate
tokens = min(capacity, tokens + refill_tokens)
if tokens >= 1: consume, allow
else: reject (429)
```

---

## 5. At-Least-Once over Exactly-Once Delivery

**Decision:** Use at-least-once delivery semantics with an idempotent sink.

**Why:**
- **Simplicity:** Exactly-once in Kafka requires transactional producers + consumers, which adds significant complexity
- **Idempotent sink:** PostgreSQL `INSERT ON CONFLICT DO UPDATE` (UPSERT) makes duplicate delivery harmless
- **Correctness:** `at-least-once + idempotent sink = effectively exactly-once`
- **Trade-off:** Slight overhead from rare duplicate processing, but no data loss risk

---

## 6. Partition Key: `transaction_id` (Even Distribution)

**Decision:** Partition by `transaction_id % partition_count` for even distribution.

**Why:**
- **Scoring is stateless:** Each transaction is scored independently. There's no need to keep the same account's transactions on the same partition.
- **Even distribution:** `transaction_id` distributes uniformly across partitions, maximizing parallelism.

**When I'd change this:**
- If the model needed sequential processing per account (e.g., detecting velocity patterns), I'd partition by `account_id` — but that risks hot partitions for high-volume accounts.

---

## 7. Circuit Breaker (Fail Fast, Don't Queue)

**Decision:** Wrap the scoring path with a circuit breaker that fails fast when scoring is degraded.

**Why:**
- **Without circuit breaker:** Slow scoring → messages queue up → memory pressure → cascading failure
- **With circuit breaker:** After N failures, immediately reject new requests (< 1ms) rather than waiting for timeout
- **Recovery:** HALF_OPEN state tests recovery with a few requests before fully reopening

---

## 8. SHAP/LLM Removed from Streaming Hot Path

**Decision:** Streaming workers score with Isolation Forest + Logistic Regression + A* only. SHAP explanations and LLM calls are available on-demand via the API.

**Why:**
- SHAP computation adds ~50-100ms per transaction
- LLM calls add ~500ms-2s per transaction
- At 500 tx/sec, this would reduce throughput by 10-100x
- Explanations are needed for human review, not for every scored transaction
- The `/analyze` endpoint still provides full SHAP+LLM for individual transactions

---

## 9. Single Machine, Multi-Process (Honest Framing)

**Decision:** Simulate distribution with multiple processes on a single machine rather than a true multi-node setup.

**Why:**
- The architecture is genuinely distributed (Kafka consumer groups, independent worker containers, shared-nothing scoring)
- Adding machines is a deployment configuration change, not an architectural one
- Single-machine testing is standard for portfolio projects and is stated honestly

**Interview framing:**
> "The architecture is designed for multi-node deployment — each worker is a separate container with its own model copy and no shared state. I tested on a single machine with multiple processes because the scaling pattern is identical: Kafka consumer groups handle partition assignment the same way whether workers are on one machine or ten."
