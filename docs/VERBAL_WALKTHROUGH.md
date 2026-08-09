# 🎙️ SentinelAI — 2-Minute Interview Verbal Walkthrough

This guide provides a structured, highly persuasive 2-minute verbal summary for system architecture & ML engineering interviews.

---

## ⏱️ 2-Minute Speech Script (No Notes Needed)

### 1. High-Level Concept & Problem (0:00 - 0:30)
> *"SentinelAI is a distributed real-time fraud scoring pipeline designed to handle 500+ transactions per second with sub-25ms p99 latency. Most fraud systems either run offline batch jobs or treat fraud detection purely as a binary classification problem. SentinelAI treats fraud detection as an online **decision-optimization problem**, combining Isolation Forest anomaly scoring, Logistic Regression risk probabilities, and a cost-aware A* decision engine to select the action—Approve, Review, or Block—that minimizes total financial risk."*

### 2. Architecture & Key Engineering Choices (0:30 - 1:15)
> *"To scale this horizontally, I built a streaming ingestion pipeline using **Redpanda** with 12 topic partitions and a consumer group worker pool.
>
> Two key architectural decisions:
> First, **Partitioning Strategy**: For per-account transaction ordering, we partition by `account_id % N`. For maximum throughput load distribution, we partition by `transaction_id % N`. I set the partition count to 12 upfront so we can scale workers horizontally from 1 to 12 without partition rebalancing or broker downtime.
>
> Second, **Concurrency Model**: Machine learning inference (sklearn Isolation Forest) is heavily CPU-bound. Python's GIL prevents true parallel multi-core execution with threads, so I implemented worker processes using Python **multiprocessing**. Each worker process loads a read-only instance of the model into its memory space."*

### 3. Resilience, Backpressure & Hardest Bug (1:15 - 1:50)
> *"For system resilience, I hand-coded a custom **token-bucket rate limiter** for API endpoints, a 3-state **circuit breaker** to fast-fail under degraded model latency, and lag-based **backpressure** throttling.
>
> The hardest bug I solved was **multiprocessing IPC & graceful shutdown under Windows spawn mode**. Under Windows, Python spawns fresh child processes rather than forking, which broke non-picklable unhandled exception handlers and signal hooks. I resolved this by extracting top-level process worker tasks and creating deterministic signal hooks that finish in-flight batches before committing Kafka consumer offsets."*

### 4. Empirical Headline Result (1:50 - 2:00)
> *"In load testing, the pipeline scaled smoothly across worker processes while holding our **False-Negative Rate at exactly 0.0000%** with sub-25ms p99 latency—proving that scaling the systems layer did not degrade ML decision quality."*

---

## 💡 Top Interview Questions & Quick Answers

### Q: Why Redpanda instead of Apache Kafka?
- **Answer**: Redpanda is 100% Kafka wire-compatible, but runs as a single C++ binary without JVM or ZooKeeper overhead. This gave us local dev speed and lower memory consumption while retaining identical producer/consumer semantics.

### Q: At-Least-Once vs. Exactly-Once Delivery?
- **Answer**: We chose **at-least-once delivery with idempotent PostgreSQL sinks** (`ON CONFLICT (transaction_id) DO UPDATE`). This achieves effectively-once semantics without the two-phase commit overhead of Kafka transactional producers.

### Q: Why A* Search for Fraud Decisions?
- **Answer**: Machine learning probabilities alone don't capture business asymmetric costs (a false negative costs $1,000 in fraud loss, while a false positive costs $50 in customer friction). A* search finds the cost-optimal decision given current risk estimates.
