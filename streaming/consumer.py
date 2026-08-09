"""
SentinelAI — Consumer Group Worker.

Each worker process:
    1. Joins the 'sentinel-scorers' consumer group
    2. Is assigned a subset of partitions by the group coordinator
    3. Polls messages in batches
    4. Scores each transaction via ThreadSafeScorer
    5. Writes results to PostgreSQL + Redis
    6. Commits offsets only AFTER successful sink write

This is the component that makes the system horizontally scalable:
    - Adding workers (up to partition count) adds throughput linearly
    - Consumer group protocol handles partition rebalancing automatically
    - At-least-once semantics: offsets committed after sink write

Concurrency model: multiprocessing (one process per worker)
    - Scoring is CPU-bound (sklearn inference) → Python's GIL prevents
      true parallelism with threads
    - Each process loads its own model copy (~few MB, acceptable)
    - No shared mutable state between processes

Graceful shutdown:
    - SIGTERM/SIGINT: finish current batch, commit offsets, close sinks
    - In-flight transactions complete; nothing is silently dropped

Usage:
    python -m streaming.consumer
    python -m streaming.consumer --workers 4
"""

import argparse
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
import uuid
from typing import Optional

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from confluent_kafka import Consumer, KafkaError, KafkaException
from streaming.config import get_streaming_config, StreamingConfig
from streaming.models import TransactionMessage, ScoringResult
from streaming.scorer import ThreadSafeScorer
from streaming.sink import PostgresSink, RedisSink, CompositeSink

logger = logging.getLogger(__name__)


class ScoringWorker:
    """A single scoring worker that consumes from Kafka and scores transactions.

    Each worker runs in its own process, owns a subset of partitions,
    and writes results to the configured sinks.

    Attributes:
        worker_id: Unique identifier for this worker.
        config: Streaming configuration.
        scorer: Thread-safe scoring pipeline.
        sink: Composite sink (PostgreSQL + Redis).
        consumer: Kafka consumer instance.
    """

    def __init__(
        self,
        worker_id: str,
        config: Optional[StreamingConfig] = None,
    ) -> None:
        """Initialize the worker.

        Args:
            worker_id: Unique worker identifier (used in logging + metrics).
            config: Streaming configuration.
        """
        self.worker_id = worker_id
        self.config = config or get_streaming_config()
        self._shutdown_requested = False
        self._running = False

        # Will be initialized in run() (after fork, if multiprocessing)
        self.scorer: Optional[ThreadSafeScorer] = None
        self.sink: Optional[CompositeSink] = None
        self.consumer: Optional[Consumer] = None

    def _init_components(self) -> None:
        """Initialize scorer, sinks, and Kafka consumer.

        Called inside the worker process (after fork) to ensure
        each process has its own connections and model copies.
        """
        logger.info("Worker %s initializing components...", self.worker_id)

        # Initialize scorer (loads and trains models)
        self.scorer = ThreadSafeScorer(
            data_path=os.environ.get(
                "SCORER_DATA_PATH", "data/transactions.csv"
            ),
            worker_id=self.worker_id,
        )

        # Initialize sinks
        postgres_sink = None
        redis_sink = None

        try:
            postgres_sink = PostgresSink(
                dsn=self.config.postgres.dsn,
                batch_size=self.config.postgres.batch_size,
            )
        except Exception:
            logger.warning(
                "PostgreSQL sink unavailable, results will not be persisted",
                exc_info=True,
            )

        try:
            redis_sink = RedisSink(
                url=self.config.redis.url,
            )
        except Exception:
            logger.warning(
                "Redis sink unavailable, real-time feed disabled",
                exc_info=True,
            )

        self.sink = CompositeSink(
            postgres_sink=postgres_sink,
            redis_sink=redis_sink,
        )

        # Initialize Kafka consumer
        self.consumer = Consumer({
            "bootstrap.servers": self.config.kafka.bootstrap_servers,
            "group.id": self.config.kafka.consumer_group,
            "auto.offset.reset": self.config.kafka.auto_offset_reset,
            "enable.auto.commit": False,  # Manual commit after sink write
            "session.timeout.ms": self.config.kafka.session_timeout_ms,
            "max.poll.interval.ms": 300000,
            "client.id": self.worker_id,
        })

        self.consumer.subscribe([self.config.kafka.topic])
        logger.info(
            "Worker %s subscribed to topic '%s' in group '%s'",
            self.worker_id,
            self.config.kafka.topic,
            self.config.kafka.consumer_group,
        )

    def _handle_shutdown(self, signum, frame):
        """Signal handler for graceful shutdown."""
        logger.info(
            "Worker %s received shutdown signal (%s), "
            "finishing current batch...",
            self.worker_id, signum,
        )
        self._shutdown_requested = True

    def run(self) -> None:
        """Main worker loop: poll → score → sink → commit.

        This method blocks until shutdown is requested via signal
        or an unrecoverable error occurs.
        """
        # Register signal handlers for this process
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self._init_components()
        self._running = True

        total_scored = 0
        total_errors = 0
        batch_start = time.time()

        logger.info("Worker %s entering main loop", self.worker_id)

        try:
            while not self._shutdown_requested:
                # Poll for messages
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition — normal, keep polling
                        continue
                    else:
                        logger.error(
                            "Worker %s Kafka error: %s",
                            self.worker_id, msg.error(),
                        )
                        total_errors += 1
                        continue

                # Deserialize message
                try:
                    tx_msg = TransactionMessage.from_json(
                        msg.value().decode("utf-8")
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.error(
                        "Worker %s failed to deserialize message: %s",
                        self.worker_id, e,
                    )
                    total_errors += 1
                    # Commit offset to skip malformed messages
                    self.consumer.commit(msg)
                    continue

                # Score transaction
                try:
                    result = self.scorer.score(tx_msg)
                except Exception as e:
                    logger.error(
                        "Worker %s scoring failed for tx %d: %s",
                        self.worker_id, tx_msg.transaction_id, e,
                    )
                    total_errors += 1
                    continue

                # Write to sinks
                try:
                    self.sink.write_batch([result])
                except Exception as e:
                    logger.error(
                        "Worker %s sink write failed for tx %d: %s",
                        self.worker_id, result.transaction_id, e,
                    )
                    total_errors += 1
                    # Don't commit offset — message will be reprocessed
                    continue

                # Commit offset ONLY after successful sink write
                self.consumer.commit(msg)
                total_scored += 1

                # Periodic progress logging
                if total_scored % 500 == 0:
                    elapsed = time.time() - batch_start
                    rate = 500 / max(elapsed, 0.001)
                    logger.info(
                        "Worker %s: %d scored (%.0f tx/sec, %d errors)",
                        self.worker_id, total_scored, rate, total_errors,
                    )
                    batch_start = time.time()

        except KeyboardInterrupt:
            logger.info("Worker %s interrupted", self.worker_id)
        finally:
            self._shutdown(total_scored, total_errors)

    def _shutdown(self, total_scored: int, total_errors: int) -> None:
        """Graceful shutdown: flush sinks, close consumer.

        Args:
            total_scored: Total transactions scored.
            total_errors: Total errors encountered.
        """
        logger.info(
            "Worker %s shutting down (scored=%d, errors=%d)",
            self.worker_id, total_scored, total_errors,
        )

        self._running = False

        # Flush remaining sink buffer
        if self.sink:
            try:
                self.sink.flush()
                self.sink.close()
            except Exception:
                logger.exception("Worker %s sink cleanup failed", self.worker_id)

        # Close Kafka consumer
        if self.consumer:
            try:
                self.consumer.close()
            except Exception:
                logger.exception(
                    "Worker %s consumer cleanup failed", self.worker_id
                )

        logger.info("Worker %s shutdown complete", self.worker_id)


def _run_worker(worker_id: str) -> None:
    """Entry point for a worker process.

    Args:
        worker_id: Unique worker identifier.
    """
    # Configure logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{worker_id}] [%(levelname)s] %(name)s: %(message)s",
    )

    worker = ScoringWorker(worker_id=worker_id)
    worker.run()


def main() -> None:
    """CLI entry point: launch N worker processes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SentinelAI Scoring Worker"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker processes to launch (default: 1). "
             "When running in Docker, use docker compose --scale instead."
    )
    args = parser.parse_args()

    if args.workers == 1:
        # Single worker — run in main process (simpler debugging)
        worker_id = os.environ.get(
            "WORKER_ID",
            f"worker-{uuid.uuid4().hex[:8]}",
        )
        _run_worker(worker_id)
    else:
        # Multiple workers — launch as separate processes
        logger.info("Launching %d worker processes", args.workers)
        processes = []

        for i in range(args.workers):
            worker_id = f"worker-{i}"
            p = multiprocessing.Process(
                target=_run_worker,
                args=(worker_id,),
                name=worker_id,
            )
            p.start()
            processes.append(p)
            logger.info("Started %s (pid=%d)", worker_id, p.pid)

        # Wait for all workers to complete
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Main process interrupted, terminating workers...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join(timeout=10)

        logger.info("All workers stopped")


if __name__ == "__main__":
    main()
