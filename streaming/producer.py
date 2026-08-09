"""
SentinelAI — Transaction Stream Producer.

Replays a transaction dataset as a Kafka/Redpanda stream at a
configurable rate with support for burst injection.

Usage:
    python -m streaming.producer
    python -m streaming.producer --rate 500 --duration 60
    python -m streaming.producer --rate 500 --burst 5 --burst-duration 10

Design decisions:
    - Idempotent producer (enable.idempotence=true): prevents duplicate
      messages on retry without requiring consumer-side deduplication.
    - Partition key = transaction_id: distributes evenly across partitions.
      Since scoring is stateless per-transaction, we don't need per-account
      ordering — even distribution maximizes parallelism.
    - Rate control via time.sleep with adaptive adjustment: simple,
      deterministic, and observable. For higher accuracy at extreme rates,
      would use a token-bucket approach.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from typing import Optional

import pandas as pd

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from confluent_kafka import Producer, KafkaError
from streaming.config import get_streaming_config, StreamingConfig
from streaming.models import TransactionMessage

logger = logging.getLogger(__name__)

# ── Graceful shutdown ──────────────────────────
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown signal received, finishing current batch...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class TransactionProducer:
    """Replays transaction data as a Kafka/Redpanda stream.

    Attributes:
        producer: confluent-kafka Producer instance.
        topic: Target Kafka topic.
        config: Streaming configuration.
    """

    def __init__(self, config: Optional[StreamingConfig] = None) -> None:
        """Initialize the producer.

        Args:
            config: Streaming configuration. Uses defaults if None.
        """
        self.config = config or get_streaming_config()
        self.topic = self.config.kafka.topic

        self.producer = Producer({
            "bootstrap.servers": self.config.kafka.bootstrap_servers,
            "enable.idempotence": True,
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 100,
            "linger.ms": 5,
            "batch.num.messages": 100,
            "compression.type": "lz4",
            "client.id": "sentinel-producer",
        })

        self._total_sent = 0
        self._total_errors = 0

    def _delivery_callback(self, err, msg):
        """Kafka delivery callback for async produce confirmations."""
        if err is not None:
            self._total_errors += 1
            logger.error(
                "Delivery failed for tx %s: %s",
                msg.key().decode() if msg.key() else "?",
                err,
            )
        else:
            self._total_sent += 1

    def produce_one(self, tx: TransactionMessage) -> None:
        """Produce a single transaction message.

        Args:
            tx: Transaction to publish.
        """
        key = str(tx.transaction_id).encode("utf-8")
        value = tx.to_json().encode("utf-8")

        self.producer.produce(
            topic=self.topic,
            key=key,
            value=value,
            callback=self._delivery_callback,
        )

    def replay_dataset(
        self,
        data_path: str,
        rate: int = 500,
        duration: int = 60,
        burst_multiplier: int = 1,
        burst_duration: int = 10,
        burst_interval: int = 30,
    ) -> dict:
        """Replay a CSV dataset as a stream at a controlled rate.

        Args:
            data_path: Path to the transaction CSV.
            rate: Target transactions per second.
            duration: Total run duration in seconds (0 = until data exhausted).
            burst_multiplier: Spike multiplier during burst periods.
            burst_duration: Duration of each burst in seconds.
            burst_interval: Seconds between burst starts.

        Returns:
            dict: Summary statistics (total_sent, errors, actual_rate, etc.)
        """
        logger.info(
            "Loading dataset from %s for replay at %d tx/sec...",
            data_path, rate,
        )

        df = pd.read_csv(data_path)
        logger.info("Loaded %d transactions for replay", len(df))

        start_time = time.time()
        batch_start = start_time
        sent_in_batch = 0
        total_produced = 0
        row_index = 0

        # Calculate timing
        interval = 1.0 / rate  # seconds between messages at base rate
        burst_active = False
        next_burst_at = start_time + burst_interval

        logger.info(
            "Starting replay: rate=%d tx/sec, duration=%ds, "
            "burst=%dx every %ds for %ds",
            rate, duration, burst_multiplier, burst_interval, burst_duration,
        )

        while not _shutdown_requested:
            elapsed = time.time() - start_time

            # Check duration limit
            if duration > 0 and elapsed >= duration:
                logger.info("Duration limit reached (%ds)", duration)
                break

            # Burst logic
            current_rate = rate
            now = time.time()
            if burst_multiplier > 1:
                if not burst_active and now >= next_burst_at:
                    burst_active = True
                    burst_end = now + burst_duration
                    logger.info(
                        "🔥 BURST START: %dx rate (%d tx/sec) for %ds",
                        burst_multiplier, rate * burst_multiplier,
                        burst_duration,
                    )
                elif burst_active and now >= burst_end:
                    burst_active = False
                    next_burst_at = now + burst_interval
                    logger.info("📉 BURST END: back to %d tx/sec", rate)

                if burst_active:
                    current_rate = rate * burst_multiplier

            current_interval = 1.0 / current_rate

            # Get next transaction (wrap around if dataset exhausted)
            row = df.iloc[row_index % len(df)]
            row_index += 1

            # Build message
            tx = TransactionMessage(
                transaction_id=total_produced + 1,  # Unique across replays
                amount=float(row["amount"]),
                transaction_time=int(row["transaction_time"]),
                location_change=int(row["location_change"]),
                device_change=int(row["device_change"]),
                merchant_risk=float(row["merchant_risk"]),
                is_fraud=bool(row["is_fraud"]) if "is_fraud" in row.index else None,
                ingested_at=time.time(),
            )

            self.produce_one(tx)
            total_produced += 1
            sent_in_batch += 1

            # Periodic flush and progress reporting
            if sent_in_batch >= 500:
                self.producer.poll(0)  # Trigger delivery callbacks
                elapsed_batch = time.time() - batch_start
                actual_rate = sent_in_batch / max(elapsed_batch, 0.001)
                logger.info(
                    "Progress: %d sent (%.0f tx/sec actual, %d errors)",
                    total_produced, actual_rate, self._total_errors,
                )
                sent_in_batch = 0
                batch_start = time.time()

            # Rate control
            time.sleep(current_interval)

        # Final flush
        logger.info("Flushing remaining messages...")
        remaining = self.producer.flush(timeout=10)
        if remaining > 0:
            logger.warning("%d messages were not delivered", remaining)

        # Summary
        total_elapsed = time.time() - start_time
        actual_rate = total_produced / max(total_elapsed, 0.001)

        summary = {
            "total_produced": total_produced,
            "total_delivered": self._total_sent,
            "total_errors": self._total_errors,
            "duration_seconds": round(total_elapsed, 2),
            "actual_rate_tx_sec": round(actual_rate, 1),
            "target_rate_tx_sec": rate,
        }

        logger.info("Replay complete: %s", json.dumps(summary, indent=2))
        return summary


def main() -> None:
    """CLI entry point for the producer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SentinelAI Transaction Stream Producer"
    )
    parser.add_argument(
        "--rate", type=int, default=None,
        help="Target transactions per second (default: from config)"
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Duration in seconds (default: from config)"
    )
    parser.add_argument(
        "--burst", type=int, default=None,
        help="Burst multiplier (default: from config)"
    )
    parser.add_argument(
        "--burst-duration", type=int, default=10,
        help="Burst duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to transaction CSV (default: from config)"
    )
    args = parser.parse_args()

    config = get_streaming_config()

    producer = TransactionProducer(config)
    summary = producer.replay_dataset(
        data_path=args.data_path or config.producer.data_path,
        rate=args.rate or config.producer.rate,
        duration=args.duration or config.producer.duration,
        burst_multiplier=args.burst or config.producer.burst_multiplier,
        burst_duration=args.burst_duration,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
