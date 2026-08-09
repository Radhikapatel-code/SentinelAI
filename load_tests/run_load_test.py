"""
SentinelAI — Load Test Runner.

Orchestrates load tests at multiple throughput levels and records
key metrics for the writeup.

Usage:
    python load_tests/run_load_test.py
    python load_tests/run_load_test.py --levels 1x,5x,10x --duration 60

Output:
    load_tests/results/load_test_TIMESTAMP.json — full results
    load_tests/results/summary_TIMESTAMP.csv   — tabular summary
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from streaming.config import get_streaming_config
from streaming.models import TransactionMessage
from streaming.scorer import ThreadSafeScorer

logger = logging.getLogger(__name__)


def run_single_worker_benchmark(
    scorer: ThreadSafeScorer,
    count: int = 1000,
) -> dict:
    """Benchmark a single worker's scoring throughput and latency.

    Args:
        scorer: Initialized scorer instance.
        count: Number of transactions to score.

    Returns:
        dict: Benchmark results (throughput, latency percentiles).
    """
    import numpy as np

    latencies = []

    for tx_id in range(1, count + 1):
        msg = TransactionMessage(
            transaction_id=tx_id,
            amount=500.0 + (tx_id % 1000),
            transaction_time=tx_id % 24,
            location_change=tx_id % 5 == 0,
            device_change=tx_id % 10 == 0,
            merchant_risk=round((tx_id % 100) / 100, 2),
        )

        start = time.perf_counter()
        scorer.score(msg)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    latencies_arr = np.array(latencies)
    total_time = sum(latencies) / 1000  # seconds

    return {
        "count": count,
        "total_time_seconds": round(total_time, 3),
        "throughput_tx_sec": round(count / total_time, 1),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "latency_mean_ms": round(float(latencies_arr.mean()), 3),
        "latency_max_ms": round(float(latencies_arr.max()), 3),
    }


def _worker_task(worker_id: int, count_per_worker: int, result_queue):
    """Top-level worker process task for Windows multiprocessing compatibility."""
    scorer = ThreadSafeScorer(
        data_path="data/transactions.csv",
        worker_id=f"bench-worker-{worker_id}",
    )

    latencies = []
    start_time = time.perf_counter()

    for tx_id in range(
        worker_id * count_per_worker + 1,
        (worker_id + 1) * count_per_worker + 1,
    ):
        msg = TransactionMessage(
            transaction_id=tx_id,
            amount=500.0,
            transaction_time=14,
            location_change=0,
            device_change=0,
            merchant_risk=0.2,
        )

        t0 = time.perf_counter()
        scorer.score(msg)
        latencies.append((time.perf_counter() - t0) * 1000)

    elapsed = time.perf_counter() - start_time
    result_queue.put({
        "worker_id": worker_id,
        "count": count_per_worker,
        "elapsed_seconds": elapsed,
        "latencies": latencies,
    })


def run_multiworker_benchmark(
    worker_count: int,
    count_per_worker: int = 500,
) -> dict:
    """Benchmark throughput with multiple worker processes.

    Args:
        worker_count: Number of parallel workers.
        count_per_worker: Transactions per worker.

    Returns:
        dict: Aggregate benchmark results.
    """
    import multiprocessing
    import numpy as np

    result_queue = multiprocessing.Queue()
    processes = []

    overall_start = time.perf_counter()

    for i in range(worker_count):
        p = multiprocessing.Process(
            target=_worker_task,
            args=(i, count_per_worker, result_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    overall_elapsed = time.perf_counter() - overall_start

    # Collect results
    all_latencies = []
    worker_results = []
    while not result_queue.empty():
        r = result_queue.get()
        all_latencies.extend(r["latencies"])
        worker_results.append(r)

    total_count = worker_count * count_per_worker
    latencies_arr = np.array(all_latencies)

    return {
        "worker_count": worker_count,
        "total_transactions": total_count,
        "total_time_seconds": round(overall_elapsed, 3),
        "aggregate_throughput_tx_sec": round(
            total_count / overall_elapsed, 1
        ),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "latency_mean_ms": round(float(latencies_arr.mean()), 3),
    }


def main() -> None:
    """Run the full load test suite."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="SentinelAI Load Test Runner"
    )
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Transactions per worker per test (default: 1000)"
    )
    parser.add_argument(
        "--workers", type=str, default="1,2,4,8",
        help="Comma-separated worker counts to test (default: 1,2,4,8)"
    )
    args = parser.parse_args()

    worker_counts = [int(w) for w in args.workers.split(",")]

    print("=" * 60)
    print("  SentinelAI Load Test Suite")
    print("=" * 60)

    # Single-worker baseline
    print("\n[+] Single-worker baseline benchmark...")
    scorer = ThreadSafeScorer(
        data_path="data/transactions.csv",
        worker_id="bench-baseline",
    )
    baseline = run_single_worker_benchmark(scorer, count=args.count)
    print(f"   Throughput: {baseline['throughput_tx_sec']} tx/sec")
    print(f"   Latency p50: {baseline['latency_p50_ms']}ms")
    print(f"   Latency p99: {baseline['latency_p99_ms']}ms")

    # Multi-worker scaling tests
    results = {"baseline": baseline, "scaling": []}

    for wc in worker_counts:
        print(f"\n[+] {wc}-worker benchmark...")
        result = run_multiworker_benchmark(
            worker_count=wc,
            count_per_worker=args.count // wc if wc > 1 else args.count,
        )
        results["scaling"].append(result)
        print(f"   Throughput: {result['aggregate_throughput_tx_sec']} tx/sec")
        print(f"   Latency p50: {result['latency_p50_ms']}ms")
        print(f"   Latency p99: {result['latency_p99_ms']}ms")

    # Save results
    os.makedirs("load_tests/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = f"load_tests/results/load_test_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV summary
    csv_path = f"load_tests/results/summary_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "workers", "throughput_tx_sec", "p50_ms", "p95_ms", "p99_ms",
        ])
        writer.writerow([
            1, baseline["throughput_tx_sec"],
            baseline["latency_p50_ms"],
            baseline["latency_p95_ms"],
            baseline["latency_p99_ms"],
        ])
        for r in results["scaling"]:
            writer.writerow([
                r["worker_count"],
                r["aggregate_throughput_tx_sec"],
                r["latency_p50_ms"],
                r["latency_p95_ms"],
                r["latency_p99_ms"],
            ])

    print(f"\n[SUCCESS] Results saved:")
    print(f"   {results_path}")
    print(f"   {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
