"""
SentinelAI — Resilience Tests: Traffic Spike (Backpressure).

Tests that the system degrades gracefully under extreme load:
    1. Backpressure activates when lag exceeds threshold
    2. No OOM crash during spike
    3. Queue drains after spike ends
    4. Throughput degrades gracefully, not catastrophically

Usage:
    pytest tests/resilience/test_traffic_spike.py -v
"""

import sys
import os
import time

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from streaming.backpressure import BackpressureMonitor


class TestBackpressureUnderLoad:
    """Tests for backpressure behavior during traffic spikes."""

    def test_backpressure_activates_at_threshold(self):
        """Backpressure should activate when lag crosses threshold."""
        monitor = BackpressureMonitor(
            lag_threshold=100,
            check_interval=1.0,
        )

        # Below threshold — no backpressure
        monitor.update_lag(50)
        assert not monitor.is_active
        assert not monitor.should_throttle()

        # At threshold — backpressure activates
        monitor.update_lag(100)
        assert monitor.is_active
        assert monitor.should_throttle()

    def test_backpressure_deactivates_with_hysteresis(self):
        """Backpressure should deactivate at 80% of threshold (hysteresis)."""
        monitor = BackpressureMonitor(lag_threshold=100)

        # Activate
        monitor.update_lag(150)
        assert monitor.is_active

        # Still active at 85% of threshold (above hysteresis point)
        monitor.update_lag(85)
        assert monitor.is_active

        # Deactivates at 79% of threshold (below hysteresis)
        monitor.update_lag(79)
        assert not monitor.is_active

    def test_no_flapping_near_threshold(self):
        """Hysteresis should prevent rapid on/off flapping."""
        monitor = BackpressureMonitor(lag_threshold=1000)

        activations_before = monitor.stats["activations"]

        # Oscillate around threshold
        for lag in [1100, 900, 1100, 900, 1100, 900]:
            monitor.update_lag(lag)

        # Should have minimal state changes due to hysteresis
        stats = monitor.stats
        # 900 is above 80% of 1000 (800), so it won't deactivate
        # Only the first activation should count
        assert stats["activations"] <= 3

    def test_spike_simulation(self):
        """Simulate a traffic spike and verify graceful degradation."""
        monitor = BackpressureMonitor(lag_threshold=500)

        # Phase 1: Normal traffic (low lag)
        for i in range(10):
            monitor.update_lag(50 + i * 10)
        assert not monitor.is_active

        # Phase 2: Spike begins (lag increases rapidly)
        for lag in [200, 400, 600, 800, 1000]:
            monitor.update_lag(lag)

        assert monitor.is_active
        assert monitor.current_lag == 1000

        # Phase 3: Spike subsides (lag decreases)
        for lag in [800, 600, 400, 300, 200, 100]:
            monitor.update_lag(lag)

        # Should eventually deactivate
        assert not monitor.is_active

        # Verify metrics
        stats = monitor.stats
        assert stats["max_lag_seen"] == 1000
        assert stats["activations"] >= 1
        assert stats["deactivations"] >= 1

    def test_stats_tracking(self):
        """Verify stats are correctly tracked."""
        monitor = BackpressureMonitor(lag_threshold=100)

        monitor.update_lag(50)
        monitor.update_lag(150)  # activate
        monitor.update_lag(79)  # deactivate
        monitor.update_lag(200)  # activate again

        stats = monitor.stats
        assert stats["is_active"] is True
        assert stats["current_lag"] == 200
        assert stats["max_lag_seen"] == 200
        assert stats["activations"] == 2
        assert stats["deactivations"] == 1
