#!/bin/bash
# ──────────────────────────────────────────────
# SentinelAI — Create Redpanda Topics
# ──────────────────────────────────────────────
# Usage:
#   ./scripts/create_topics.sh
#   ./scripts/create_topics.sh --brokers localhost:19092
#
# Why 12 partitions:
#   - Allows scaling from 1 to 12 workers without repartitioning
#   - At 500 tx/sec target, each partition handles ~42 tx/sec
#   - Partition count is a ONE-WAY DOOR in Kafka: can increase, cannot decrease
#   - 12 was chosen deliberately higher than initial worker count (4)
#     to support demonstrating horizontal scaling

set -euo pipefail

BROKERS="${1:-localhost:19092}"
PARTITIONS=12
REPLICAS=1
RETENTION_MS=86400000  # 24 hours

echo "🔧 Creating SentinelAI topics on ${BROKERS}..."

rpk topic create transactions \
    --brokers "${BROKERS}" \
    --partitions "${PARTITIONS}" \
    --replicas "${REPLICAS}" \
    --topic-config "retention.ms=${RETENTION_MS}" \
    2>/dev/null || echo "  ℹ️  'transactions' topic already exists"

rpk topic create scored-transactions \
    --brokers "${BROKERS}" \
    --partitions "${PARTITIONS}" \
    --replicas "${REPLICAS}" \
    --topic-config "retention.ms=${RETENTION_MS}" \
    2>/dev/null || echo "  ℹ️  'scored-transactions' topic already exists"

echo "✅ Topics ready:"
rpk topic list --brokers "${BROKERS}"
