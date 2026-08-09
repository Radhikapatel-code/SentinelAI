"""
SentinelAI Streaming Ingestion Package.
"""

from streaming.producer import TransactionProducer, main as run_producer

__all__ = ["TransactionProducer", "run_producer"]
