"""
SentinelAI Streaming Ingestion Producer Entrypoint.

Re-exports streaming.producer for package organization.
Allows invoking via:
    python -m ingestion.producer --rate 500 --burst 2500
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from streaming.producer import main

if __name__ == "__main__":
    main()
