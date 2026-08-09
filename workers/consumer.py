"""
SentinelAI Worker Pool Consumer Entrypoint.

Re-exports streaming.consumer for package organization.
Allows invoking via:
    python -m workers.consumer --worker-id worker-1
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from streaming.consumer import main

if __name__ == "__main__":
    main()
