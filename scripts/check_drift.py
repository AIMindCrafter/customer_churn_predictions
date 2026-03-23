"""CLI script to run drift detection on recent predictions.

Usage:
    python scripts/check_drift.py
    python scripts/check_drift.py --n-recent 200 --threshold 3.0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.drift import run_drift_check
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run data drift check")
    parser.add_argument("--n-recent", type=int, default=100,
                        help="Number of recent prediction batches to analyze")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Z-score threshold for numeric drift")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to JSON file")
    args = parser.parse_args()

    try:
        results = run_drift_check(n_recent=args.n_recent, threshold=args.threshold)
        print(json.dumps(results, indent=2))

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Report saved to %s", args.output)

        # Exit with code 1 if drift detected (useful for CI)
        sys.exit(1 if results["drift_detected"] else 0)

    except FileNotFoundError as exc:
        logger.error("Cannot run drift check: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
