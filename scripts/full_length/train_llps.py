#!/usr/bin/env python3
"""Train the full-length LLPS model from a YAML configuration."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseflow.full_length.train import main


if __name__ == "__main__":
    main()
