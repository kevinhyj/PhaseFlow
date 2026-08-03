"""Prepare public protein inputs through the stable workflow CLI."""

from __future__ import annotations

import argparse
import sys

from run import _delegate


COMMANDS = {
    "features": "build-features",
    "llps-inputs": "compile-llps-inputs",
    "region-targets": "region-targets",
    "dpr-sidecar": "build-dpr-sidecar",
    "validate-data": "validate-data",
}


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if len(raw) == 2 and raw[1] in {"-h", "--help"} and raw[0] in COMMANDS:
        return _delegate(COMMANDS[raw[0]], [raw[1]])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(COMMANDS))
    args, remaining = parser.parse_known_args(raw)
    return _delegate(COMMANDS[args.stage], remaining)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
