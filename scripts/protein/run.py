"""Run the public protein LLPS-to-DPR workflow from one stable entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _plan(argv: list[str]) -> int:
    from scripts.protein.workflows.release import RebuildPlan

    parser = argparse.ArgumentParser(description="Print the portable protein reproduction stage map.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        raise SystemExit("Only --dry-run is available until every rebuild stage is independently validated.")
    plan = RebuildPlan.from_roots(args.data_root, args.work_root, args.output_root)
    print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
    return 0


def _compile_llps_inputs(argv: list[str]) -> int:
    from scripts.protein.workflows.release import compile_llps_inputs

    parser = argparse.ArgumentParser(description="Compile local feature caches into fixed-plan LLPS inputs.")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compile_llps_inputs(
        release_root=args.release_root,
        feature_root=args.feature_root,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _build_dpr_sidecar(argv: list[str]) -> int:
    from scripts.protein.workflows.release import (
        build_packed_sidecar_from_feature_cache,
        make_llps_hidden_provider,
        validate_packed_sidecar,
    )

    parser = argparse.ArgumentParser(description="Build a validated DPR sidecar from local feature caches.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--llps-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protein-ids-file", type=Path)
    args = parser.parse_args(argv)
    protein_ids = None
    if args.protein_ids_file is not None:
        protein_ids = [line.strip() for line in args.protein_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = build_packed_sidecar_from_feature_cache(
        feature_dir=args.feature_dir,
        output_root=args.output_root,
        llps_hidden_provider=make_llps_hidden_provider(checkpoint=args.llps_checkpoint, device=args.device),
        protein_ids=protein_ids,
    )
    output = {"output_root": str(report.output_root.resolve()), "hidden_key": report.hidden_key}
    output.update(validate_packed_sidecar(report.output_root))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _delegate(command: str, argv: list[str]) -> int:
    from scripts.protein.workflows import evaluation, features, region_targets, release, training

    direct = {
        "validate-data": release.main,
        "region-targets": region_targets.main,
        "evaluate-llps": evaluation.evaluate_llps_main,
        "evaluate-phasepro": evaluation.main,
        "reproduce": _plan,
        "compile-llps-inputs": _compile_llps_inputs,
        "build-dpr-sidecar": _build_dpr_sidecar,
    }
    if command in direct:
        result = direct[command](argv)
        return 0 if result is None else int(result)
    sys.argv = [sys.argv[0], *argv]
    if command == "build-features":
        return int(features.build_features_main(argv) or 0)
    if command == "train-llps":
        return int(training.train_llps_main() or 0)
    if command == "train-dpr":
        return int(training.train_dpr_main() or 0)
    if command == "refine-dpr":
        return int(training.refine_dpr_main() or 0)
    raise AssertionError(command)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if len(raw) == 2 and raw[1] in {"-h", "--help"}:
        return _delegate(raw[0], [raw[1]])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "validate-data", "build-features", "compile-llps-inputs", "train-llps",
            "region-targets", "build-dpr-sidecar", "train-dpr", "refine-dpr",
            "evaluate-llps", "evaluate-phasepro", "reproduce",
        ),
    )
    args, remaining = parser.parse_known_args(raw)
    return _delegate(args.command, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
