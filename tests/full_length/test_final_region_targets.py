from argparse import Namespace

import h5py
import numpy as np

from scripts.full_length.data.build_region_targets import build_targets_for_group


def _args(use_phaseflow: bool) -> Namespace:
    return Namespace(
        policy="stratified",
        use_phaseflow=use_phaseflow,
        phaseflow_pos=0.62,
        phasemotif_pos=0.70,
        pstp_pos=0.75,
        catgranule_pos=0.70,
        psphunter_key_pos=0.65,
        consensus_pos=0.55,
        confidence_pos=0.55,
        consensus_neg=0.25,
        phaseflow_neg=0.20,
        phasemotif_neg=0.15,
        pstp_neg=0.20,
        catgranule_neg=0.58,
        psphunter_neg=0.05,
        disorder_hard_neg=0.55,
        min_pos_len=3,
        min_neg_len=3,
        merge_gap=0,
        boundary_radius=1,
    )


def test_final_region_targets_ignore_phaseflow_by_default(tmp_path) -> None:
    path = tmp_path / "teachers.h5"
    length = 8
    with h5py.File(path, "w") as handle:
        group = handle.create_group("p1")
        group.create_dataset("teacher_consensus", data=np.full(length, 0.9, dtype=np.float32))
        group.create_dataset("teacher_confidence", data=np.full(length, 0.9, dtype=np.float32))
        group.create_dataset("phaseflow_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("phasemotif_score", data=np.full(length, 0.05, dtype=np.float32))
        group.create_dataset("pstp_scan_score", data=np.full(length, 0.05, dtype=np.float32))
        group.create_dataset("catgranule_score", data=np.full(length, 0.05, dtype=np.float32))
        group.create_dataset("psphunter_key_score", data=np.full(length, 0.05, dtype=np.float32))

    with h5py.File(path, "r") as handle:
        no_phaseflow = build_targets_for_group("p1", handle["p1"], [], _args(False))
        with_phaseflow = build_targets_for_group("p1", handle["p1"], [], _args(True))

    assert int(np.nansum(no_phaseflow["region_teacher_target"] == 1.0)) == 0
    assert int(np.nansum(with_phaseflow["region_teacher_target"] == 1.0)) == length


def test_pstp_scan_policy_uses_only_pstp_profile(tmp_path) -> None:
    path = tmp_path / "teachers.h5"
    length = 10
    with h5py.File(path, "w") as handle:
        group = handle.create_group("p1")
        group.create_dataset("phaseflow_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("phasemotif_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("catgranule_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("psphunter_key_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("pstp_scan_score", data=np.full(length, 0.05, dtype=np.float32))

    args = _args(False)
    args.policy = "pstp_scan"
    with h5py.File(path, "r") as handle:
        low_pstp = build_targets_for_group("p1", handle["p1"], [], args)

    assert len(low_pstp["positive_spans"]) == 0
    assert int(np.sum(np.asarray(low_pstp["region_teacher_target"]) >= 0.5)) == 0

    with h5py.File(path, "a") as handle:
        del handle["p1"]["pstp_scan_score"]
        handle["p1"].create_dataset("pstp_scan_score", data=np.full(length, 0.9, dtype=np.float32))

    with h5py.File(path, "r") as handle:
        high_pstp = build_targets_for_group("p1", handle["p1"], [], args)

    assert len(high_pstp["positive_spans"]) == 1
    assert int(np.sum(np.asarray(high_pstp["region_teacher_target"]) >= 0.5)) == length
