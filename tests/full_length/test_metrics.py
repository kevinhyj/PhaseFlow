import numpy as np

from phaseflow.full_length.metrics.key_metrics import key_topk_metrics
from phaseflow.full_length.metrics.protein_metrics import binary_classification_metrics
from phaseflow.full_length.metrics.region_metrics import boundary_f1, region_metrics
from phaseflow.full_length.metrics.residue_metrics import residue_binary_metrics


def test_metrics_do_not_crash() -> None:
    assert "auc" in binary_classification_metrics(np.array([0, 1]), np.array([0.2, 0.8]))
    assert "residue_dice" in residue_binary_metrics(np.array([0, 1, -100]), np.array([0.1, 0.9, 0.5]))
    assert "key_top2_precision" in key_topk_metrics(np.array([[0, 1, -100]]), np.array([[0.2, 0.8, 0.1]]), k=2)
    assert "region_iou@0.5_precision" in region_metrics(
        [[{"start": 1, "end": 4, "score": 0.9}]],
        [[{"start": 1, "end": 4, "type": "DPR_candidate"}]],
    )
    assert "boundary_f1" in boundary_f1(
        [[{"start": 1, "end": 4, "score": 0.9}]],
        [[{"start": 1, "end": 4, "region_type": "DPR_gold"}]],
    )
