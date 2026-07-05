import numpy as np

from phaseflow.full_length.features.starling_runner import (
    assemble_starling_segments,
    candidate_starling_segments,
    starling_features_from_distance_maps,
)


def test_starling_distance_maps_become_node_and_contacts() -> None:
    maps = np.stack(
        [
            np.asarray(
                [
                    [0.0, 4.0, 12.0],
                    [4.0, 0.0, 5.0],
                    [12.0, 5.0, 0.0],
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    [0.0, 6.0, 14.0],
                    [6.0, 0.0, 6.0],
                    [14.0, 6.0, 0.0],
                ],
                dtype=np.float32,
            ),
        ]
    )
    node, missing, reliability, contacts = starling_features_from_distance_maps(
        maps,
        "ACD",
        contact_threshold=11.0,
        contact_topk=2,
    )
    assert node.shape == (3, 8)
    assert missing.sum() == 0.0
    assert reliability.min() > 0.0
    assert contacts.shape[1] == 5


def test_long_sequence_starling_segments_map_to_full_length() -> None:
    sequence = "A" * 400 + "GPGPGPGPGPGPGPGPGPGPGPGPGPGPGP" + "A" * 400
    segments = candidate_starling_segments("p1", sequence, max_segment_length=64, min_segment_length=16)
    assert segments
    segment = segments[0]
    segment_node = np.ones((len(segment.sequence), 8), dtype=np.float32)
    segment_missing = np.zeros(len(segment.sequence), dtype=np.float32)
    segment_reliability = np.ones(len(segment.sequence), dtype=np.float32)
    segment_contacts = np.asarray([[0, 1, 0.8, 5.0]], dtype=np.float32)
    node, missing, reliability, contacts = assemble_starling_segments(
        len(sequence),
        [(segment, segment_node, segment_missing, segment_reliability, segment_contacts)],
    )
    assert node.shape == (len(sequence), 8)
    assert missing[segment.start : segment.end].sum() == 0.0
    assert reliability[segment.start : segment.end].min() == 1.0
    assert contacts[0, 0] == segment.start
