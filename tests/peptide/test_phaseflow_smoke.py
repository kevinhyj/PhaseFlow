import unittest

from phaseflow import AminoAcidTokenizer

try:
    from phaseflow import PhaseFlow
    PHASEFLOW_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    PhaseFlow = None
    PHASEFLOW_IMPORT_ERROR = exc


class TokenizerSmokeTest(unittest.TestCase):
    def test_encode_decode_sequence(self):
        tokenizer = AminoAcidTokenizer()
        tokens = tokenizer.encode_sequence("ACDE")

        self.assertEqual(tokens, [0, 1, 2, 3])
        self.assertEqual(tokenizer.decode_sequence(tokens), "ACDE")

    def test_build_input_sequence_contains_modal_markers(self):
        tokenizer = AminoAcidTokenizer()
        tokens = tokenizer.build_input_sequence("ACDE")

        self.assertEqual(tokens[0], tokenizer.SOS_ID)
        self.assertIn(tokenizer.META_ID, tokens)
        self.assertEqual(tokens[-1], tokenizer.SOM_ID)


class ModelSmokeTest(unittest.TestCase):
    @unittest.skipIf(PhaseFlow is None, f"missing model dependency: {PHASEFLOW_IMPORT_ERROR}")
    def test_small_model_initializes(self):
        model = PhaseFlow(
            dim=32,
            depth=1,
            heads=2,
            dim_head=16,
            vocab_size=32,
            phase_dim=16,
            max_seq_len=16,
            dropout=0.0,
            use_set_encoder=True,
        )

        self.assertEqual(model.phase_dim, 16)
        self.assertEqual(model.vocab_size, 32)


if __name__ == "__main__":
    unittest.main()
