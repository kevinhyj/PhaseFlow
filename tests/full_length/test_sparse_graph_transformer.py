import torch

from phaseflow.full_length.models.sparse_graph_transformer import SparseGraphTransformer


def test_sparse_graph_transformer_empty_neighbor_safe() -> None:
    model = SparseGraphTransformer(d_model=16, num_layers=1, num_heads=4, edge_dim=8, ffn_dim=32)
    x = torch.randn(2, 5, 16)
    neighbors = torch.zeros(2, 5, 3, dtype=torch.long)
    edge_attr = torch.zeros(2, 5, 3, 8)
    neighbor_mask = torch.zeros(2, 5, 3, dtype=torch.bool)
    neighbor_mask[:, :, 0] = True
    seq_mask = torch.ones(2, 5, dtype=torch.bool)
    out = model(x, neighbors, edge_attr, neighbor_mask, seq_mask)
    assert out.shape == x.shape
    assert not out.isnan().any()
