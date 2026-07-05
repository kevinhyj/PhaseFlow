from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

from phaseflow.full_length.models.phaseflow import PhaseFlowModel
from phaseflow.full_length.models.sparse_graph_transformer import SparseGraphTransformer


DEFAULT_PHASEFLOW_REPO = Path("external/phaseflow-peptide")
DEFAULT_PHASEFLOW_SITE_PACKAGES = Path("external/phaseflow-peptide/site-packages")
PROTENIX_EDGE_TYPE = 20
PHASEFLOW_LLPS_LAYER_KEY = "phaseflow_llps"
PHASEFLOW_LLPS_HIDDEN_LAYERS_KEY = "phaseflow_llps_hidden_layers"
PHASEFLOW_LLPS_HIDDEN_DIM_KEY = "phaseflow_llps_hidden_dim"
LEGACY_LLPS_PREFIX = "phase" + "gt"
LEGACY_LLPS_HIDDEN_LAYERS_KEY = LEGACY_LLPS_PREFIX + "_hidden_layers"
LEGACY_LLPS_HIDDEN_DIM_KEY = LEGACY_LLPS_PREFIX + "_hidden_dim"


class ScalarLayerMix(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        self.logits = nn.Parameter(torch.zeros(int(num_layers)))
        self.gamma = nn.Parameter(torch.ones(()))

    def forward(self, layers: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(layers) != int(self.logits.numel()):
            raise ValueError(f"Expected {self.logits.numel()} layers, got {len(layers)}")
        stacked = torch.stack([layer.float() for layer in layers], dim=0)
        weights = torch.softmax(self.logits.float(), dim=0).view(-1, 1, 1, 1)
        return self.gamma.float() * torch.sum(weights * stacked, dim=0)


class PhaseStackDPR(nn.Module):
    """Frozen PhaseFlow + frozen PhaseFlow with a trainable DPR graph head."""

    def __init__(
        self,
        *,
        llps_backbone: nn.Module | None = None,
        phaseflow: nn.Module,
        dpr_config: dict[str, Any] | None = None,
        llps_hidden_names: tuple[str, ...] = ("llps_residue_repr", "dpr_residue_repr"),
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        if llps_backbone is None:
            llps_backbone = legacy_kwargs.pop(LEGACY_LLPS_PREFIX, None)
        if legacy_kwargs:
            raise TypeError(f"unexpected keyword argument(s): {sorted(legacy_kwargs)}")
        if llps_backbone is None:
            raise ValueError("llps_backbone is required")
        self.llps_backbone = llps_backbone
        self.phaseflow = phaseflow
        self.dpr_config = _default_dpr_config() | dict(dpr_config or {})
        self.llps_hidden_names = tuple(llps_hidden_names)
        self.phaseflow_window_size = int(self.dpr_config.get("phaseflow_window_size", 20))
        self.phaseflow_stride = int(self.dpr_config.get("phaseflow_stride", 1))
        self.phaseflow_shape = str(self.dpr_config.get("phaseflow_shape", "4x4"))
        self.disable_protenix_for_dpr = bool(self.dpr_config.get("disable_protenix_for_dpr", True))
        self.disable_protenix_edges_for_dpr = bool(self.dpr_config.get("disable_protenix_edges_for_dpr", True))
        self.phaseflow_max_windows_per_sequence = int(self.dpr_config.get("phaseflow_max_windows_per_sequence", 0))
        self.phaseflow_window_batch_size = int(self.dpr_config.get("phaseflow_window_batch_size", 128))

        d_model = int(self.dpr_config["d_model"])
        llps_dim = int(self.dpr_config.get(PHASEFLOW_LLPS_HIDDEN_DIM_KEY, self.dpr_config.get(LEGACY_LLPS_HIDDEN_DIM_KEY, d_model)))
        phaseflow_dim = int(self.dpr_config.get("phaseflow_hidden_dim", getattr(phaseflow, "dim", d_model)))
        edge_dim = int(self.dpr_config.get("edge_dim", 32))

        self.layer_mix = nn.ModuleDict(
            {
                PHASEFLOW_LLPS_LAYER_KEY: ScalarLayerMix(len(self.llps_hidden_names)),
                "phaseflow": ScalarLayerMix(int(self.dpr_config.get("phaseflow_num_mixed_layers", 4))),
            }
        )
        self.dpr_stem = nn.Sequential(
            nn.Linear(llps_dim + phaseflow_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.dpr_blocks = SparseGraphTransformer(
            d_model=d_model,
            num_layers=int(self.dpr_config["num_graph_transformer_layers"]),
            num_heads=int(self.dpr_config["num_heads"]),
            edge_dim=edge_dim,
            ffn_dim=int(self.dpr_config["ffn_dim"]),
            dropout=float(self.dpr_config["dropout"]),
            num_edge_types=int(self.dpr_config.get("num_edge_types", 40)),
            relative_position_bins=int(self.dpr_config.get("relative_position_bins", 32)),
        )
        self.dpr_residue_head = _mlp_head(d_model, 1, float(self.dpr_config["dropout"]))
        self.dpr_boundary_head = _mlp_head(d_model, 2, float(self.dpr_config["dropout"]))

        self._freeze_backbones()

    def _freeze_backbones(self) -> None:
        self.llps_backbone.requires_grad_(False)
        self.phaseflow.requires_grad_(False)
        self.llps_backbone.eval()
        self.phaseflow.eval()

    def train(self, mode: bool = True) -> "PhaseStackDPR":
        super().train(mode)
        self.llps_backbone.eval()
        self.phaseflow.eval()
        self.layer_mix.train(mode)
        self.dpr_stem.train(mode)
        self.dpr_blocks.train(mode)
        self.dpr_residue_head.train(mode)
        self.dpr_boundary_head.train(mode)
        return self

    def forward_llps(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.llps_backbone(batch)

    def forward_phaseflow(self, phaseflow_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            velocity = self.phaseflow.forward_flow(
                input_ids=phaseflow_batch["input_ids"],
                attention_mask=phaseflow_batch["attention_mask"],
                phase_t=phaseflow_batch["phase_t"],
                phase_mask=phaseflow_batch["phase_mask"],
                time=phaseflow_batch["time"],
                seq_len=phaseflow_batch["seq_len"],
            )
        return {"velocity": velocity}

    def forward_dpr(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        seq_mask = batch["seq_mask"].bool()
        llps_layers, pf_layers = self.extract_frozen_dpr_layers(batch)
        llps_hidden = torch.nan_to_num(self.layer_mix[PHASEFLOW_LLPS_LAYER_KEY](llps_layers), nan=0.0, posinf=0.0, neginf=0.0)
        pf_hidden = torch.nan_to_num(self.layer_mix["phaseflow"](pf_layers), nan=0.0, posinf=0.0, neginf=0.0)
        x = self.dpr_stem(torch.cat([llps_hidden, pf_hidden], dim=-1))
        neighbor_mask = batch["neighbor_mask"].bool()
        if self.disable_protenix_edges_for_dpr and "neighbor_edge_type" in batch:
            neighbor_mask = neighbor_mask & (batch["neighbor_edge_type"].to(neighbor_mask.device) != PROTENIX_EDGE_TYPE)
        x = self.dpr_blocks(
            x=x,
            neighbors=batch["neighbors"].long(),
            edge_attr=batch["edge_attr"].float(),
            neighbor_mask=neighbor_mask,
            seq_mask=seq_mask,
        )
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        residue_logits = self.dpr_residue_head(x).squeeze(-1).masked_fill(~seq_mask, 0.0)
        boundary_logits = self.dpr_boundary_head(x).masked_fill(~seq_mask.unsqueeze(-1), 0.0)
        out = {
            "residue_logits": residue_logits,
            "residue_prob": torch.sigmoid(residue_logits.float()),
            "start_logits": boundary_logits[..., 0],
            "end_logits": boundary_logits[..., 1],
            "boundary_logits": boundary_logits,
            "dpr_hidden": x,
            "seq_mask": seq_mask,
        }
        if bool(batch.get("return_regions", False)):
            out["regions"] = self.decode_regions(residue_logits, boundary_logits, seq_mask)
        return out

    def extract_frozen_dpr_layers(self, batch: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return DPR hidden taps without touching trainable DPR modules."""

        cached_llps = batch.get(PHASEFLOW_LLPS_HIDDEN_LAYERS_KEY, batch.get(LEGACY_LLPS_HIDDEN_LAYERS_KEY))
        if torch.is_tensor(cached_llps):
            llps_layers = [cached_llps[:, layer_index].detach() for layer_index in range(int(cached_llps.shape[1]))]
        else:
            dpr_llps_batch = self._llps_batch_for_dpr(batch)
            with torch.no_grad():
                llps_outputs = self.llps_backbone(dpr_llps_batch)
                llps_layers = [llps_outputs[name].detach() for name in self.llps_hidden_names if name in llps_outputs]
                if not llps_layers:
                    raise KeyError(f"PhaseFlow outputs did not include any of {self.llps_hidden_names}")
                while len(llps_layers) < len(self.llps_hidden_names):
                    llps_layers.append(llps_layers[-1])

        cached_pf = batch.get("phaseflow_hidden_layers")
        if torch.is_tensor(cached_pf):
            pf_layers = [cached_pf[:, layer_index].detach() for layer_index in range(int(cached_pf.shape[1]))]
        else:
            with torch.no_grad():
                pf_layers = [layer.detach() for layer in self.phaseflow_residue_hidden(batch)]
        return llps_layers, pf_layers

    def forward(
        self,
        batch: dict[str, Any] | None = None,
        *,
        task: str,
        phaseflow_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if task == "llps":
            if batch is None:
                raise ValueError("batch is required for task='llps'")
            return {"llps": self.forward_llps(batch)}
        if task == "phaseflow":
            if phaseflow_batch is None:
                raise ValueError("phaseflow_batch is required for task='phaseflow'")
            return {"phaseflow": self.forward_phaseflow(phaseflow_batch)}
        if task == "dpr":
            if batch is None:
                raise ValueError("batch is required for task='dpr'")
            return {"dpr": self.forward_dpr(batch)}
        raise ValueError(f"Unknown task: {task}")

    def phaseflow_residue_hidden(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        tokenizer = _phaseflow_tokenizer()
        sequences = [str(seq) for seq in batch["sequences"]]
        lengths = batch["lengths"].detach().cpu().tolist()
        device = next(self.phaseflow.parameters()).device
        num_layers = int(self.layer_mix["phaseflow"].logits.numel())
        hidden_dim = int(getattr(self.phaseflow, "dim", self.dpr_config.get("phaseflow_hidden_dim", 256)))
        max_len = int(batch["seq_mask"].shape[1])
        accum = [torch.zeros(len(sequences), max_len, hidden_dim, device=device) for _ in range(num_layers)]
        counts = torch.zeros(len(sequences), max_len, 1, device=device)
        windows: list[tuple[int, int, str, int]] = []
        for batch_index, (sequence, length) in enumerate(zip(sequences, lengths, strict=False)):
            for start, end in _phaseflow_windows(int(length), self.phaseflow_window_size, self.phaseflow_stride):
                windows.append((batch_index, start, sequence[start:end], end - start))
                if self.phaseflow_max_windows_per_sequence > 0:
                    seen = sum(1 for item in windows if item[0] == batch_index)
                    if seen >= self.phaseflow_max_windows_per_sequence:
                        break
        if not windows:
            return accum
        for offset in range(0, len(windows), self.phaseflow_window_batch_size):
            chunk = windows[offset : offset + self.phaseflow_window_batch_size]
            encoded = []
            window_lengths = []
            for _, _, subseq, window_length in chunk:
                tokens = tokenizer.build_input_sequence(_phaseflow_safe_sequence(subseq), shape=self.phaseflow_shape)
                encoded.append(tokenizer.pad_sequence(tokens, int(self.phaseflow.max_seq_len)))
                window_lengths.append(window_length)
            input_ids = torch.tensor(encoded, dtype=torch.long, device=device)
            attention_mask = (input_ids != int(tokenizer.PAD_ID)).float()
            phase_t = torch.zeros(len(chunk), int(self.phaseflow.phase_dim), dtype=torch.float32, device=device)
            phase_mask = torch.ones_like(phase_t)
            time = torch.zeros(len(chunk), dtype=torch.float32, device=device)
            with torch.no_grad():
                layer_outputs = _phaseflow_forward_flow_layers(
                    self.phaseflow,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    phase_t=phase_t,
                    phase_mask=phase_mask,
                    time=time,
                )
            selected_layers = layer_outputs[-num_layers:]
            for local_index, (batch_index, start, _subseq, window_length) in enumerate(chunk):
                end = min(start + window_length, max_len)
                residue_slice = slice(1, 1 + (end - start))
                for layer_index, layer_hidden in enumerate(selected_layers):
                    accum[layer_index][batch_index, start:end] += layer_hidden[local_index, residue_slice].detach()
                counts[batch_index, start:end] += 1.0
        counts = counts.clamp_min(1.0)
        return [layer / counts for layer in accum]

    def _llps_batch_for_dpr(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not self.disable_protenix_for_dpr:
            return batch
        out = dict(batch)
        if "protenix_embed" in batch:
            out["protenix_embed"] = torch.zeros_like(batch["protenix_embed"])
        if "modality_mask" in batch:
            mask = batch["modality_mask"].clone()
            mask[..., 3] = 1.0
            out["modality_mask"] = mask
        if "reliability" in batch:
            reliability = batch["reliability"].clone()
            reliability[..., 3] = 0.0
            out["reliability"] = reliability
        return out

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    @staticmethod
    def decode_regions(
        residue_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        seq_mask: torch.Tensor,
        *,
        residue_threshold: float = 0.5,
        min_length: int = 3,
    ) -> list[list[dict[str, float | int]]]:
        residue_prob = torch.sigmoid(residue_logits.float()).detach().cpu()
        start_prob = torch.sigmoid(boundary_logits[..., 0].float()).detach().cpu()
        end_prob = torch.sigmoid(boundary_logits[..., 1].float()).detach().cpu()
        mask = seq_mask.detach().cpu().bool()
        all_regions: list[list[dict[str, float | int]]] = []
        for b in range(residue_prob.shape[0]):
            valid_len = int(mask[b].sum().item())
            regions: list[dict[str, float | int]] = []
            in_region = False
            start = 0
            for i in range(valid_len):
                active = bool(residue_prob[b, i].item() >= residue_threshold)
                if active and not in_region:
                    start = i
                    in_region = True
                if in_region and (not active or i == valid_len - 1):
                    end = i if active and i == valid_len - 1 else i - 1
                    if end - start + 1 >= min_length:
                        score = float(residue_prob[b, start : end + 1].mean().item())
                        regions.append(
                            {
                                "start": int(start),
                                "end": int(end),
                                "score": score,
                                "start_score": float(start_prob[b, start].item()),
                                "end_score": float(end_prob[b, end].item()),
                            }
                        )
                    in_region = False
            all_regions.append(regions)
        return all_regions


def load_phaseflow_llps_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[PhaseFlowModel, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = PhaseFlowModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


def load_phaseflow_checkpoint(
    checkpoint_path: str | Path,
    *,
    repo_path: str | Path = DEFAULT_PHASEFLOW_REPO,
    dependency_site_packages: str | Path = DEFAULT_PHASEFLOW_SITE_PACKAGES,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    _prepare_phaseflow_imports(repo_path=repo_path, dependency_site_packages=dependency_site_packages)
    from phaseflow.model import PhaseFlow  # type: ignore

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["config"]["model"])
    model = PhaseFlow(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


def _prepare_phaseflow_imports(*, repo_path: str | Path, dependency_site_packages: str | Path) -> None:
    dep = str(Path(dependency_site_packages))
    repo = str(Path(repo_path))
    if dep not in sys.path:
        sys.path.append(dep)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    if "ot" not in sys.modules:
        ot_module = types.ModuleType("ot")

        def _unavailable_emd(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("ot.emd is unavailable in the PhaseStackDPR inference environment")

        ot_module.emd = _unavailable_emd  # type: ignore[attr-defined]
        sys.modules["ot"] = ot_module


def _phaseflow_tokenizer() -> Any:
    from phaseflow.tokenizer import AminoAcidTokenizer  # type: ignore

    return AminoAcidTokenizer()


def _phaseflow_forward_flow_layers(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    phase_t: torch.Tensor,
    phase_mask: torch.Tensor,
    time: torch.Tensor,
) -> list[torch.Tensor]:
    token_emb = model.embed_tokens(input_ids)
    phase_emb, phase_attn_mask = model.embed_phase(phase_t, phase_mask, time)
    x = torch.cat([token_emb, phase_emb], dim=1)
    extended_mask = torch.cat([attention_mask, phase_attn_mask], dim=1)
    phase_start_idx = int(model.max_seq_len)
    layers: list[torch.Tensor] = []
    for layer in model.transformer.layers:
        x = layer(
            x,
            model.transformer.rotary,
            extended_mask,
            phase_start_idx,
            None,
            bool(model.use_set_encoder),
        )
        layers.append(x[:, : input_ids.shape[1], :])
    x = model.transformer.final_norm(x)
    layers.append(x[:, : input_ids.shape[1], :])
    return layers


def _phaseflow_windows(length: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    length = int(length)
    window_size = max(int(window_size), 1)
    stride = max(int(stride), 1)
    if length <= window_size:
        return [(0, length)]
    starts = list(range(0, max(length - window_size + 1, 1), stride))
    last_start = length - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, min(start + window_size, length)) for start in starts]


def _phaseflow_safe_sequence(sequence: str) -> str:
    replacements = {"X": "G", "B": "N", "Z": "Q", "U": "C", "O": "K"}
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    out = []
    for aa in str(sequence).upper():
        aa = replacements.get(aa, aa)
        out.append(aa if aa in allowed else "G")
    return "".join(out)


def _mlp_head(d_model: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, 128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, out_dim),
    )


def _default_dpr_config() -> dict[str, Any]:
    return {
        "d_model": 256,
        "num_graph_transformer_layers": 4,
        "num_heads": 8,
        "ffn_dim": 1024,
        "dropout": 0.10,
        "pre_layer_norm": True,
        "edge_dim": 32,
        "num_edge_types": 40,
        "relative_position_bins": 32,
        PHASEFLOW_LLPS_HIDDEN_DIM_KEY: 256,
        "phaseflow_hidden_dim": 256,
        "phaseflow_num_mixed_layers": 4,
        "phaseflow_window_size": 20,
        "phaseflow_stride": 1,
        "phaseflow_shape": "4x4",
        "disable_protenix_for_dpr": True,
        "disable_protenix_edges_for_dpr": True,
        "phaseflow_max_windows_per_sequence": 0,
        "phaseflow_window_batch_size": 128,
    }
