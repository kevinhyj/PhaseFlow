from __future__ import annotations

import copy

import torch
from torch import nn

from phaseflow.full_length.models.adapters import ModalityAdapters
from phaseflow.full_length.models.fusion import ConcatFusion, ReliabilityGatedFusion
from phaseflow.full_length.models.heads import (
    DPRSummaryFusionHead,
    DPRBranchAdapter,
    DPRLocalizationBranch,
    GatedDPRScanResidual,
    LLPSProteinHead,
    MultiScaleDPRHead,
    PhaseDiagramHead,
    ResidueHead,
)
from phaseflow.full_length.models.local_motif_encoder import LocalMotifEncoder
from phaseflow.full_length.models.region_decoder import RegionQueryDecoder
from phaseflow.full_length.models.sparse_graph_transformer import SparseGraphTransformer
from phaseflow.full_length.features.bio_vec import BIO_VEC_DIM, BIO_VEC_NAMES


class BioMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int] | tuple[int, ...], dropout: float) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(value) for value in hidden]
        layers: list[nn.Module] = [nn.LayerNorm(dims[0])]
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=False):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(float(dropout)), nn.LayerNorm(out_dim)])
        self.net = nn.Sequential(*layers)
        self.output_dim = dims[-1]

    def forward(self, bio_vec: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(bio_vec.float(), nan=0.0, posinf=10.0, neginf=-10.0))


class BioFusionResidualHead(nn.Module):
    def __init__(self, protein_dim: int, bio_dim: int, hidden_dim: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(protein_dim + bio_dim),
            nn.Linear(protein_dim + bio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, base_logits: torch.Tensor, protein_repr: torch.Tensor, bio_repr: torch.Tensor) -> torch.Tensor:
        residual = self.net(torch.cat([protein_repr, bio_repr], dim=-1)).squeeze(-1)
        return base_logits + self.residual_scale * residual


class ProteinAuxHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class PhaseFlowModel(nn.Module):
    modality_indices = {
        "plm": 0,
        "physchem": 1,
        "disorder": 2,
        "protenix_embed": 3,
        "protenix_embedding": 3,
        "protenix": 3,
        "star": 4,
        "starling": 4,
        "starling_embed": 4,
        "star_node": 4,
    }
    edge_type_indices = {
        "local": 0,
        "sequence": 0,
        "star": 2,
        "starling": 2,
        "physchem": 3,
        "candidate": 4,
        "candidate_segment": 4,
    }
    named_ablations = {
        "no_physchem": (("physchem",), ()),
        "no_disorder": (("disorder",), ()),
        "no_protenix": (("protenix",), ()),
        "no_starling": (("starling",), ("starling",)),
        "no_protenix_starling": (("protenix", "starling"), ("starling",)),
    }
    bio_vec_groups = {
        "plm": ("esm_mean", "esm_std"),
        "esm2": ("esm_mean", "esm_std"),
        "physchem": (
            "ncpr",
            "charge_kappa_proxy",
            "sticker_spacer_kappa_proxy",
            "frac_g",
            "frac_p",
            "frac_r",
            "frac_y",
            "frac_f",
            "frac_w",
            "frac_fyw",
            "rgg_density",
            "aromatic_cluster_density",
            "charge_blockiness",
            "hydropathy_mean",
            "rna_binding_proxy",
            "dna_binding_proxy",
            "ptm_density_proxy",
        ),
        "disorder": ("idr_fraction", "ordered_fraction", "prld_fraction", "low_complexity_fraction"),
        "protenix": ("protenix_available",),
        "protenix_embed": ("protenix_available",),
        "protenix_embedding": ("protenix_available",),
        "star": ("starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"),
        "starling": ("starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"),
        "starling_embed": ("starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"),
        "star_node": ("starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"),
    }

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config.get("model", config)
        self.model_type = str(model_config.get("model_type", "v2_region"))
        self.is_decoupled = self.model_type in {"v3", "v3_decoupled", "decoupled"}
        self.forward_mode = str(model_config.get("forward_mode", "full")).strip().lower()
        self.llps_only_forward = self.is_decoupled and self.forward_mode in {"llps_only", "protein_only"}
        self.ablation_name = str(model_config.get("ablation", {}).get("name", "full"))
        ablation_config = model_config.get("ablation", {})
        self.disabled_modality_indices = self._disabled_indices(
            ablation_config,
            key="disabled_modalities",
            lookup=self.modality_indices,
            named_index=0,
        )
        self.disabled_edge_types = self._disabled_indices(
            ablation_config,
            key="disabled_edge_types",
            lookup=self.edge_type_indices,
            named_index=1,
        )
        d_model = int(model_config.get("d_model", 128))
        dropout = float(model_config.get("dropout", 0.1))
        self.adapters = ModalityAdapters(model_config["input_dims"], d_model, dropout)
        if self.ablation_name == "concat_fusion":
            self.fusion = ConcatFusion(d_model=d_model, num_modalities=5, dropout=dropout)
        else:
            self.fusion = ReliabilityGatedFusion(d_model=d_model, num_modalities=5)
        if self.is_decoupled:
            self._init_decoupled_encoders(model_config, d_model, dropout)
        else:
            local_config = model_config.get("local_encoder", {})
            self.local_encoder = self._make_local_encoder(local_config, d_model, dropout)
            self.encoder, self.uses_graph = self._make_sequence_encoder(model_config, model_config, d_model, dropout)
        llps_head_config = model_config.get("llps_head", {})
        self.llps_head = LLPSProteinHead(
            d_model,
            dropout,
            use_dpr_pooling=bool(llps_head_config.get("use_dpr_pooling", not self.is_decoupled)),
        )
        bio_config = model_config.get("bio_mlp", {}) or {}
        self.bio_mlp_enabled = bool(bio_config.get("enabled", False))
        self.bio_vec_dim = int(bio_config.get("input_dim", BIO_VEC_DIM))
        self.disabled_bio_vec_indices = self._disabled_bio_vec_indices(ablation_config)
        self.bio_mlp = None
        self.bio_fusion_head = None
        self.driver_head = None
        self.client_head = None
        self.negtype_head = None
        if self.bio_mlp_enabled:
            bio_hidden = list(bio_config.get("hidden", [256, 256, 128]))
            bio_dropout = float(bio_config.get("dropout", dropout))
            self.bio_mlp = BioMLP(self.bio_vec_dim, bio_hidden, bio_dropout)
            fusion_dim = 3 * d_model + self.bio_mlp.output_dim
            aux_hidden = int(bio_config.get("aux_hidden", max(64, d_model // 2)))
            self.bio_fusion_head = BioFusionResidualHead(
                protein_dim=3 * d_model,
                bio_dim=self.bio_mlp.output_dim,
                hidden_dim=int(bio_config.get("fusion_hidden", d_model)),
                dropout=bio_dropout,
                residual_scale=float(bio_config.get("residual_scale", 1.0)),
            )
            self.driver_head = ProteinAuxHead(fusion_dim, aux_hidden, 1, bio_dropout)
            self.client_head = ProteinAuxHead(fusion_dim, aux_hidden, 1, bio_dropout)
            self.negtype_head = ProteinAuxHead(fusion_dim, aux_hidden, 2, bio_dropout)
        mil_config = model_config.get("region_mil_head", {})
        self.dpr_head = MultiScaleDPRHead(
            d_model=d_model,
            dropout=dropout,
            windows=list(mil_config.get("windows", [33, 129, 257])),
            topk_ratio=float(mil_config.get("topk_ratio", 0.05)),
            max_weight=float(mil_config.get("max_weight", 0.3)),
        )
        scan_config = model_config.get("dpr_scan_residual", {})
        self.dpr_scan_residual = (
            GatedDPRScanResidual(
                d_model=d_model,
                dropout=dropout,
                windows=list(scan_config.get("windows", [9, 17, 33, 65, 129])),
                residual_scale=float(scan_config.get("residual_scale", 0.5)),
            )
            if bool(scan_config.get("enabled", False))
            else None
        )
        adapter_config = model_config.get("dpr_adapter", {})
        self.dpr_adapter = (
            DPRBranchAdapter(
                d_model=d_model,
                dropout=dropout,
                bottleneck_dim=int(adapter_config.get("bottleneck_dim", max(16, d_model // 4))),
                kernel_size=int(adapter_config.get("kernel_size", 9)),
                residual_scale=float(adapter_config.get("residual_scale", 0.25)),
            )
            if bool(adapter_config.get("enabled", False))
            else None
        )
        reference_config = model_config.get("llps_reference_dpr_head", {})
        self.llps_reference_dpr_head = None
        if bool(reference_config.get("enabled", False)):
            self.llps_reference_dpr_head = copy.deepcopy(self.dpr_head)
        if self.llps_reference_dpr_head is not None:
            for parameter in self.llps_reference_dpr_head.parameters():
                parameter.requires_grad = False
        self.key_head = ResidueHead(d_model, dropout)
        self.final_llps_alpha = float(model_config.get("final_llps_alpha", 0.8))
        self.llps_logit_bias = float(model_config.get("llps_logit_bias", 0.0))
        self.llps_logit_temperature = max(float(model_config.get("llps_logit_temperature", 1.0)), 1.0e-6)
        summary_config = model_config.get("dpr_summary", {})
        self.dpr_summary_dim = 6
        self.dpr_summary_enabled = self.is_decoupled and bool(summary_config.get("enabled", True))
        self.dpr_summary_detach = bool(summary_config.get("detach", True))
        self.dpr_summary_threshold = float(summary_config.get("threshold", 0.5))
        self.dpr_summary_temperature = max(float(summary_config.get("temperature", 0.08)), 1.0e-6)
        self.dpr_summary_head = (
            DPRSummaryFusionHead(
                summary_dim=self.dpr_summary_dim,
                hidden_dim=int(summary_config.get("hidden_dim", max(16, d_model // 2))),
                dropout=dropout,
                residual_scale=float(summary_config.get("residual_scale", 0.5)),
            )
            if self.dpr_summary_enabled
            else None
        )
        phase_aux_config = model_config.get("phase_aux", {})
        self.phase_head = (
            PhaseDiagramHead(d_model, int(phase_aux_config.get("phase_dim", 16)), dropout)
            if bool(phase_aux_config.get("enabled", False))
            else None
        )
        region_config = model_config.get("region_decoder", {})
        self.region_decoder = RegionQueryDecoder(
            d_model=d_model,
            num_queries=int(region_config.get("num_queries", 16)),
            num_layers=int(region_config.get("num_layers", 1)),
            dropout=dropout,
        )
        independent_dpr_config = model_config.get("dpr_localization_branch", {}) or model_config.get(
            "independent_dpr_branch",
            {},
        )
        self.dpr_localization_detach_input = bool(independent_dpr_config.get("detach_input", True))
        self.dpr_localization_branch = (
            DPRLocalizationBranch(
                d_model=d_model,
                dropout=float(independent_dpr_config.get("dropout", dropout)),
                bottleneck_dim=int(independent_dpr_config.get("bottleneck_dim", max(16, d_model // 4))),
                kernel_size=int(independent_dpr_config.get("kernel_size", 9)),
                residual_scale=float(independent_dpr_config.get("residual_scale", 0.25)),
                presence_topk_ratio=float(independent_dpr_config.get("presence_topk_ratio", 0.05)),
                windows=list(independent_dpr_config.get("windows", [9, 17, 33, 64, 129, 257])),
                aux_feature_dim=int(independent_dpr_config.get("aux_feature_dim", 106)),
            )
            if bool(independent_dpr_config.get("enabled", False))
            else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.is_decoupled:
            return self._forward_decoupled(batch)
        seq_mask = batch["seq_mask"].bool()
        adapter_batch, modality_mask, reliability = self._prepare_modality_inputs(batch)
        modality_repr = self.adapters(adapter_batch)
        modality_mask, reliability = self._apply_ablation(modality_mask, reliability)
        x, weights = self.fusion(modality_repr, modality_mask, reliability)
        x = self.local_encoder(x, seq_mask)
        if self.uses_graph:
            neighbor_mask = self._apply_edge_ablation(batch["edge_attr"], batch["neighbor_mask"])
            x = self.encoder(
                x=x,
                neighbors=batch["neighbors"],
                edge_attr=batch["edge_attr"],
                neighbor_mask=neighbor_mask,
                seq_mask=seq_mask,
            )
        else:
            x = self.encoder(x, src_key_padding_mask=~seq_mask)
            x = x * seq_mask.unsqueeze(-1)
        dpr_x = self._apply_dpr_adapter(x, seq_mask)
        region = self.region_decoder(dpr_x, seq_mask)
        dpr = self.dpr_head(dpr_x, seq_mask)
        dpr = self._apply_scan_residual(dpr, dpr_x, seq_mask)
        dpr_logits = dpr["dpr_logits"]
        llps_reference = self.llps_reference_dpr_head(x, seq_mask) if self.llps_reference_dpr_head is not None else dpr
        llps_reference_logits = llps_reference["dpr_logits"]
        llps_logits = self.llps_head(x, seq_mask, dpr_logits=llps_reference_logits)
        llps_logits, aux_outputs = self._apply_bio_mlp(
            batch=batch,
            base_logits=llps_logits,
            protein_x=x,
            seq_mask=seq_mask,
            dpr_logits=llps_reference_logits,
        )
        final_llps_prob = (
            self.final_llps_alpha * torch.sigmoid(llps_logits.float())
            + (1.0 - self.final_llps_alpha) * torch.sigmoid(llps_reference["region_global_logits"].float())
        ).clamp(min=1.0e-6, max=1.0 - 1.0e-6)
        outputs = {
            "residue_repr": dpr_x,
            "llps_residue_repr": x,
            "dpr_residue_repr": dpr_x,
            "llps_logits": llps_logits,
            "final_llps_logits": self._calibrated_llps_logits(torch.logit(final_llps_prob, eps=1.0e-6)),
            "dpr_logits": dpr_logits,
            "residue_dpr_logits": dpr_logits,
            "key_logits": self.key_head(dpr_x),
            "modality_weights": weights,
            **aux_outputs,
            **dpr,
            **region,
        }
        outputs.update(self._independent_dpr_outputs(x, seq_mask, batch=batch))
        if self.phase_head is not None:
            outputs["phase_values"] = self.phase_head(dpr_x, seq_mask, dpr_logits=dpr_logits)
        return outputs

    def _forward_decoupled(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seq_mask = batch["seq_mask"].bool()
        adapter_batch, modality_mask, reliability = self._prepare_modality_inputs(batch)
        modality_repr = self.adapters(adapter_batch)
        modality_mask, reliability = self._apply_ablation(modality_mask, reliability)
        x, weights = self.fusion(modality_repr, modality_mask, reliability)
        shared = self.shared_local_encoder(x, seq_mask)
        if self.uses_graph:
            neighbor_mask = self._apply_edge_ablation(batch["edge_attr"], batch["neighbor_mask"])
            shared = self.shared_encoder(
                x=shared,
                neighbors=batch["neighbors"],
                edge_attr=batch["edge_attr"],
                neighbor_mask=neighbor_mask,
                seq_mask=seq_mask,
            )
        else:
            shared = self.shared_encoder(shared, src_key_padding_mask=~seq_mask)
            shared = shared * seq_mask.unsqueeze(-1)

        llps_x = self.llps_branch_norm(shared)
        llps_x = self.llps_local_encoder(llps_x, seq_mask)
        if self.uses_graph:
            neighbor_mask = self._apply_edge_ablation(batch["edge_attr"], batch["neighbor_mask"])
            llps_x = self.llps_encoder(
                x=llps_x,
                neighbors=batch["neighbors"],
                edge_attr=batch["edge_attr"],
                neighbor_mask=neighbor_mask,
                seq_mask=seq_mask,
            )
        else:
            llps_x = self.llps_encoder(llps_x, src_key_padding_mask=~seq_mask) * seq_mask.unsqueeze(-1)

        raw_llps_logits = self.llps_head(llps_x, seq_mask, dpr_logits=None)
        raw_llps_logits, aux_outputs = self._apply_bio_mlp(
            batch=batch,
            base_logits=raw_llps_logits,
            protein_x=llps_x,
            seq_mask=seq_mask,
            dpr_logits=None,
        )
        if self.llps_only_forward:
            final_llps_logits = self._calibrated_llps_logits(raw_llps_logits)
            return {
                "residue_repr": llps_x,
                "shared_residue_repr": shared,
                "llps_residue_repr": llps_x,
                "llps_logits": raw_llps_logits,
                "raw_llps_logits": raw_llps_logits,
                "final_llps_logits": final_llps_logits,
                "loss_llps_logits": final_llps_logits,
                "modality_weights": weights,
                **aux_outputs,
            }

        dpr_x = self.dpr_branch_norm(shared)
        dpr_x = self.dpr_local_encoder(dpr_x, seq_mask)
        if self.uses_graph:
            neighbor_mask = self._apply_edge_ablation(batch["edge_attr"], batch["neighbor_mask"])
            dpr_x = self.dpr_encoder(
                x=dpr_x,
                neighbors=batch["neighbors"],
                edge_attr=batch["edge_attr"],
                neighbor_mask=neighbor_mask,
                seq_mask=seq_mask,
            )
        else:
            dpr_x = self.dpr_encoder(dpr_x, src_key_padding_mask=~seq_mask) * seq_mask.unsqueeze(-1)

        dpr_x = self._apply_dpr_adapter(dpr_x, seq_mask)
        region = self.region_decoder(dpr_x, seq_mask)
        dpr = self.dpr_head(dpr_x, seq_mask)
        dpr = self._apply_scan_residual(dpr, dpr_x, seq_mask)
        dpr_logits = dpr["dpr_logits"]
        dpr_summary = self._dpr_summary_features(dpr, dpr_logits, seq_mask)
        if self.dpr_summary_head is not None:
            summary_for_final = dpr_summary.detach() if self.dpr_summary_detach else dpr_summary
            final_llps_logits = self.dpr_summary_head(raw_llps_logits, summary_for_final)
        else:
            final_llps_logits = raw_llps_logits
        final_llps_logits = self._calibrated_llps_logits(final_llps_logits)
        outputs = {
            "residue_repr": dpr_x,
            "shared_residue_repr": shared,
            "llps_residue_repr": llps_x,
            "dpr_residue_repr": dpr_x,
            "llps_logits": raw_llps_logits,
            "raw_llps_logits": raw_llps_logits,
            "final_llps_logits": final_llps_logits,
            "loss_llps_logits": final_llps_logits,
            "dpr_logits": dpr_logits,
            "residue_dpr_logits": dpr_logits,
            "key_logits": self.key_head(dpr_x),
            "dpr_summary_features": dpr_summary,
            "dpr_summary_detached": torch.full_like(raw_llps_logits, float(self.dpr_summary_detach)),
            "modality_weights": weights,
            **aux_outputs,
            **dpr,
            **region,
        }
        outputs.update(self._independent_dpr_outputs(shared, seq_mask, batch=batch))
        if self.phase_head is not None:
            outputs["phase_values"] = self.phase_head(dpr_x, seq_mask, dpr_logits=dpr_logits)
        return outputs

    def _calibrated_llps_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.llps_logit_bias == 0.0 and self.llps_logit_temperature == 1.0:
            return logits
        return logits / float(self.llps_logit_temperature) + float(self.llps_logit_bias)

    def _apply_bio_mlp(
        self,
        *,
        batch: dict[str, torch.Tensor],
        base_logits: torch.Tensor,
        protein_x: torch.Tensor,
        seq_mask: torch.Tensor,
        dpr_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.bio_mlp_enabled or self.bio_mlp is None or self.bio_fusion_head is None:
            return base_logits, {}
        bio_vec = batch.get("bio_vec")
        if bio_vec is None:
            bio_vec = torch.zeros(base_logits.shape[0], self.bio_vec_dim, dtype=protein_x.dtype, device=protein_x.device)
        bio_vec = bio_vec.to(device=protein_x.device, dtype=protein_x.dtype)
        if bio_vec.shape[-1] != self.bio_vec_dim:
            if bio_vec.shape[-1] > self.bio_vec_dim:
                bio_vec = bio_vec[:, : self.bio_vec_dim]
            else:
                pad = torch.zeros(
                    bio_vec.shape[0],
                    self.bio_vec_dim - bio_vec.shape[-1],
                    dtype=bio_vec.dtype,
                    device=bio_vec.device,
                )
                bio_vec = torch.cat([bio_vec, pad], dim=-1)
        if self.disabled_bio_vec_indices:
            bio_vec = bio_vec.clone()
            bio_vec[:, list(self.disabled_bio_vec_indices)] = 0.0
        bio_repr = self.bio_mlp(bio_vec)
        protein_repr = self._llps_protein_repr(protein_x, seq_mask, dpr_logits)
        final_logits = self.bio_fusion_head(base_logits, protein_repr, bio_repr)
        aux_input = torch.cat([protein_repr, bio_repr], dim=-1)
        aux: dict[str, torch.Tensor] = {
            "bio_vec": bio_vec,
            "bio_repr": bio_repr,
            "bio_llps_logits": final_logits,
        }
        if self.driver_head is not None:
            aux["driver_logits"] = self.driver_head(aux_input)
        if self.client_head is not None:
            aux["client_logits"] = self.client_head(aux_input)
        if self.negtype_head is not None:
            aux["negtype_logits"] = self.negtype_head(aux_input)
        return final_logits, aux

    def _llps_protein_repr(
        self,
        x: torch.Tensor,
        seq_mask: torch.Tensor,
        dpr_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = self.llps_head.pool(x).squeeze(-1).masked_fill(~seq_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        attention_pool = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        mean_pool = torch.sum(x * seq_mask.unsqueeze(-1), dim=1) / seq_mask.sum(dim=1, keepdim=True).clamp(min=1)
        if dpr_logits is None:
            dpr_scores = self.llps_head.dpr_pool(x).squeeze(-1)
        else:
            dpr_scores = dpr_logits
        dpr_weights = torch.softmax(dpr_scores.masked_fill(~seq_mask, -1.0e4), dim=-1)
        high_dpr_pool = torch.sum(dpr_weights.unsqueeze(-1) * x, dim=1)
        return torch.cat([attention_pool, mean_pool, high_dpr_pool], dim=-1)

    def _apply_dpr_adapter(self, x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        if self.dpr_adapter is None:
            return x
        return self.dpr_adapter(x, seq_mask)

    def _apply_scan_residual(
        self,
        dpr: dict[str, torch.Tensor],
        x: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.dpr_scan_residual is None:
            return dpr
        scan = self.dpr_scan_residual(x, seq_mask)
        base_logits = dpr["dpr_logits"]
        dpr_logits = (base_logits + scan["dpr_scan_residual_logits"]).masked_fill(~seq_mask, -1.0e4)
        global_scores = self._dpr_global_scores(
            dpr_logits,
            seq_mask,
            topk_ratio=float(getattr(self.dpr_head, "topk_ratio", 0.05)),
            max_weight=float(getattr(self.dpr_head, "max_weight", 0.3)),
        )
        return {
            **dpr,
            "base_dpr_logits": base_logits,
            "dpr_logits": dpr_logits,
            "region_global_logits": global_scores["region_global_logits"],
            "region_global_score": global_scores["region_global_score"],
            "region_topk_score": global_scores["region_topk_score"],
            "region_max_score": global_scores["region_max_score"],
            **scan,
        }

    def _independent_dpr_outputs(
        self,
        x: torch.Tensor,
        seq_mask: torch.Tensor,
        *,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.dpr_localization_branch is None:
            return {}
        dpr_input = x.detach() if self.dpr_localization_detach_input else x
        return self.dpr_localization_branch(dpr_input, seq_mask, batch=batch)

    @staticmethod
    def _dpr_global_scores(
        dpr_logits: torch.Tensor,
        seq_mask: torch.Tensor,
        *,
        topk_ratio: float,
        max_weight: float,
    ) -> dict[str, torch.Tensor]:
        probs = torch.sigmoid(dpr_logits.float()).masked_fill(~seq_mask, 0.0)
        topk_values = []
        max_values = []
        for index in range(probs.shape[0]):
            length = int(seq_mask[index].sum().item())
            if length == 0:
                topk_values.append(probs[index].sum() * 0.0)
                max_values.append(probs[index].sum() * 0.0)
                continue
            k = max(1, int(round(length * topk_ratio)))
            values = probs[index, :length]
            topk_values.append(torch.topk(values, k=min(k, length)).values.mean())
            max_values.append(values.max())
        topk_mean = torch.stack(topk_values)
        max_score = torch.stack(max_values)
        region_global_score = ((1.0 - max_weight) * topk_mean + max_weight * max_score).float().clamp(
            min=1.0e-6,
            max=1.0 - 1.0e-6,
        )
        return {
            "region_global_logits": torch.logit(region_global_score, eps=1.0e-6),
            "region_global_score": region_global_score,
            "region_topk_score": topk_mean,
            "region_max_score": max_score,
        }

    def _init_decoupled_encoders(self, model_config: dict, d_model: int, dropout: float) -> None:
        local_config = dict(model_config.get("local_encoder", {}))
        graph_config = dict(model_config.get("graph_transformer", {}))
        decoupled_config = dict(model_config.get("decoupled", {}))
        total_local_layers = int(local_config.get("num_layers", 2))
        total_graph_layers = int(graph_config.get("num_layers", 2))

        shared_local_layers = int(decoupled_config.get("shared_local_layers", min(max(total_local_layers, 0), 1)))
        branch_local_layers = int(
            decoupled_config.get("branch_local_layers", max(total_local_layers - shared_local_layers, 1))
        )
        shared_graph_layers = int(decoupled_config.get("shared_graph_layers", max(total_graph_layers // 2, 1)))
        branch_graph_layers = int(
            decoupled_config.get("branch_graph_layers", max(total_graph_layers - shared_graph_layers, 1))
        )

        self.shared_local_encoder = self._make_local_encoder(
            _merged_encoder_config(local_config, model_config.get("shared_local_encoder", {}), shared_local_layers),
            d_model,
            dropout,
        )
        self.llps_local_encoder = self._make_local_encoder(
            _merged_encoder_config(local_config, model_config.get("llps_local_encoder", {}), branch_local_layers),
            d_model,
            dropout,
        )
        self.dpr_local_encoder = self._make_local_encoder(
            _merged_encoder_config(local_config, model_config.get("dpr_local_encoder", {}), branch_local_layers),
            d_model,
            dropout,
        )

        shared_sequence_config = dict(model_config)
        shared_sequence_config["graph_transformer"] = _merged_encoder_config(
            graph_config,
            model_config.get("shared_graph_transformer", {}),
            shared_graph_layers,
        )
        self.shared_encoder, self.uses_graph = self._make_sequence_encoder(
            model_config,
            shared_sequence_config,
            d_model,
            dropout,
        )

        llps_sequence_config = dict(model_config)
        llps_sequence_config["graph_transformer"] = _merged_encoder_config(
            graph_config,
            model_config.get("llps_graph_transformer", {}),
            branch_graph_layers,
        )
        dpr_sequence_config = dict(model_config)
        dpr_sequence_config["graph_transformer"] = _merged_encoder_config(
            graph_config,
            model_config.get("dpr_graph_transformer", {}),
            branch_graph_layers,
        )
        self.llps_encoder, llps_uses_graph = self._make_sequence_encoder(
            model_config,
            llps_sequence_config,
            d_model,
            dropout,
        )
        self.dpr_encoder, dpr_uses_graph = self._make_sequence_encoder(
            model_config,
            dpr_sequence_config,
            d_model,
            dropout,
        )
        if llps_uses_graph != self.uses_graph or dpr_uses_graph != self.uses_graph:
            raise ValueError("Decoupled encoders must all use the same graph/non-graph mode.")
        self.llps_branch_norm = nn.LayerNorm(d_model)
        self.dpr_branch_norm = nn.LayerNorm(d_model)

    def _make_local_encoder(self, local_config: dict, d_model: int, dropout: float) -> LocalMotifEncoder:
        return LocalMotifEncoder(
            d_model=d_model,
            num_layers=int(local_config.get("num_layers", 2)),
            kernels=list(local_config.get("kernels", [3, 5, 9])),
            dilations=list(local_config.get("dilations", [2, 4])),
            dropout=dropout,
        )

    def _make_sequence_encoder(
        self,
        root_model_config: dict,
        encoder_model_config: dict,
        d_model: int,
        dropout: float,
    ) -> tuple[nn.Module, bool]:
        if self.model_type == "v0":
            trans_config = encoder_model_config.get("transformer", root_model_config.get("transformer", {}))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(trans_config.get("num_heads", 4)),
                dim_feedforward=int(trans_config.get("ffn_dim", 4 * d_model)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=int(trans_config.get("num_layers", 2))), False

        graph_config = encoder_model_config.get("graph_transformer", root_model_config.get("graph_transformer", {}))
        return (
            SparseGraphTransformer(
                d_model=d_model,
                num_layers=int(graph_config.get("num_layers", 2)),
                num_heads=int(graph_config.get("num_heads", 4)),
                edge_dim=int(graph_config.get("edge_dim", 8)),
                ffn_dim=int(graph_config.get("ffn_dim", 4 * d_model)),
                dropout=dropout,
                num_edge_types=int(graph_config.get("num_edge_types", 8)),
                relative_position_bins=int(graph_config.get("relative_position_bins", 32)),
            ),
            True,
        )

    def _dpr_summary_features(
        self,
        dpr: dict[str, torch.Tensor],
        dpr_logits: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(dpr_logits).masked_fill(~seq_mask, 0.0)
        lengths = seq_mask.float().sum(dim=1).clamp(min=1.0)
        mean_score = probs.sum(dim=1) / lengths
        high_fraction = torch.sigmoid((probs - self.dpr_summary_threshold) / self.dpr_summary_temperature)
        high_fraction = high_fraction.masked_fill(~seq_mask, 0.0).sum(dim=1) / lengths
        uncertainty = (probs * (1.0 - probs)).sum(dim=1) / lengths
        return torch.stack(
            [
                dpr["region_global_score"],
                dpr["region_topk_score"],
                dpr["region_max_score"],
                mean_score,
                high_fraction,
                uncertainty,
            ],
            dim=-1,
        )

    def _prepare_modality_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        modality_mask = batch["modality_mask"]
        reliability = batch["reliability"]
        esm2_available_mask = batch.get("esm2_available_mask")
        if esm2_available_mask is None:
            return batch, modality_mask, reliability

        esm2_available = esm2_available_mask.to(dtype=batch["plm"].dtype).clamp(min=0.0, max=1.0)
        adapter_batch = dict(batch)
        adapter_batch["plm"] = batch["plm"] * esm2_available.unsqueeze(-1)

        modality_mask = modality_mask.clone()
        reliability = reliability.clone()
        plm_missing = 1.0 - esm2_available.to(dtype=modality_mask.dtype)
        modality_mask[..., self.modality_indices["plm"]] = torch.maximum(
            modality_mask[..., self.modality_indices["plm"]],
            plm_missing,
        )
        reliability[..., self.modality_indices["plm"]] = (
            reliability[..., self.modality_indices["plm"]] * esm2_available.to(dtype=reliability.dtype)
        )
        return adapter_batch, modality_mask, reliability

    def _apply_ablation(self, modality_mask: torch.Tensor, reliability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.disabled_modality_indices:
            return modality_mask, reliability
        modality_mask = modality_mask.clone()
        reliability = reliability.clone()
        for modality_index in self.disabled_modality_indices:
            modality_mask[..., modality_index] = 1.0
            reliability[..., modality_index] = 0.0
        return modality_mask, reliability

    def _apply_edge_ablation(self, edge_attr: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        if not self.disabled_edge_types or edge_attr.shape[-1] <= 3:
            return neighbor_mask
        edge_types = _edge_type_from_attr(edge_attr)
        disabled = torch.zeros_like(neighbor_mask, dtype=torch.bool)
        for edge_type in self.disabled_edge_types:
            disabled |= edge_types == edge_type
        return neighbor_mask & ~disabled

    def _disabled_indices(
        self,
        ablation_config: dict,
        key: str,
        lookup: dict[str, int],
        named_index: int,
    ) -> tuple[int, ...]:
        names: list[str] = []
        if self.ablation_name in self.named_ablations:
            names.extend(self.named_ablations[self.ablation_name][named_index])
        raw = ablation_config.get(key, ())
        if isinstance(raw, str):
            raw = [raw]
        names.extend(str(name) for name in raw)
        indices = []
        for name in names:
            normalized = name.strip().lower()
            if not normalized:
                continue
            if normalized not in lookup:
                raise ValueError(f"Unknown ablation {key} entry: {name}")
            value = lookup[normalized]
            if isinstance(value, tuple):
                indices.extend(value)
            else:
                indices.append(value)
        return tuple(sorted(set(indices)))

    def _disabled_bio_vec_indices(self, ablation_config: dict) -> tuple[int, ...]:
        if not bool(ablation_config.get("zero_disabled_bio_vec", True)):
            return ()
        names: list[str] = []
        if self.ablation_name in self.named_ablations:
            names.extend(self.named_ablations[self.ablation_name][0])
        raw = ablation_config.get("disabled_modalities", ())
        if isinstance(raw, str):
            raw = [raw]
        names.extend(str(name) for name in raw)
        raw_groups = ablation_config.get("disabled_bio_vec_groups", ())
        if isinstance(raw_groups, str):
            raw_groups = [raw_groups]
        names.extend(str(name) for name in raw_groups)
        raw_features = ablation_config.get("disabled_bio_vec_features", ())
        if isinstance(raw_features, str):
            raw_features = [raw_features]
        feature_names = [str(name).strip() for name in raw_features if str(name).strip()]
        for name in names:
            normalized = str(name).strip().lower()
            if normalized in self.bio_vec_groups:
                feature_names.extend(self.bio_vec_groups[normalized])
        indices: list[int] = []
        for feature_name in feature_names:
            if feature_name in BIO_VEC_NAMES:
                indices.append(BIO_VEC_NAMES.index(feature_name))
        return tuple(sorted(set(index for index in indices if index < self.bio_vec_dim)))


def _edge_type_from_attr(edge_attr: torch.Tensor) -> torch.Tensor:
    type_slice = edge_attr[..., 3:11]
    if type_slice.numel() == 0:
        return torch.zeros(edge_attr.shape[:-1], dtype=torch.long, device=edge_attr.device)
    return torch.argmax(type_slice, dim=-1)


def _merged_encoder_config(base: dict, override: dict | None, num_layers: int) -> dict:
    config = dict(base)
    config.update(dict(override or {}))
    config["num_layers"] = int(config.get("num_layers", num_layers))
    if override is None or "num_layers" not in override:
        config["num_layers"] = int(num_layers)
    return config
