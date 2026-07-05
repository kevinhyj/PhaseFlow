# PhaseFlow full-length architecture schematic 与 Methods 一致性核对

对应图文件：

`artifacts/model/structure/structure_ppt.svg`

目标：核对 Supplementary Figure 中 `Short-Peptide Input -> PhaseFlow Peptide Module -> Bridge and Remapping -> DPR Scanner` 以及 `Full Length Protein -> Full-length LLPS Module -> LLPS / DPR outputs` 的技术表述是否与实现、配置和现有审计文件一致。本文只整理可验证事实；未在代码或配置中定位到的细节写为 `implementation not found / uncertain`。

## Evidence Sources

主要证据来自以下文件：

- `external/phaseflow-peptide/phaseflow/tokenizer.py`
- `external/phaseflow-peptide/phaseflow/data.py`
- `external/phaseflow-peptide/phaseflow/model.py`
- `external/phaseflow-peptide/phaseflow/transformer.py`
- `external/phaseflow-peptide/phaseflow/utils.py`
- `external/phaseflow-peptide/train/train.py`
- `external/phaseflow-peptide/outputs_set/output_set_flow32_missing15/config.yaml`
- `./phaseflow/models/adapters.py`
- `./phaseflow/models/fusion.py`
- `./phaseflow/models/local_motif_encoder.py`
- `./phaseflow/models/sparse_graph_transformer.py`
- `./phaseflow/models/heads.py`
- `./phaseflow/models/phaseflow.py`
- `./src/models/phasestack_v2_direct.py`
- `./phaseflow/models/dpr_v6.py`
- `./phaseflow/features/bio_vec.py`
- `paper/full_length/audit/final_llps_config.yaml`
- `paper/full_length/audit/final_dpr_config.yaml`

## A. Peptide Module

### A1. Short-peptide input length 与 token length

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| 5-20 aa 与 `<=32 tokens` 的关系 | `Short-Peptide Input`; `Peptide tokens (<=32)` | 输入原始短肽序列为 5-20 aa；tokenized sequence padded/truncated to `max_seq_len=32`; `input_ids: B x 32`; `attention_mask: B x 32` | Stage 1 peptide training: trainable; full-length DPR reuse path: frozen | `tokens = [SOS] + AA_tokens + [EOS] + [META] + shape_tokens + [SOM]`; `pad_sequence(tokens, 32)` | "Peptide-only training used 5-20 aa peptides encoded into sequences of at most 32 tokens after special and shape tokens." | verified |
| 是否所有长度都直接进入 peptide module | `Short-Peptide Input -> PhaseFlow Peptide Module` | peptide-only path accepts short peptide token sequence; full-length path does not feed raw length-`L` sequence into tokenizer | Stage 1 trainable; full-length reuse frozen | peptide-only: `build_input_sequence(short_seq)`；full-length: `z_i -> 32 bridge tokens -> frozen transformer` | "Short peptides enter the peptide module directly; full-length proteins reuse it indirectly through learned 32-token bridge summaries." | verified |

解释：`5-20 aa` 是原始短肽长度范围；`<=32 tokens` 是加上 `SOS/EOS/META/shape/SOM` 后的 token 序列上限。图中同时写这两者是可以的，但 caption 需要明确 `<=32 tokens` 不是 `<=32 aa`。

### A2. Special tokens and sequence construction

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| SOS/EOS/META/SOM/EOM/shape tokens | `Peptide tokens (<=32)` | vocabulary size 32: 20 AA + 6 special + 6 shape IDs；`input_ids: B x 32` | Stage 1 trainable; full-length frozen reuse | `PAD=20, SOS=21, EOS=22, META=23, SOM=24, EOM=25, shape_offset=26`; `shape="4x4" -> [30,30]` | "The tokenizer defines PAD/SOS/EOS/META/SOM/EOM and shape tokens; the implemented batch input sequence contains SOS, EOS, META, shape tokens and SOM." | verified with caveat |
| EOM 是否实际进入 batch input | `Peptide tokens (<=32)` | `build_input_sequence` returns tokens ending at `SOM`; no appended `EOM` on this path | same as above | `return [SOS] + AA + [EOS] + [META] + shape + [SOM]` | "Do not state that EOM is always appended to the peptide input unless another decoding path is explicitly documented." | implementation not found / uncertain for EOM use beyond tokenizer definition |

解释：tokenizer 定义了 `EOM_ID`，但 `build_input_sequence` 的实际输入路径没有追加 `EOM`。Methods 中不应笼统写成所有 special tokens 都一定进入输入流。

### A3. 16 phase slots and PSSI grid

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| `16 phase slots` 是否对应 4x4 PSSI grid | `Peptide-token hidden context (16 phase slots)` | phase target `phase_values: B x 16`; phase mask `phase_mask: B x 16`; phase tokens `B x 16 x 256` | Stage 1 trainable; full-length reuse frozen | CSV columns `group_11..group_44`; `phase_dim=16`; `SetPhaseEncoder(... phase_dim=16)` | "The 16 phase slots correspond to the 4x4 PSSI grid positions, encoded as 16 set tokens." | verified |
| phase tokens 如何编码 | `Peptide-token hidden context (16 phase slots)` | scalar PSSI value per slot -> 256-d token；output `phase_emb: B x 16 x 256` | Stage 1 trainable; frozen when reused in DPR bridge | `val = MLP(SinusoidalPosEmb(pssi)); token_i = val_i + pos_emb[i] + time_emb`; invalid tokens multiplied by `phase_mask` | "Each PSSI value is embedded by sinusoidal scalar encoding plus an MLP and an independent learned position embedding." | verified |
| missing phase-grid values 如何 mask | same | missing values stored as NaN, converted to 0 placeholder; mask `1=valid, 0=missing` | same | `phase_mask = ~isnan(data)`; `phase_data = nan_to_num(...,0)`; `tokens *= phase_mask`; loss uses `diff * phase_mask` | "Missing PSSI entries are zero-filled only as placeholders and are excluded by phase masks in attention and loss." | verified |

解释：`16 phase slots` 是 4x4 grid 的 16 个位置，但实现中是 16 个 independent learned position embeddings；没有定位到 row/column decomposition 或 2D convolutional grid encoder。

### A4. Peptide Transformer architecture

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| backbone 是 Transformer 还是 diffusion blocks | `Peptide Transformer`; `Frozen Peptide Transformer` | token embeddings `B x 32 x 256`; phase tokens `B x 16 x 256`; hidden dim 256 | Stage 1 trainable; full-length DPR reuse frozen | 6 Transformer blocks, 8 heads, dim_head 32, RMSNorm, RoPE on sequence tokens, SwiGLU FFN | "The peptide module uses a 6-layer Transformer backbone; flow matching is the training/generation objective, not a separate diffusion-block backbone." | verified |
| architecture details | `Peptide Transformer` | flow layout `B x (32+16) x 256`; LM layout `B x (16+32) x 256`; logits `B x 32 x 32`; velocity `B x 16` | same | attention q/k/v/out `256 -> 256`; FFN `256 -> 1024 -> 256` with SwiGLU; final RMSNorm | "Peptide Transformer uses hidden size 256, 6 layers, 8 heads, RoPE, RMSNorm and SwiGLU FFNs." | verified |

解释：图中应保留 `Peptide Transformer`，不要改成 `diffusion blocks`。Flow Matching 对应的是 `sequence -> phase` 的 velocity field training and sampling；language modeling 对应 `phase -> sequence` 的 autoregressive readout。

### A5. Phase diagram / peptide design readout

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| Phase diagram readout | `Phase Diagram / Peptide design readout` | input `input_ids: B x 32`; output generated phase `B x 16` | Stage 1 trainable for training; frozen when full-length DPR bridge calls transformer | `x_0 ~ N(0,I)`; ODE `dx/dt = v_theta(seq, x_t, t)` from `t=0` to `1` | "Phase diagrams are generated by integrating the learned flow-matching velocity field to produce 16 PSSI values." | verified |
| Peptide design readout | same | input phase `B x 16`; output sampled token sequence up to `max_len` | Stage 1 trainable; not used in full-length DPR scanner | start `SOS`; repeat `logits=forward_lm(tokens, phase)`; sample next token until EOS/META/SOM/PAD | "Peptide design uses the phase-conditioned language-model direction of the same Transformer." | verified |
| head names | same | `velocity_per_pos: 256 -> 64 -> 1`; `lm_head: 256 -> vocab_size(32)` | Stage 1 trainable; frozen in full-length reuse | `velocity_per_pos(phase_hidden).squeeze(-1)`; `lm_head(token_hidden)` | "Use implementation names `velocity_per_pos` and `lm_head`; `Phase Diagram / Peptide design readout` is a conceptual figure label." | verified |

解释：图中的 readout 标签不是一个单一 head 名称。它概括了两个方向：`generate_phase` 的 flow integration 和 `generate_sequence` 的 phase-conditioned LM decoding。

## B. Full-length LLPS Module

### B1. Residue-aligned input streams

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| ESM2 input | `ESM2 (L x 1280)` | `plm: B x L x 1280`; adapter output `B x L x 256` | offline/frozen representation; adapter frozen in reported LLPS checkpoint | `A_plm(plm_i) -> 256` | "Residue-aligned ESM2 embeddings of dimension 1280 are projected to the 256-dimensional full-length encoder space." | verified; exact ESM2 variant implementation not found / uncertain |
| PhysChem input | `PhysChem (L x 90)` | `physchem: B x L x 90`; adapter output `B x L x 256` | adapter frozen in reported LLPS checkpoint | `A_physchem(physchem_i) -> 256` | "Physicochemical residue features are a 90-dimensional stream in the LLPS encoder." | verified; per-feature names implementation not found / uncertain |
| Disorder input | `Disorder (L x 6)` | `disorder: B x L x 6`; adapter output `B x L x 256` | adapter frozen in reported LLPS checkpoint | `A_disorder(disorder_i) -> 256` | "Disorder features are a 6-dimensional residue stream in the LLPS encoder." | verified; per-feature names implementation not found / uncertain |
| Protenix input | `Protenix (L x 512)` | `protenix_embed: B x L x 512`; adapter output `B x L x 256` | adapter frozen in reported LLPS checkpoint | `A_protenix(protenix_i) -> 256` | "Protenix embeddings enter the full-length encoder through the modality adapter and are not directly concatenated in the final DPR scanner." | verified |
| STARLING stream | not shown in current figure | implementation supports `starling_embed: L x 512`, but reported checkpoint disables/masks it | disabled in reported LLPS checkpoint | `disabled_modalities: [starling_embed]`; reliability set to 0 and modality mask set missing | "Do not draw STARLING as an active main stream for the reported no-starling LLPS checkpoint." | verified |

解释：当前图只画 ESM2, PhysChem, Disorder, Protenix 是合理的。注意 DPR scanner 右侧不应再把 Protenix 作为 direct input；Protenix 只通过 full-length encoder state `z_i` 间接影响 DPR。

### B2. Raw BioPhys 112 and split

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| raw BioPhys 112 | `BioPhys` in DPR scanner; PhysChem/Disorder in LLPS module | raw `biophys: B x L x 112` exists for DPR; LLPS adapter streams use `physchem: B x L x 90` and `disorder: B x L x 6` | raw DPR input frozen/offline; LLPS adapters frozen | DPR: concat full 112; LLPS: separate adapter streams for 90 and 6 | "The raw BioPhys tensor is 112-dimensional in DPR, whereas the LLPS encoder consumes 90-dimensional PhysChem and 6-dimensional Disorder streams." | verified |
| remaining 16 features from raw 112 | not explicitly shown | `112 - 90 - 6 = 16`; no explicit LLPS adapter stream for remaining 16 found | not applicable | no located LLPS code path consuming the remaining 16 as a residue adapter stream | "If discussed, state that the remaining raw BioPhys dimensions are used in the DPR raw BioPhys stream; their exact feature identities were not recovered here." | implementation not found / uncertain |

解释：图里左侧 LLPS 写 PhysChem 90 和 Disorder 6，右侧 DPR 写 BioPhys 112，这是可以同时成立的。Methods 需要避免让读者以为 `90+6=112`；剩余 16 维的定义没有在本次核对中定位到。

### B3. Modality adapters and gated fusion

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| modality adapters | `Adapters each -> 256` | each stream `B x L x d_m` -> `B x L x 256`; stacked `B x L x 5 x 256` | frozen in reported LLPS checkpoint | `Linear(d_m,256) -> LayerNorm -> GELU -> Dropout(0.10) -> Linear(256,256) -> LayerNorm`; add learned modality embedding | "Each modality is independently projected into a 256-dimensional residue representation before fusion." | verified |
| gated fusion formula | `Gated Fusion (L x 256)` | inputs `modality_repr: B x L x M x 256`, `reliability: B x L x M`, `modality_mask: B x L x M`; output `B x L x 256` | frozen in reported LLPS checkpoint | `gate_input=[h_m; r_m; mask_m]` gives 258 dims; `logit_m=Linear(128,1)(GELU(Linear(258,128)(gate_input)))`; missing logits `=-1e4`; `w=softmax(logits)`; `h=sum_m w_m h_m` | "The gating network uses a 258-dimensional input composed of the 256-dimensional adapted feature plus reliability and missingness scalars." | verified |
| missing-modality masking | same | missing modality logits masked before softmax | frozen | `logits = logits.masked_fill(modality_mask.bool(), -1e4)` | "Missing modality logits are masked before the fusion softmax." | verified |

解释：用户提出的句子 `The gating network is Linear(258,128)->GELU->Linear(128,1), and missing-modality logits are masked before the softmax.` 与实现一致。

### B4. Local motif blocks

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| local motif block structure | `Local motif Blocks kernels 3/5/9 dilation 2/4` | input/output `B x L x 256`; 3 layers in reported config | all local encoder layers frozen in reported LLPS checkpoint | per block: parallel Conv1d kernels 3/5/9 plus kernel-7 dilated Conv1d at dilations 2 and 4; concat -> `1x1 Conv to 512 -> GLU -> Dropout -> 1x1 Conv to 256`; residual + LayerNorm | "Local motif blocks use parallel 1D convolution branches with kernels 3/5/9 and kernel-7 dilated branches at dilations 2 and 4." | verified |

解释：图中如果只写 `kernels 3/5/9 dilation 2/4` 略有歧义；更精确写法是 `k=3/5/9 + k7 d=2/4`。

### B5. Sparse graph Transformer

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| graph transformer configuration | `Sparse Graph Transformer 4 layers, 8 heads, edge dim 32` | input/output `B x L x 256`; neighbors `B x L x K`, `K<=96`; edge_attr `B x L x K x 32` | layers 0-1 frozen; layers 2-3 trainable in reported LLPS checkpoint | 4 layers, 8 heads, FFN dim 1024, edge dim 32, edge types 40, relative-position bins 32, max neighbors 96 | "A sparse graph Transformer updates residue states over up to 96 neighbors using edge attributes, edge-type bias and relative-position bias." | verified |
| attention formula | same | output `z_i: B x L x 256` | as above | `score_ijh = q_ih dot k_jh / sqrt(d_h) + b_edge(edge_attr_ij)_h + b_type(type_ij)_h + b_rel(relbin_ij)_h`; softmax over valid neighbors; residual/FFN/LayerNorm | "The graph encoder is sparse neighbor attention, not dense all-pairs sequence attention." | verified |

解释：图中主标签是正确的。Methods 或 supplementary table 应补足 edge types、relative-position bins 和 max neighbors，因为这些不适合全部放进图内。

### B6. Protein-level LLPS readout and BioPhys residual

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| `z_i` | `z_i (L x 256)` | output residue state `z_i: B x L x 256` | top graph layers trainable; lower modules mostly frozen in reported LLPS checkpoint | after adapters -> fusion -> local motif -> sparse graph Transformer | "The full-length encoder outputs residue-level states `z_i in R^{L x 256}`." | verified |
| attention + mean + local pooling | `Attention + mean + local pooling` | input `z_i: B x L x 256`; output protein representation `B x 768` | `llps_head` trainable | attention pool 256 + masked mean 256 + DPR/reference-logit or learned local-evidence pool 256; concat `B x 768` | "The protein-level LLPS head concatenates learned attention pooling, masked mean pooling and local-evidence pooling into a 768-dimensional representation." | verified |
| BioPhys residual | `BioPhys residual` | input protein-level `bio_vec: B x 33`; `bio_repr: B x 128`; concat with 768 -> 896; output residual logit | `bio_mlp` and `bio_fusion_head` trainable | `BioMLP: LayerNorm(33) -> 33->256->256->128`; `fusion: LayerNorm(896)->Linear(896,256)->GELU->Dropout(0.15)->Linear(256,1)`; output `base_logits + residual` | "Call this `BioPhys 33 residual MLP` to distinguish it from the 112-dimensional residue-level BioPhys tensor used by DPR." | verified |
| 33-dimensional features | `BioPhys residual` | `bio_vec` names include length, disorder fractions, charge/aromatic/hydropathy proxies, contact/protenix/graph/ESM/starling summaries, long-range contact fraction | trainable MLP | `BIO_VEC_NAMES` length is 33 | "The 33-dimensional protein-level BioPhys residual uses summary features listed in a supplementary table." | verified |

解释：`BioPhys residual` 是 protein-level 33 维 summary residual，不是 residue-level `L x 112` DPR BioPhys。图中建议改成 `BioPhys 33 residual MLP`。

### B7. Fixed mixture and LLPS branch independence from bridge

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| fixed mixture 0.8/0.2 | `Fixed mixture: 0.8 protein + 0.2 region` | inputs protein logit and internal region-global logit; output scalar `P_LLPS: B` | mixture weight fixed, not trained | `P_LLPS = 0.8 * sigmoid(llps_logits) + 0.2 * sigmoid(region_global_logits)`; then logit transform for calibrated output | "The final LLPS probability is a fixed mixture of the protein-level probability and an internal local-region-evidence probability." | verified |
| region in mixture vs final DPR region calls | same | `region_global_logits` from internal `MultiScaleDPRHead`, not final DPR v6 postprocessed intervals | internal LLPS `dpr_head` frozen in reported LLPS checkpoint | `MultiScaleDPRHead` computes region-global score from internal residue logits | "Clarify that the 0.2 region term is internal LLPS evidence, not the final right-panel DPR region calls." | verified |
| whether LLPS branch uses bridge state `c_i` | no direct label in LLPS box | LLPS uses `z_i` and internal heads; final DPR uses `c_i`; no path from `c_i` to LLPS classifier found | `peptide_module: absent in LLPS run`; bridge absent in LLPS checkpoint | LLPS forward does not concatenate `c_i`; final DPR `DPRV6PhaseStack.forward_llps` only calls frozen PhaseFlow | "The LLPS classifier does not use peptide bridge states; the bridge is used for the final DPR scanner." | verified |

解释：当前图里从 bridge 的 `c_i` 箭头进入右侧 DPR scanner 是对的；不应再画 `c_i` 回流到 LLPS classifier。

## C. Bridge and Remapping

### C1. From `z_i` to 32 bridge tokens

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| bridge input | `z_i (L x 256)` in bridge panel | input `h_gt/z_i: B x L x 256`; `seq_mask: B x L` | final DPR checkpoint: frozen/no_grad; generic module has parameters but not trainable in final v6 | `phaseflow_bridge(self.phaseflow, full_length_llps_hidden, seq_mask)` | "The bridge receives full-length residue states, not raw amino-acid tokens." | verified |
| ordered pooling algorithm | `Ordered pooling over residues` | input `B x L x 256`; output `B x 32 x 256` | frozen/no_grad in final DPR | `x=pool_proj(h)`; residue positions normalized by valid length; 32 centers `(k+0.5)/32`; logits = Gaussian distance logits + 0.25 content score; masked softmax over residues; token = weighted sum + token_bias | "Ordered pooling compresses length-`L` residue states into 32 sequence-ordered bridge tokens using position- and content-weighted masked pooling." | verified |
| bridge token dimension | `32 bridge tokens` | `B x 32 x 256` | frozen/no_grad in final DPR | `tokens = weights @ x + token_bias` | "The bridge tokens have the same 256-dimensional width as the peptide Transformer." | verified |

解释：这一步解决了“全长蛋白如何进入短肽模块”的逻辑问题。全长序列没有直接 tokenized 成短肽输入；它先被压缩为 32 个 ordered bridge embeddings。

### C2. Bridge adapter and frozen peptide Transformer reuse

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| 2-layer bridge adapter | `2-layer bridge adapter` | projections `256 -> 256`; pre-adapter output `B x 32 x 256` | final DPR checkpoint: frozen; used under `torch.no_grad()` | `_bridge_projection`: `LayerNorm -> Linear -> GELU -> Dropout -> LayerNorm -> Linear` when layers=2; plus `adapter_layers-1` TransformerEncoderLayer before frozen PhaseFlow | "The bridge includes two-layer projection adapters and a one-layer pre-adapter Transformer because `adapter_layers=2`." | verified |
| frozen peptide Transformer reuse | `Frozen Peptide Transformer (shared with peptide module)` | input token embeddings `B x 32 x 256`; phase tokens from zeros `B x 16 x 256`; hidden output for bridge tokens `B x 32 x 256` | final DPR checkpoint: PhaseFlow module frozen; bridge also frozen | `_phaseflow_forward_token_embeddings` calls `model.embed_phase(phase_t=0, mask=1, time=0)`, concatenates bridge token embeddings with phase tokens, runs frozen `model.transformer.layers` | "Full-length DPR reuses the Stage-1 PhaseFlow Transformer weights in frozen mode, with bridge embeddings supplied directly as token embeddings." | verified |
| whether token embedding table is used by bridge | `Frozen Peptide Transformer` | bridge path provides `token_emb` directly; raw token IDs are not embedded by `token_embed` in this path | frozen | `token_emb=pooled_tokens`; no `model.embed_tokens(input_ids)` inside bridge helper | "In the bridge path, the peptide token embedding table is bypassed because bridge tokens are already continuous embeddings." | verified |

解释：图中 `Frozen Peptide Transformer` 标签是正确的。更精确地说，full-length bridge reuses the Transformer and phase encoder context machinery; it does not feed a 5-20 aa string or token IDs into the tokenizer.

### C3. Residue-query cross-attention remap and gated bridge

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| peptide-token hidden context | `Peptide-token hidden context` | output from frozen PhaseFlow for 32 bridge token positions: `B x 32 x 256` | frozen/no_grad in final DPR | `hidden[:, :token_emb.shape[1], :]` from `_phaseflow_forward_token_embeddings` | "The peptide context used in full-length DPR is the hidden context of the 32 bridge token embeddings after frozen PhaseFlow processing." | verified |
| cross-attention remap | `Residue-query cross-attention remap query=z_i, key/value=peptide context` | query `B x L x 256`; key/value `B x 32 x 256`; output `B x L x 256` | frozen/no_grad in final DPR | `Q=query_proj(z_i)`; `A=softmax(QK^T/sqrt(d))`; `attended=A V`; PyTorch `MultiheadAttention(256, heads=8)` | "Residue positions query the 32-token peptide context by cross-attention to remap bridge information back to length `L`." | verified |
| gated bridge | `Gated Bridge`; `c_i (L x 256)` | input attended context `B x L x 256`; output `c_i: B x L x 256` | frozen/no_grad in final DPR | `context = LayerNorm(attended + FFN(attended))`; `gate=sigmoid(gate_raw)`, initialized to 0.075; `c_i = gate * context`, masked at padded residues | "The bridge contribution is globally gated; the gate parameter is initialized so that `sigmoid(gate_raw)=0.075`." | verified |
| whether `c_i` enters LLPS classifier | arrow from bridge to DPR only | `c_i` concatenated in DPR input; no LLPS use found | not applicable for LLPS; frozen DPR input stream | DPR concat `[ESM2; BioPhys; z_i; c_i]`; LLPS forward lacks `c_i` | "State that `c_i` is used by the final DPR scanner and is not used by the LLPS classifier." | verified |

解释：`Gated Bridge` 是最终 `c_i` 的定义，不是一个 independent DPR head。Methods 可以用一个小公式展示 `c_i = sigma(g) * CrossAttn(Q=z_i, K=C, V=C)`，其中 `C` 是冻结 PhaseFlow 输出的 32-token context。

## D. DPR Scanner

### D1. Scanner input composition

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| DPR scanner concat | `Per-residue concat: u_i = [ESM2; BioPhys; z_i; c_i] = L x 1904` | `ESM2: B x L x 1280`; `BioPhys: B x L x 112`; `z_i: B x L x 256`; `c_i: B x L x 256`; concat `u: B x L x 1904` | input streams frozen/offline; only v6 projection/scanner trainable | `1904 = 1280 + 112 + 256 + 256`; `h=torch.cat(streams, dim=-1)` | "The final DPR scanner concatenates ESM2, raw 112-dimensional BioPhys, direct full-length state `z_i`, and bridge state `c_i`." | verified |
| Protenix direct DPR input | DPR scanner input list | Protenix is not a direct concatenated stream in final DPR; influence is through `z_i` | not direct | final config `direct_protenix_stream: false`; `direct_starling_stream: false` | "Do not list Protenix as a direct DPR scanner input; it only contributes indirectly through the full-length encoder state." | verified |

解释：图中右侧只写 ESM2、BioPhys、`z_i`、`c_i` 是正确的。Protenix 不应在 DPR scanner input row 中再出现。

### D2. Residue adapter and multi-window scanner

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| residue adapter | `Residue Adapter 1904 -> 256` | input `B x L x 1904`; output features `B x L x 256` | `DPR_adapter_v6.projection` trainable | `LayerNorm(1904) -> Linear(1904,256) -> GELU -> Dropout(0.10) -> Linear(256,256) -> residual add -> LayerNorm(256)` | "The big DPR head first applies a residual adapter from the 1904-dimensional residue vector to 256 dimensions." | verified |
| centered masked AvgPool | `Centered Masked AvgPool(window) 33/129/257` | input features `B x L x 256`; each pooled output `B x L x 256` | no trainable parameters | mask invalid residues to zero; grouped `conv1d` with all-ones kernel and `padding=kernel//2`; divide by valid counts; odd centered windows | "Each profile is computed after centered mask-normalized average pooling with window sizes 33, 129 and 257." | verified |
| shared scanner structure | `Shared Scanner g_DPR LN -> 128 -> 32 -> 1; LeakyReLU; Sigmoid output` | pooled `B x L x 256`; logits `z33/z129/z257: B x L`; probabilities `p33/p129/p257: B x L` | `scanner_v6.shared_scanner` trainable | `LayerNorm(256)->Linear(256,128)->LeakyReLU(0.1)->Dropout(0.1)->Linear(128,32)->LeakyReLU(0.1)->Linear(32,1)`; `p=sigmoid(z)` | "The same scanner MLP is shared across the three window scales; there are no profile-specific scanner parameters." | verified |

解释：`DPRV6Head(head_type="big")` 是外层 head 名称；`g_DPR` 是共享 scanner MLP。图中 `DPR region calls` 不是 head 名称，而是 post-processing 输出。

### D3. p33, p129, p257 and reported profile

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| profile outputs | `p33`, `p129`, `p257` | each `B x L`; invalid residues zeroed | scanner trainable | `p_w = sigmoid(g_DPR(AvgPool_w(features)))` for `w in {33,129,257}` | "Report `p33`, `p129` and `p257` as three multi-scale residue probability profiles." | verified |
| reported profile | `Post-process p257`; `DPR region calls` | reported residue profile `p257: B x L`; region calls are intervals | postprocess nontrainable | final config `reported_metrics.profile: p257`; `selection_scale: p257`; selected raw p257 on PlanD non-PhasePro validation | "The reported DPR profile is raw `p257`, selected by the predeclared validation procedure." | verified |
| current SVG label risk | three profile mini-panels in DPR scanner | expected order should be 33/129/257 | not applicable | current SVG text appears to have center/right mini-plot internal labels swapped (`P257` image above lower `p129`, and `P129` image above lower `p257`) | "Ensure figure labels and plotted miniature labels consistently use `p33`, `p129`, `p257` from left to right." | potential conflict |

解释：The neural head outputs all three profiles. The final reported DPR profile and region calls use `p257`; `p33` and `p129` still exist and contribute to bag-level training aggregation.

### D4. Post-process and DPR region calls

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| smoothing | `Post-process p257: smooth + threshold + merge` | input raw `p257: L`; output smoothed profile `L` | nontrainable | uniform moving average, window 5, edge padding | "Region calling smooths raw `p257` with a length-5 uniform moving average using edge padding." | verified |
| threshold | same | smoothed profile -> binary mask `L` | nontrainable | `positive_i = smooth(p257_i) >= 0.5` | "The fixed threshold for Table 2 region calls is 0.5." | verified |
| merge gap | same | binary positive runs -> merged runs | nontrainable | merge adjacent predicted segments separated by gap `<=5` residues | "Predicted positive runs separated by at most 5 residues are merged." | verified |
| minimum segment length | `DPR region calls` | merged intervals -> filtered intervals | nontrainable | discard segments shorter than 6 residues | "Segments shorter than 6 residues are discarded." | verified |
| segment score and coordinates | `DPR region calls` | output intervals and scores | nontrainable | score = mean of top 30% smoothed residues within segment; internal coordinates 0-based inclusive, manuscript may convert to 1-based | "Report coordinate convention explicitly when presenting DPR region calls." | verified |
| benchmark IoU | not in figure | predicted intervals vs reference intervals | nontrainable evaluation | one-to-one matching at IoU threshold 0.25 | "IoU 0.25 is an evaluation matching criterion, not part of the neural scanner." | verified |

解释：`DPR region calls` means intervals produced from postprocessed `p257`. It is not a learned head and not a direct tensor key from `DPRV6Head`.

## E. Training and Objectives

### E1. Stage 1 peptide training

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| Flow Matching objective | `PhaseFlow Peptide Module`; `Phase Diagram readout` | `input_ids: B x 32`; target phase `B x 16`; velocity output `B x 16` | trainable in Stage 1 | `t~U(0,1)`; `x0~N(0,I)`; `x_t=(1-t)x0+t*x1`; target `v=x1-x0`; MSE over valid phase slots with `(n_valid/16)^2` weighting | "The peptide sequence-to-phase direction is trained with conditional flow matching over valid PSSI grid entries." | verified |
| LM objective | `Peptide design readout` | input phase `B x 16`; token logits `B x 32 x 32` | trainable in Stage 1 | phase tokens prepended; seq tokens causal and attend phase; shifted-token cross entropy ignoring PAD positions | "The reverse phase-to-sequence direction is trained as a phase-conditioned language model." | verified |
| loss weights | same | scalar total loss | trainable in Stage 1 | `L = 32 * L_flow + 1 * L_LM` in final config | "For the audited Stage-1 checkpoint, flow and LM losses were weighted 32 and 1." | verified |
| checkpoint selection | not in figure | validation metrics; model checkpoint | trainable Stage 1 | `best_model.pt` saved when validation loss improves; early stopping patience 20 | "The peptide checkpoint was selected by validation loss with early stopping." | verified |
| LM label caveat | not in figure | `labels: B x 32` | trainable Stage 1 | `labels=input_ids[:,1:]` + final `-100`; mask positions where `input_ids==PAD`; code does not use `som_token_id/eos_token_id` arguments | "Do not claim EOS/SOM-specific LM label masking unless the code path is revised or separately documented." | implementation caveat verified |

解释：The peptide module is trained with both sequence-to-phase and phase-to-sequence directions. It is frozen when reused by full-length DPR.

### E2. Stage 2 full-length LLPS training

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| LLPS objective | `Full-length LLPS Module`; `P_LLPS` | protein-level labels; output scalar `P_LLPS` | selected modules trainable | weighted focal BCE weight 1.0, focal gamma 1.5, class normalized pos/neg alpha 0.5/0.5 | "The full-length LLPS branch is optimized with class-normalized weighted focal BCE." | verified |
| ranking/hard-negative objectives | same | protein logits over sampled pools | trainable heads/top layers | ranking weight 0.2, margin 0.15, topk negatives 8; pairwise rank weight 0.4; hard-negative focal weight 2.0, gamma 2.0 | "Auxiliary ranking and hard-negative terms are used to separate positive, client/driver and curated negative pools." | verified |
| auxiliary heads | `BioPhys residual` area / not explicitly shown | driver/client scalar logits; negtype 2-class logits | trainable | driver weight 0.15; client weight 0.20; negtype weight 0.03 | "Driver, client and negative-type auxiliary losses are trained from the protein-level representation plus BioPhys residual features." | verified |
| frozen/trainable split | whole LLPS module | trainable: `llps_head`, `bio_mlp`, `bio_fusion_head`, `driver_head`, `client_head`, `negtype_head`, `encoder.layers.2`, `encoder.layers.3`; frozen: adapters, gate, local encoder, graph layers 0-1, region decoder, internal DPR head | mixed | `freeze_all_except_trainable_prefixes: true` | "Only the LLPS heads, BioPhys MLP and top graph layers are updated in the reported LLPS checkpoint." | verified |
| selection/curriculum | not in figure | checkpoint epoch 2; no internal validation | trainable during LLPS run | final config: full train no internal validation; selected epoch 2; selection basis `ppmc_shadow_AUPRC`; EMA eval true | "LLPS checkpoint selection and EMA evaluation should be described in Methods or supplementary training details, not in the schematic." | verified |

解释：图中不需要画所有 loss terms，但 Methods 必须区分 architecture from training. The frozen/trainable split is central because the figure uses trained/frozen color coding.

### E3. Stage 2 / final DPR training

| 核对项 | 图中对应模块名称 | 输入张量、输出张量和维度 | 训练状态 | 关键公式或伪代码 | 论文 Methods 一句话建议 | 证据状态 |
|---|---|---|---|---|---|---|
| DPR objective | `DPR Scanner`; `p257`; `DPR region calls` | residue profiles `B x L`; bag labels and residue targets | only v6 projection and shared scanner trainable | objective `rank_p257`; bag BCE weight 1.0; top-residue aggregation 0.25; positive residue BCE effective 0.35; Dice effective 0.06; strong pairwise ranking 0.75; weak pairwise ranking 0.1875; negative top suppression 0.35 | "The final DPR scanner is trained with a rank-focused `p257` objective combining bag, top-residue, residue-supervised and negative-suppression terms." | verified |
| bag aggregation | `DPR Scanner` | `p33/p129/p257: B x L` -> bag scores `B` | scanner trainable | mean of max-over-residue scores across scales plus optional top-fraction term; topk fraction 0.05 | "Bag-level supervision aggregates multi-scale residue profiles by max and top-fraction statistics." | verified |
| frozen/trainable split | whole right side and upstream modules | upstream full-length stream, PhaseFlow and bridge frozen; DPR adapter/scanner trainable | mixed | `_freeze_original_models`; `phaseflow_bridge` under `torch.no_grad()`; trainable names start with `v6.` | "In final DPR training, the full-length encoder, peptide module and bridge are frozen; only the DPR v6 adapter and shared scanner are updated." | verified |
| optimization/checkpoint | not in figure | checkpoint update 50; batch size 2 | trainable scanner | AdamW lr 5e-6, final lr 1.5e-6 at update 50, bf16, EMA 0.997, grad clip 1.0 | "Optimization details belong in Methods/training table, not the architecture figure." | verified |
| curriculum/checkpoint sequence | not in figure | final checkpoint initialized from previous `strong_p257` run; selected raw p257 | trainable scanner in final 50 updates | init checkpoint from PlanC strong_p257; final `rank_p257` 50 updates; selection split Plan D mixed HQ non-PhasePro validation | "The reported raw `p257` scanner was selected after a rank-focused fine-tuning stage using the predeclared validation rule." | verified |

解释：The DPR panel should be interpreted as the final frozen-upstream / trainable-scanner inference path. If Methods discuss earlier pretraining or ablation variants, they should clearly separate them from the final checkpoint shown in the schematic.

## F. Consistency Checklist

### F1. Figure labels and implementation support

| 图中文字标签 | 对应实现说明 | 当前状态 | 建议放置位置 |
|---|---|---|---|
| `Short-Peptide Input; 5-20 aa, <=32 tokens` | peptide dataset sequences are 5-20 aa; tokenizer pads to max 32 after special/shape tokens | consistent, but caption must distinguish aa length vs token length | figure caption |
| `PhaseFlow Peptide Module` | Stage-1 PhaseFlow model with Transformer backbone | consistent | main Methods + supplementary table |
| `Peptide Transformer` | 6 layers, 8 heads, dim 256, RoPE, RMSNorm, SwiGLU | consistent | supplementary table |
| `Peptide-token hidden context (16 phase slots)` | 16 PSSI phase slots exist, but phrase may imply peptide token count is 16 | minor ambiguity | figure caption |
| `Phase Diagram / Peptide design readout` | conceptual readout label for `generate_phase` and `generate_sequence`; not a single head name | consistent if explained | figure caption + Methods |
| `Full-length LLPS Module` | PhaseFlow full-length model, residue-aligned streams, graph encoder, LLPS head | consistent | main Methods |
| `ESM2 (L x 1280)` | verified input stream; exact ESM2 variant not located here | mostly consistent | supplementary table |
| `PhysChem (L x 90)` | verified LLPS adapter stream; per-feature list not located here | consistent with caveat | supplementary table |
| `Disorder (L x 6)` | verified LLPS adapter stream; per-feature list not located here | consistent with caveat | supplementary table |
| `Protenix (L x 512)` | verified LLPS adapter stream; not direct DPR stream | consistent | main Methods |
| `Adapters each -> 256` | FeatureAdapter per modality to 256 | consistent | supplementary table |
| `Gated Fusion (L x 256)` | ReliabilityGatedFusion with 258-d gate input and missing-mask softmax | consistent | main Methods |
| `Local motif Blocks kernels 3/5/9 dilation 2/4` | implementation is k=3/5/9 plus kernel-7 dilated branches at d=2/4 | should clarify wording | figure text or caption |
| `Sparse Graph Transformer 4 layers, 8 heads, edge dim 32` | verified; also edge types 40, rel bins 32, max neighbors 96 | consistent but incomplete | supplementary table |
| `Attention + mean + local pooling` | 3 x 256 pooling = 768-d protein representation | consistent | Methods |
| `BioPhys residual` | actually protein-level 33-d BioPhys residual MLP | ambiguous | figure text should say `BioPhys 33 residual MLP`; details in Methods |
| `Fixed mixture: 0.8 protein + 0.2 region` | verified fixed mixture; region is internal LLPS local-region evidence | consistent but requires disambiguation | caption |
| `z_i (L x 256)` | full-length residue state used by bridge and DPR | consistent | figure/caption |
| `Ordered pooling over residues` | verified monotonic position/content pooling to 32 tokens | consistent | Methods |
| `32 bridge tokens` | verified `B x 32 x 256` | consistent | figure |
| `2-layer bridge adapter` | verified projections; implementation also has one pre-adapter TransformerEncoderLayer when `adapter_layers=2` | simplified label; Methods must define | Methods |
| `Frozen Peptide Transformer` | PhaseFlow frozen in final DPR; bridge uses token embeddings directly | consistent | figure/caption |
| `Residue-query cross-attention remap` | verified query=`z_i`, key/value=PhaseFlow bridge context | consistent | Methods |
| `Gated Bridge` | verified global gate initialized to 0.075 | consistent | Methods |
| `c_i (L x 256)` | verified final bridge state; enters DPR concat | consistent | figure |
| `DPR Scanner` | final `DPRV6Head(head_type="big")` | consistent | Methods |
| `u_i=[ESM2; BioPhys; z_i; c_i]=L x 1904` | verified 1280+112+256+256=1904 | consistent | figure/caption |
| `Residue Adapter 1904 -> 256` | verified residual adapter | consistent; "Residual adapter" more precise | figure text |
| `Centered Masked AvgPool(33/129/257)` | verified centered odd-window mask-normalized pooling | consistent | Methods |
| `Shared Scanner g_DPR LN -> 128 -> 32 -> 1; LeakyReLU; Sigmoid output` | verified shared scanner | consistent | figure/caption |
| `p33, p129, p257` | verified three profiles | current SVG appears to swap internal mini-plot labels for p129/p257 | fix figure |
| `Post-process p257: smooth + threshold + merge` | verified; exact params are window 5, threshold 0.5, merge gap 5, min len 6 | consistent but incomplete | caption or supplementary table |
| `DPR region calls` | postprocessed interval calls, not a neural head output name | consistent if explained | figure caption + Methods |

### F2. What the paper should supplement

| 信息 | 是否正文需要补充 | 推荐位置 | 理由 |
|---|---|---|---|
| Short peptide 5-20 aa vs <=32 tokens | yes | figure caption or peptide Methods | prevents confusion that 32 means 32 aa |
| Special tokens and EOM caveat | yes, concise | supplementary table | tokenizer defines EOM, but build path does not append it |
| Peptide Transformer architecture | yes | supplementary table | too detailed for figure |
| Flow Matching + LM objective weights | yes | Methods training subsection | key training fact |
| Full-length input stream dimensions | yes | Methods or supplementary table | direct consistency with figure |
| Raw BioPhys 112 vs LLPS 90+6 split | yes | Methods + supplementary table | current figure can otherwise look dimension-inconsistent |
| Remaining 16 BioPhys dimensions | only if identified | supplementary table | currently `implementation not found / uncertain` |
| Gated fusion 258-d input | yes | Methods | users/readers may question 258 |
| Local motif exact branches | yes | Methods | figure shorthand is not exact enough |
| Sparse graph edge parameters | yes | supplementary table | important but too dense for figure |
| LLPS pooling 768-d and BioPhys 33 residual | yes | Methods | avoids ambiguity in `BioPhys residual` |
| Fixed 0.8/0.2 mixture and internal region meaning | yes | caption + Methods | avoids confusing LLPS internal region evidence with final DPR calls |
| Bridge pooling and gate 0.075 | yes | Methods | central novelty/information routing detail |
| Frozen/trainable split in full-length DPR | yes | Methods + supplementary training table | figure uses frozen/trainable semantics |
| DPR scanner exact input 1904 and no direct Protenix | yes | Methods | avoids input-stream misstatement |
| p257 postprocessing parameters | yes | figure caption or supplementary table | needed to reproduce region calls |
| Current p129/p257 mini-label mismatch | yes, by fixing figure | figure itself | visual inconsistency |

### F3. Potential conflicts or uncertainties

1. The current SVG appears to have a `p129`/`p257` mini-panel label mismatch: the center mini-plot contains an internal `P257` label while the lower text says `p129`, and the right mini-plot contains an internal `P129` label while the lower text says `p257`. The figure should show `p33`, `p129`, `p257` consistently from left to right.

2. `BioPhys residual` is ambiguous. Implementation shows a 33-dimensional protein-level `bio_vec` residual path; DPR uses a separate 112-dimensional residue-level raw BioPhys stream. The figure should distinguish these.

3. `2-layer bridge adapter` is an acceptable schematic shorthand only if Methods explains the exact implementation: two-layer projection adapters plus a pre-adapter TransformerEncoderLayer due to `adapter_layers=2`.

4. `Phase Diagram / Peptide design readout` is a conceptual label, not the exact name of a single neural head. Exact implementation names are `velocity_per_pos` for phase velocity prediction and `lm_head` for sequence modeling.

5. The tokenizer defines `EOM`, but the implemented `build_input_sequence` used by dataset encoding returns up to `SOM` and does not append `EOM`. Avoid claiming that `EOM` is always in the input stream.

6. The exact identities of the remaining 16 raw BioPhys dimensions after the 90 PhysChem and 6 Disorder streams were not recovered in this audit. It is safe to state that DPR uses the full 112-dimensional raw BioPhys tensor, while LLPS adapters use the 90 and 6 dimensional streams.

7. The exact ESM2 model variant corresponding to the 1280-dimensional embedding was not recovered in this audit. State the verified tensor dimension unless the model variant is separately confirmed.

8. The figure should not imply the LLPS classifier consumes `c_i`. Implementation routes `c_i` to the final DPR scanner only.

9. `DPR region calls` should be described as postprocessed intervals from raw `p257`, not as a trainable head output.

10. The Stage-2/Final DPR figure should be read as the final audited checkpoint path: upstream full-length encoder, PhaseFlow and bridge frozen; only DPR v6 projection and shared scanner trainable. If the paper discusses earlier training variants, separate them clearly from this final schematic.
