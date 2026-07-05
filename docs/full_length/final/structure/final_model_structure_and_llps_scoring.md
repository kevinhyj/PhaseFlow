# 最终模型结构与 LLPS 打分机制说明

本文档记录最终发布模型的结构、两个 checkpoint 文件的关系，以及 LLPS 分数到底由哪些 head 共同给出。

涉及的模型文件：

- `artifacts/model/checkpoints/update_000050.pt`
- `artifacts/model/checkpoints/update_000050.final_ema_inference.pt`

## 1. 两个 checkpoint 的关系

`update_000050.pt` 是完整训练 checkpoint。它包含：

- `model_state_dict`
- `ema`
- `optimizer`
- `scheduler`
- `sampler`
- RNG 状态
- PhaseFlow 原始 checkpoint metadata
- PhaseFlow 原始 checkpoint metadata

`update_000050.final_ema_inference.pt` 是最终推理 artifact。它保留完全相同的模型图和同一套 414 个 `model_state_dict` tensor，但把原始 checkpoint 里的 EMA shadow 直接写入外层 `v6.*` DPR head，并删掉训练恢复才需要的内容。

最终推理权重等价于：

```text
update_000050.pt:model_state_dict
+ update_000050.pt:ema.shadow 应用到 v6.*
```

已经做过 tensor 级校验：

```text
final_ema_inference.pt 与 update_000050.pt + EMA(v6) 的最大差 = 0.0
```

两者的模型结构完全相同。区别是：

- `update_000050.pt` 是训练态大包，约 326 MB。
- `update_000050.final_ema_inference.pt` 是最终推理模型，约 97 MB。
- `final_ema_inference.pt` 中只有外层 `v6.*` DPR head 使用了 EMA 后权重。
- `phaseflow`、`phaseflow`、`phaseflow_bridge` 权重与 raw `model_state_dict` 一致。

## 2. 顶层模型结构

最终部署模型是 `DPRV6PhaseStack`。

| 模块 | 类 | 参数量 | 作用 |
|---|---:|---:|---|
| `phaseflow` | `PhaseFlowModel` | 15,810,452 | 冻结的 LLPS / 表征模型 |
| `phaseflow` | `PhaseFlow` | 7,124,385 | 冻结的 phase-diagram flow 模型 |
| `phaseflow_bridge` | `PhaseFlow32GlobalBridge` | 1,852,675 | 冻结的 PhaseFlow 到 PhaseFlow 上下文桥接模块 |
| `v6` | `DPRV6Head` | 595,361 | 最终 EMA 后的 DPR head |
| `v6_feature_projectors` | 空 `ModuleDict` | 0 | 当前 arm 未使用 |

`model_state_dict` 总参数量：

```text
25,382,873
```

最终 arm 是：

```text
arm = d1_flat
head_type = big
```

关键配置：

```text
esm2_dim = 1280
biophys_dim = 112
full_length_llps_hidden_dim = 256
phaseflow_hidden_dim = 256
phaseflow_bridge_tokens = 32
phaseflow_bridge_adapter_layers = 2
phaseflow_bridge_gate_init = 0.075
num_heads = 8
adapter_dim = 256
window_sizes = [33, 129, 257]
topk_fraction = 0.05
use_base_streams = true
use_full_length_llps_stream = true
use_phaseflow_stream = true
use_pstp650 = false
use_pstp_esm8 = false
use_pstp_alb = false
```

外层 DPR v6 输入向量共 1904 维：

| 输入片段 | 维度 | 含义 |
|---|---:|---|
| `esm2` | 1280 | 离线 residue-aligned ESM2 表征 |
| `biophys` | 112 | 离线 DPR BioPhys 表征 |
| `full_length_llps_direct` | 256 | PhaseFlow residue hidden state |
| `phaseflow_bridge` | 256 | bridge 派生的 PhaseFlow context |

注意：外层 DPR 的 112 维 `biophys` stream 和 PhaseFlow 内部的 `physchem` residue stream、`bio_vec` protein-level vector 不是同一个东西。DPR ablation 中的 `w/o BioPhys` 指的是外层 DPR 的 112 维输入流，不是移除 PhaseFlow 内部所有理化信息路径。

## 3. PhaseFlow 主干结构

`phaseflow` 是非 decoupled 的 `PhaseFlowModel`：

```text
model_type = v2_region
d_model = 256
forward_mode = full
ablation.name = no_starling_embed_train
bio_mlp.enabled = true
local_encoder.num_layers = 3
graph_transformer.num_layers = 4
graph_transformer.num_heads = 8
graph_transformer.ffn_dim = 1024
graph_transformer.edge_dim = 32
graph_transformer.num_edge_types = 40
region_mil_head.windows = [33, 129, 257]
region_mil_head.topk_ratio = 0.04
region_mil_head.max_weight = 0.25
region_decoder.num_queries = 24
region_decoder.num_layers = 2
final_llps_alpha = 0.8
llps_logit_bias = 0.0
llps_logit_temperature = 1.0
```

### 3.1 PhaseFlow forward 路径

最终非 decoupled PhaseFlow 的特征处理顺序是：

```text
5 路 residue modality tensor
-> 每路 FeatureAdapter
-> ReliabilityGatedFusion
-> LocalMotifEncoder，3 个 block
-> SparseGraphTransformer，4 层
-> LLPS head / 内部 DPR head / region decoder / key head / bio auxiliary heads
```

五路 residue modality：

| 模态 | 输入维度 | adapter 结构 |
|---|---:|---|
| `plm` | 1280 | `Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm` |
| `physchem` | 90 | 同上 |
| `disorder` | 6 | 同上 |
| `protenix_embed` | 512 | 同上 |
| `starling_embed` | 512 | 同上，但最终 `no_starling_embed_train` ablation 会禁用该 modality |

adapter 输出概念形状：

```text
batch x length x 5_modalities x 256
```

`ReliabilityGatedFusion` 会在每个 residue 上对存在的模态做 learned softmax。gate 输入为：

```text
[modality_repr_256, reliability, missing_mask]
```

融合后得到：

```text
batch x length x 256
```

### 3.2 LocalMotifEncoder

`LocalMotifEncoder` 有 3 个相同的 `LocalMotifBlock`。每个 block 有 5 个并行一维卷积分支：

```text
Conv1d kernel 3
Conv1d kernel 5
Conv1d kernel 9
Conv1d kernel 7 dilation 2
Conv1d kernel 7 dilation 4
```

5 个分支 concat 后进入：

```text
Conv1d 1280 -> 512
GLU -> 256
Dropout
Conv1d 256 -> 256
residual add
LayerNorm
sequence mask
```

每个 local block 参数量为：

```text
2,755,072
```

### 3.3 SparseGraphTransformer

sparse graph encoder 有 4 层 `SparseGraphTransformerLayer`。每层包含：

```text
q_proj / k_proj / v_proj / out_proj: Linear 256 -> 256
8 heads, head_dim = 32
edge_bias: Linear 32 -> 8
edge_type_bias: Embedding(40, 8)
relative_position_bias: Embedding(33, 8)
FFN: Linear 256 -> 1024 -> 256
LayerNorm x2
```

attention 是 sparse neighbor attention。score 由以下几部分相加：

```text
query-key attention
+ edge_attr bias
+ edge_type bias
+ relative_position bias
```

### 3.4 PhaseFlow 输出 head

PhaseFlow 内部 head 如下：

| Head | 参数量 | 是否直接影响最终 LLPS 分数 | 作用 |
|---|---:|---|---|
| `llps_head` | 197,635 | 是 | 主 protein-level LLPS logit |
| `bio_mlp` | 108,738 | 是 | 编码 33 维 protein-level bio vector |
| `bio_fusion_head` | 231,681 | 是 | 对 LLPS logit 做 residual 修正 |
| `dpr_head` | 264,196 | 是，间接 | PhaseFlow 内部 DPR evidence，用于 LLPS pooling 和 20% final mixture |
| `region_decoder` | 2,179,844 | 否 | region query decoder 输出 |
| `key_head` | 66,049 | 否 | residue key logits |
| `driver_head` | 116,737 | 否 | 辅助 protein label |
| `client_head` | 116,737 | 否 | 辅助 protein label |
| `negtype_head` | 116,866 | 否 | 辅助 negative-type label |

最终模型中未启用的可选 PhaseFlow 分支：

```text
dpr_scan_residual = disabled
dpr_adapter = disabled
llps_reference_dpr_head = disabled
phase_head = disabled
dpr_summary_head = disabled
dpr_localization_branch = disabled
```

## 4. PhaseFlow 主干结构

`phaseflow` 是冻结的 `PhaseFlow` 模型：

```text
dim = 256
depth = 6
heads = 8
dim_head = 32
vocab_size = 32
phase_dim = 16
max_seq_len = 32
dropout = 0.1
use_set_encoder = true
diffusion_type = flow_matching
```

主要组件：

```text
token_embed: Embedding(32, 256)
SetPhaseEncoder:
  SinusoidalPosEmb(256)
  Linear 256 -> 512 -> 256
  16 个 learned phase-position embeddings
time_mlp: Linear 256 -> 1024 -> 256
Transformer: 6 blocks, dim 256
velocity_per_pos: Linear 256 -> 64 -> 1
lm_head: Linear 256 -> 32
velocity_head: Linear 256 -> 16
```

每个 PhaseFlow transformer block：

```text
RMSNorm
Attention:
  q/k/v/out Linear 256 -> 256，无 bias
  8 heads x 32 dim
  对 sequence token 使用 RoPE
RMSNorm
SwiGLU FFN:
  w1: Linear 256 -> 1024
  w3: Linear 256 -> 1024
  w2: Linear 1024 -> 256
```

PhaseFlow 支持 sequence-to-phase flow prediction 和 phase-conditioned sequence language modeling。在最终 DPR path 里，它作为冻结的上下文 token processor 被 bridge 调用。

## 5. PhaseFlow bridge

`phaseflow_bridge` 是 `PhaseFlow32GlobalBridge`，不是普通 MLP。它把 PhaseFlow residue state 映射到 PhaseFlow token space，跑一遍冻结 PhaseFlow，再把 PhaseFlow context 投回 residue 位置。

当前 bridge gate：

```text
sigmoid(gate_raw) = 0.074999988
```

bridge 流程：

```text
PhaseFlow hidden: batch x length x 256
-> pool_proj: 256 -> 256
-> monotonic_pool 成 32 个有序 token
-> 对 32 token 跑 1 层 TransformerEncoderLayer
-> frozen PhaseFlow transformer，phase_t = 0, time = 0
-> query_proj 到每个 residue hidden
-> residue query 对 PhaseFlow tokens 做 cross-attention
-> out_ffn + LayerNorm
-> 乘以 bridge gate
-> 输出 batch x length x 256
```

`monotonic_pool` 使用沿序列均匀排列的 32 个中心点。每个 token 是 residue hidden 的 weighted summary，权重由 learned width、位置距离和 content score 共同决定。这样既保留序列顺序，又能把任意长度序列压缩成 PhaseFlow 可处理的固定 32-token 表示。

## 6. 外层 DPR v6 head

最终 DPR head 是：

```text
DPRV6Head(head_type="big")
```

输入：

```text
h = concat(ESM2_1280, DPR_BioPhys_112, PhaseFlow_256, PhaseFlowBridge_256)
h shape = batch x length x 1904
```

projection：

```text
LayerNorm(1904)
Linear 1904 -> 256
GELU
Dropout
Linear 256 -> 256
residual add
LayerNorm(256)
```

scanner：

```text
对每个 window in [33, 129, 257]:
  masked average pool over residues
  shared scanner:
    LayerNorm(256)
    Linear 256 -> 128
    LeakyReLU
    Dropout
    Linear 128 -> 32
    LeakyReLU
    Linear 32 -> 1
  sigmoid -> p33 / p129 / p257
```

三个尺度共享同一个 scanner。head 同时输出：

```text
p33
p129
p257
bag_hard
bag_topk
head_features
residue_aligned_h
```

只有 `v6.*` tensor 在 `final_ema_inference.pt` 中被 EMA 更新。PhaseFlow、PhaseFlow、bridge 都是冻结权重。

## 7. 三个 task 的实际路径

### 7.1 `task="dpr"`

DPR path 使用完整栈：

```text
offline ESM2
+ offline DPR BioPhys
+ PhaseFlow residue hidden
+ PhaseFlow bridge hidden
-> concat 成 1904
-> v6 DPR head
-> p33, p129, p257, bag_hard, bag_topk
```

### 7.2 `task="llps"`

LLPS path 只调用：

```text
phaseflow(batch)
```

然后取：

```text
logit = phaseflow_output["final_llps_logits"]
probability = sigmoid(logit)
```

外层 `v6` DPR head 不参与 LLPS 打分。`phaseflow_bridge` 和 `phaseflow` 也不参与 `task="llps"`。

### 7.3 `task="phaseflow"`

如果提供 PhaseFlow batch，则直接调用冻结 PhaseFlow 的 `forward_flow`。如果只提供 sequence，wrapper 会构造 PhaseFlow token 输入并跑 full-sequence PhaseFlow inference。

## 8. LLPS 精确打分公式

最终非 decoupled PhaseFlow 的 LLPS score 使用三类信息：

1. `llps_head`
2. `bio_mlp` + `bio_fusion_head`
3. PhaseFlow 内部 `dpr_head` 的 global region evidence

它不使用外层 `v6` DPR head。

### 8.1 residue 表征

PhaseFlow 先得到 residue 表征：

```text
x = PhaseFlowEncoder(features)
x shape = batch x length x 256
```

这里的 `x` 是 adapter、reliability-gated fusion、local motif encoder、sparse graph transformer 之后的输出。

### 8.2 PhaseFlow 内部 DPR head

PhaseFlow 内部 `dpr_head` 运行在 `x` 上：

```text
dpr = phaseflow.dpr_head(x, seq_mask)
llps_reference_logits = dpr["dpr_logits"]
region_global_logits = dpr["region_global_logits"]
```

内部 DPR global score 的定义：

```text
region_global_score =
  (1 - max_weight) * topk_mean(sigmoid(dpr_logits))
  + max_weight * max(sigmoid(dpr_logits))

topk_ratio = 0.04
max_weight = 0.25
```

所以内部 DPR global score 是一个弱 region-evidence term：如果模型认为某些局部区域很像 condensation-driving region，那么 protein-level LLPS score 会被适度抬高。

### 8.3 `LLPSProteinHead`

`llps_head` 对 `x` 做三种 pooling：

```text
attention_pool = softmax(llps_head.pool(x)) 加权求和 x
mean_pool = masked mean of x
high_dpr_pool = softmax(llps_reference_logits) 加权求和 x
```

然后：

```text
protein_repr_768 = concat(attention_pool, mean_pool, high_dpr_pool)
base_llps_logits = MLP(768 -> 256 -> 1)
```

这是主 protein-level LLPS logit。

### 8.4 protein-level bio residual

因为 `bio_mlp.enabled = true`，模型还会读取：

```text
batch["bio_vec"]
```

这是 33 维 protein-level bio vector。路径为：

```text
bio_repr = bio_mlp(bio_vec)  # 33 -> 256 -> 256 -> 128
protein_repr_768 = 上面同一个三池化 protein 表征
bio_residual = bio_fusion_head(protein_repr_768, bio_repr)
llps_logits = base_llps_logits + bio_residual
```

`bio_fusion_head` 是对 `concat(protein_repr_768, bio_repr_128)` 的 residual MLP。它在模块初始化时最后一层是零初始化，但 checkpoint 里是训练后的权重。

`driver_head`、`client_head`、`negtype_head` 也从类似的 protein/bio representation 输出辅助 logits，但它们不直接进入 `final_llps_logits`。

### 8.5 最终 LLPS mixture

最终 LLPS probability 是概率空间的固定 mixture：

```text
final_llps_prob =
  0.8 * sigmoid(llps_logits)
  + 0.2 * sigmoid(region_global_logits)
```

然后：

```text
final_llps_logits = logit(final_llps_prob)
```

当前 calibration 是 no-op：

```text
llps_logit_temperature = 1.0
llps_logit_bias = 0.0
```

所以最终发布模型里：

```text
reported LLPS probability = sigmoid(final_llps_logits)
                          = final_llps_prob
```

一句话公式：

```text
x = PhaseFlowEncoder(features)
dpr_internal = PhaseFlowInternalDPRHead(x)
base = LLPSProteinHead(x, dpr_logits=dpr_internal.dpr_logits)
bio = BioFusionResidualHead(base, pooled_x, BioMLP(bio_vec))
final_prob = 0.8 * sigmoid(bio) + 0.2 * sigmoid(dpr_internal.region_global_logits)
```

## 9. 哪些 head 实际决定 LLPS 输出

| 组件 | 是否影响最终 LLPS | 方式 |
|---|---|---|
| `llps_head` | 是 | 主 protein-level LLPS logit |
| `bio_mlp` | 是 | 编码 33 维 protein-level bio vector |
| `bio_fusion_head` | 是 | 对 `llps_head` logit 做 residual correction |
| PhaseFlow 内部 `dpr_head` | 是 | 给 `llps_head` 提供 high-DPR pooling，并通过 `region_global_logits` 贡献 20% |
| `driver_head` | 否 | auxiliary output only |
| `client_head` | 否 | auxiliary output only |
| `negtype_head` | 否 | auxiliary output only |
| `key_head` | 否 | residue key output only |
| `region_decoder` | 否 | region query outputs only |
| 外层 `v6` DPR head | 否 | 只用于最终 DPR task |
| `phaseflow_bridge` | 对 LLPS 否 | 只用于外层 DPR v6 path |
| `phaseflow` | 对 LLPS 否 | 用于 `task="phaseflow"` 和外层 DPR bridge |

## 10. 这种 LLPS 推理方式是否合理

从单纯推理结构看，这种 LLPS 分数是合理的，但论文里必须透明描述，不能写成“只由一个 LLPS head 输出”。

它本质上是：

```text
protein-level LLPS classifier
+ region-evidence prior
```

更具体地说：

```text
80% 来自 protein-level LLPS evidence
20% 来自内部 region-level condensation-driving evidence
```

这个设计在生物学上说得通。LLPS 往往不是完全由全局平均性质决定，而是由局部 IDR、低复杂度区、芳香/带电 patch、R/G-rich 区域等驱动。一个蛋白是否发生 LLPS，局部强驱动区域应该能提高 protein-level score。因此，将内部 `region_global_logits` 作为弱 region prior 融入 LLPS 分数，是合理的推理定义。

但需要注意：

1. 不能说 LLPS 分数只来自 `llps_head`。实际最终分数是 `final_llps_logits`。
2. 不能把 PhaseFlow 内部 `dpr_head` 和外层最终 DPR v6 head 混淆。LLPS 用的是 PhaseFlow 内部 `dpr_head`，不是 `phaseflow_bridge + v6` 那个最终 DPR 模型。
3. `0.8 / 0.2` 不是临时后处理，而是 checkpoint 结构里的固定 scoring rule：`final_llps_alpha = 0.8`。
4. 如果论文里使用 `final_llps_logits` 跑 LLPS benchmark，就应该把它描述为模型正式的 final protein-level LLPS score。
5. 只要 LLPS benchmark 没有 label leakage，且所有报告结果都来自同一个 `final_llps_logits` 定义，这个分数可以用于论文。

推荐论文表述：

```text
The reported LLPS probability is the model's final protein-level score, computed
as a fixed probability-space mixture of the LLPS protein head and an internal
region-evidence head. The LLPS head pools residue representations with attention,
mean pooling and DPR-weighted pooling; a protein-level biophysical residual head
then adjusts the logit. The final probability combines this LLPS probability with
the internal DPR global region probability using a fixed 0.8/0.2 weight.
```

中文表述可以写成：

```text
本文报告的 LLPS 概率为模型的最终蛋白级分数。该分数由蛋白级 LLPS head
和内部区域证据 head 在概率空间中固定加权得到。LLPS head 首先对 residue
representation 进行 attention pooling、mean pooling 和 DPR-weighted pooling；
随后 protein-level biophysical residual head 对 logit 进行修正。最终概率以
0.8/0.2 的固定权重融合 LLPS 概率和内部 DPR global region 概率。
```

更短的摘要说法：

```text
LLPS score integrates global protein-level LLPS evidence and local
condensation-driving region evidence.
```

这比“LLPS score is produced by a single LLPS head”更准确。
