# PhaseFlow 训练脚本使用说明

## 目录结构

```
PhaseFlow/
├── phaseflow/           # 核心代码包
│   ├── __init__.py
│   ├── model.py         # PhaseFlow模型
│   ├── transformer.py   # Transformer backbone
│   ├── data.py          # 数据集
│   ├── tokenizer.py     # 分词器
│   └── utils.py         # 工具函数
├── train/               # 训练脚本
│   └── train.py         # 训练入口
├── scripts/             # Shell脚本
│   ├── train.sh         # 训练
│   ├── resume.sh        # 断点恢复
│   └── infer.sh         # 推理
├── config/
│   └── default.yaml     # 配置文件
└── outputs/             # 训练输出
```

## 快速开始

### 1. 安装依赖

```bash
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认配置训练
bash scripts/train.sh 0

# 自定义参数
bash scripts/train.sh 0 --batch_size 128 --epochs 200 --lr 5e-4
```

### 3. 断点恢复

```bash
bash scripts/resume.sh 0 outputs/run_xxx/best_model.pt
```

### 4. 推理预测

```bash
# 准备序列文件 (每行一个序列)
echo -e "ACDEFGHIKLMNPQRST\nFGHIKLMNPQRSTVWY" > sequences.txt

# 预测相图
bash scripts/infer.sh outputs/run_xxx/best_model.pt 0
```

## 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --batch_size | 批量大小 | 64 |
| --lr | 学习率 | 1e-4 |
| --epochs | 训练轮数 | 100 |
| --dim | 模型维度 | 256 |
| --depth | Transformer层数 | 6 |

## 监控训练

```bash
# 查看日志
tail -f outputs/run_xxx/train.log

# 使用TensorBoard (如果安装了)
tensorboard --logdir outputs/
```

## 输出文件

训练完成后，`outputs/run_xxx/` 目录下会有：

| 文件 | 说明 |
|------|------|
| best_model.pt | 最佳模型 |
| final_model.pt | 最终模型 |
| config.yaml | 训练配置 |
| train.log | 训练日志 |
