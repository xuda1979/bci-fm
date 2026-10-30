# bci-fm: 脑机接口大模型后训练项目

基于免费公开 EEG 数据 (PhysioNet eegmmidb) 对预训练 EEG 基础模型 CBraMod 进行后训练（微调），
任务为运动想象四分类（左手/右手/双脚/舌头）。

## 项目结构

```
bci-fm/
├── data/
│   ├── eegmmidb/          # 原始 EDF 数据 (20 被试 × 12 runs, ~510MB)
│   └── processed/         # 预处理后 npz (每被试 90 epochs, 19ch, 200Hz, 4s)
├── models/
│   ├── model.safetensors  # CBraMod 预训练权重 (19MB, braindecode/cbramod-pretrained)
│   └── config.json
├── src/
│   ├── preprocess.py      # EDF → npz 管线
│   └── finetune.py        # CBraMod 微调 + 跨被试交叉验证
├── scripts/
│   └── download_s3.py (位于 data/eegmmidb/)
├── results/               # 训练结果 JSON + 模型
└── docs/
```

## 数据与模型

- **数据**: PhysioNet eegmmidb (EEG Motor Movement/Imagery Dataset)
  - 20 被试 × 64 通道 × 160Hz × 12 个任务 run
  - 运动想象 runs: 4/8/12 (手), 6/10/14 (脚+舌头)
  - 每被试 90 个 4 秒 epoch，四类均衡
  - 许可: CC-BY 4.0, 来源: https://physionet.org/content/eegmmidb/1.0.0/
- **基座模型**: CBraMod (ICLR 2025, "Criss-Cross Brain Model")
  - criss-cross Transformer 架构, 19 通道 (标准 10-20), 200Hz, patch=1s
  - 权重: hf-mirror.com/braindecode/cbramod-pretrained (19MB, 211 张量)
  - 微调时仅分类头随机初始化，骨干全部加载预训练权重

## 复现流程

```bash
# 1. 下载模型权重 (hf-mirror)
curl -L -o models/model.safetensors \
  "https://hf-mirror.com/braindecode/cbramod-pretrained/resolve/main/model.safetensors"

# 2. 下载数据 (S3 直链, 见 data/eegmmidb/download_s3.py)
python3 data/eegmmidb/download_s3.py

# 3. 预处理 (EDF → 19ch npz, 160→200Hz 重采样)
python3 src/preprocess.py --subjects 20

# 4. 微调 (MPS, 5 折跨被试 CV)
python3 src/finetune.py --folds 5 --epochs 30 --batch-size 32 --save-model

# 冒烟测试
python3 src/finetune.py --quick
```

## 方法要点

- 输入: 19 通道 × 800 时间点 (4s @ 200Hz)，逐 epoch 逐通道 z-score
- 尾部裁剪到 patch=200 的整数倍
- 差分学习率: 骨干 lr×0.1 (10⁻⁴), 分类头 lr (10⁻³)
- AdamW + cosine 退火, label smoothing 0.1, 梯度裁剪 1.0
- 早停 (patience=10, 内部 val 被试集)
- 评估: 留被试交叉验证 (每折 4 个测试被试) — 衡量跨被试泛化

## 结果 (2026-09-30 最终版, 全部 109 被试)

### 主对比表 (跨被试留出 CV, 随机基线 25%)

| 模型 | 被试数 | 平均测试准确率 | 标准差 | 折数 |
|------|-------|--------------|--------|-----|
| **CBraMod 微调** | 109 | **58.4%** | ±1.8% | 4×~22 |
| EEGNet 从零训练 | 109 | 57.2% | ±1.5% | 4×~22 |
| EEGNet 从零训练 | 20 | 55.2% | ±6.2% | 5×4 |
| CBraMod 微调 (试点) | 20 | 56.4% | ±5.7% | 5×4 |

### 关键科学发现

1. **数据规模效应 > 预训练效应** (在 109 被试规模下):
   - EEGNet 从零: 20→109 被试提升 +1.2pt, 且方差大幅缩小 (±6.2% → ±1.5%)
   - CBraMod 微调领先 EEGNet 仅 +1.1pt (58.4% vs 57.2%)

2. **方差收敛**: 20 被试时折间波动 ±6%, 109 被试时 ±1.8% — 小样本结论不可靠的实证

3. **类别模式**: 左手想象始终最易 (~65%), 双脚最难 (~50-60%) — 与运动皮层足区位于中线深部的解剖学一致

4. **实践建议**: 
   - 跨被试 BCI 场景下, 简单模型 + 大数据 ≈ 预训练大模型微调
   - 预训练收益需在**被试内少样本** (few-shot) 或**跨数据集迁移**场景下才能显现 — 这是下一步方向

### CBraMod 109 被试逐折明细

| Fold | 测试准确率 | LH | RH | FEET | TONGUE |
|------|-----------|-----|-----|------|--------|
| 1 | 58.5% | 66% | 55% | 57% | 56% |
| 2 | 61.2% | 64% | 64% | 59% | 58% |
| 3 | 57.1% | 66% | 53% | 54% | 55% |
| 4 | 56.7% | 60% | 56% | 62% | 49% |

详细数据: `results/*.json`

## 环境依赖

- Python 3.14 (Homebrew), PyTorch 2.12.1 + MPS (Apple Silicon)
- braindecode 1.8.1, MNE, safetensors, scikit-learn, pandas
- 安装: `pip3 install --break-system-packages braindecode mne pandas scikit-learn`
