# 调研报告：多速率（快/慢）时序数据的统一建模 —— 如何榨干所有可获取数据的价值

日期：2026-10-30 ｜ 项目背景：bci-fm（CBraMod 微调，eegmmidb 109 被试）

---

## 0. 问题定义

你的直觉是对的，且在文献中有正式对应物。真实 BCI 系统里"快数据"与"慢数据"天然共存：

| 数据类型 | 典型采样率 | 例子 |
|---------|-----------|------|
| 极快 | 1–20 kHz | 颅内 ECoG/sEEG（高伽马 70–170Hz 需 ≥1kHz）、神经元 spike、肌电 EMG 爆发 |
| 快 | 160–500 Hz | 头皮 EEG（我们现在的 200Hz CBraMod 流）、EOG |
| 中 | 10–100 Hz | 加速度计/陀螺仪（头部姿态、运动伪迹参考）、近红外 fNIRS 原始光强 |
| 慢 | 0.1–10 Hz | 心率/HRV、皮电 GSR/EDA、呼吸、体温 |
| 离散 | 事件级 | 刺激标记、按键、眨眼事件 |

核心矛盾：**统一采样网格的深度模型（固定 patch 长度、固定位置编码）无法直接吃不同速率的输入** —— 把 1Hz 心率重采样到 200Hz 是浪费 200 倍算力并制造虚假自相关；把 1kHz ECoG 降到 200Hz 则丢掉高伽马信息（高伽马恰恰是解码语音/运动最有效的频段）。

文献中这个问题有三个名字，其实是同一件事：
1. **多速率融合**（multi-rate fusion，信号处理/穿戴设备方向）
2. **不规则采样**（irregular sampling，医疗时序方向，GRU-D 一系）
3. **异构生物信号统一建模**（cross-data learning，BIOT/CBraMod/LaBraM 一系）

---

## 1. 方案一：通道独立 + 时间单位 patching（BIOT 路线，推荐首选）

**来源**：BIOT: Biosignal Transformer for Cross-data Learning in the Wild (NeurIPS 2023, UIUC/Harvard)。我从原论文 PDF 抽取的方法细节：

- **逐通道独立 tokenization**：每条通道单独切成"长度 L、重叠 R"的 segment/token（例如 L=1s、R=0.5s），然后把所有通道的 token **展平成一条"句子"**喂给 Transformer。
- **重叠是关键设计**：原文明确说 "the overlap is essential to maintain the temporal information for shorter signals"——短信号靠重叠凑够 token 数。
- **相对位置编码**（而非绝对索引）：位置编码加在 segment 上以保留时空特征。因为每通道独立分词，通道数可变、时长可变、**采样率可变**（200Hz 通道每秒 1 个 token，2Hz 心率通道每秒仍是 1 个 token，只要 L 以"秒"为单位定义）。
- **缺失即丢弃**：缺通道/缺段落直接不生成对应 token，模型结构不变。论文演示了 16 通道 MGH EEG 与 2 通道 SHHS 睡眠数据联合预训练，产出 18 通道模型；TUAB 微调超从零训练（0.7925→0.8019）。

**为什么这是首选**：它把"多速率"问题转化为"token 数量问题"，Transformer 天然处理变长序列。与我们 CBraMod 的兼容性也好——CBraMod 本身就是 19 通道 × 1 秒 patch 的时空模型，等价于"每通道每秒 1 token"。

**代价**：需要把绝对位置索引改为**时间戳感知**的位置编码（t_start, t_end, channel_id 三个 embedding 相加），并重训练或至少微调位置嵌入层。

---

## 2. 方案二：多分辨率金字塔（快慢双分支 + 跨注意力）

**来源**：TimesNet (ICLR 2023) 的多周期思想 + 视觉领域多尺度 ViT/Swin 的成熟做法。

结构：
```
快分支：原始 1000Hz → 小 patch (25ms) → 浅层 CNN 下采样 → token 流 A（时间分辨率高）
慢分支：降采样 200Hz → 大 patch (1s)   → CBraMod 骨干     → token 流 B（语义分辨率高）
融合层：B 作为 query 对 A 做跨注意力（或双向），或者简单 concat 后过几层 Transformer
```

- TimesNet 的具体技巧：对 1D 序列做 FFT 找主周期，把序列按周期折叠成 2D 张量再卷积——**不同周期 = 不同"快慢尺度"被并行建模**。迁移到我们这里：EEG 的 alpha (10Hz)、mu (8–13Hz)、高伽马 (70–170Hz) 就是天然的多尺度。
- **优点**：快信号的高频信息（高伽马、EMG 爆发）不被迫先降采样；慢信号不被上采样灌水。各分支用各自擅长的感受野。
- **代价**：参数量与实现复杂度最高；两个分支的预训练权重来源不同（快分支往往要从零训）。

**适合场景**：数据里有真正的 kHz 级信号（ECoG、EMG），或同一信号既要精细时序又要长程上下文。

---

## 3. 方案三：连续时间模型（把"时钟"从 tick 换成 t）

**来源**：不规则采样时序文献——GRU-D (2018)、SeFT (NeurIPS 2020)、mTAND (NeurIPS 2020)、Neural CDE (NeurIPS 2019)、CRU (ICLR 2023)。

思想：不用固定采样网格，模型直接消费 `(时间戳, 值)` 对：
- **GRU-D**：RNN 里显式建模"距上次观测的时间 decay"，缺失值按可学习衰减填充——最便宜，适合慢通道稀疏更新的场景；
- **mTAND**：连续时间嵌入 + time-aware 注意力插值，任意时刻可查询隐状态；
- **Neural CDE / CRU**：把序列视为连续动力系统的离散观测，天然零成本适配任意速率；
- **SeFT**：把每个观测当集合元素，注意力聚合——通道数、速率、缺失全都不敏感。

**优点**：理论上最优雅，任意速率、任意缺失模式一个模型通吃。
**代价**：与现有预训练权重（CBraMod 这类 patch-based）不兼容，基本要从零训练；RNN 系列并行性差；在"所有数据都均匀采样、只是速率不同"的场景里属于杀鸡用牛刀——**只在数据真的不规则（事件驱动、丢包、间歇采集）时才值得**。

---

## 4. 方案四：频率域统一表示（频谱图作为"速率归一化器"）

把所有信号转成时频表示（STFT / CWT 小波谱）， Suddenly 速率差异变成"频率轴刻度差异"：
- 200Hz EEG → 0–100Hz 频谱图；1kHz ECoG → 0–500Hz 频谱图；
- 1Hz 心率 → 直接当慢 token（或提取 HRV 特征当"慢特征"注入）。

**优点**：高伽马这类高频信息保留最完整（这是语音解码、精细运动解码 literature 里 ECoG 强于 EEG 的根本原因）；频谱图天然对齐视觉模型生态（可用 ViT/卷积处理，甚至借用图像预训练权重——MMSplice/SpeechBERT 一系的做法）。
**代价**：STFT 窗长的选择本身又引入一个人工尺度；长序列 token 数爆炸（1kHz 信号 4 秒 = 4000 点，25ms 窗就有 160 帧）。

**实践折中**：对快信号用多频段带通能量（bandpower）作为"中间速率"特征——CSP/FBCSP 几十年就是这么干的，深度模型里等价于固定第一层为 Gabor 滤波器组。我们 feet 类别难解码，恰恰因为运动区低频带区分度低——**如果将来能拿到高伽马，feet/tongue 的解码上限会明显抬升**。

---

## 5. 方案五：分层 RNN 时间尺度（工程遗产，仍实用）

快层每个 tick 更新、慢层每 N 个 tick 更新（Clockwork RNN 2014、Hierarchical Multiscale RNN、HTM 的 temporal pooling 一脉）。实现极简，推理延迟低，适合**边缘设备上的实时 BCI**（分层更新天然省算力：慢层大部分时间冻结）。缺点是表达力和预训练生态不如 Transformer 系。

---

## 6. 针对本项目的具体建议（按成本排序）

### 阶段 1（零额外成本，立即可做）
**统一重采样到模型原生速率**。我们已经在做（160→200Hz）。如果拿到新数据源：
- 慢信号（心率/EDA/姿态）：**不要重采样到 200Hz**，改为提取滑动窗口统计特征（均值/斜率/频段能量），作为一个额外通道在 **epoch 级**（而非 sample 级）拼接——即"慢特征 late fusion"。
- 这是几乎所有穿戴设备论文的实际做法，性价比最高。

### 阶段 2（小改动）
**把 CBraMod 的 patch 语义从"200 个采样点"改为"1 秒"**，patch embedder 用 stride ∝ 采样率的卷积。这样 500Hz 数据进来每秒仍产生 1 个 token，**模型权重基本不用动**（第一层 conv 换成对应 stride 版本 + 微调）。这正是 LaBraM/BIOT "采样率无关"的工程本质。

### 阶段 3（中等成本，收益最大）
**BIOT 式多模态联合微调**：
- EEG 通道：走现有 CBraMod 分支；
- 慢通道（EOG 已在 eegmmidb 里、加速度计若有）：独立小 patch 分词 + 时间戳/通道类型 embedding，token 流与 EEG token 拼接过融合层；
- 位置编码改为 (channel_id, t_start) 二元组，训练时随机丢通道（channel dropout）增强缺失鲁棒性。

### 阶段 4（研究级）
若接入 ECoG/高伽马或 EMG：方案二双分支 + 跨注意力，快分支从零训，慢分支加载 CBraMod 预训练权重。

### 何时不需要这些
若所有可用数据最终都均匀采样且速率相近（比如全是 200Hz EEG），**最优解就是全部重采样到 200Hz 拼通道**——BIOT 论文自己也是先把各数据集重采样到各自统一速率再分词的。不要为了架构优雅引入不必要的复杂度；我们 109 被试实验已经证明：数据本身 > 架构花样。

---

## 7. 关键文献清单

1. **BIOT** — Yang, Westover, Sun. *Biosignal Transformer for Cross-data Learning in the Wild*. NeurIPS 2023. 通道独立分词 + 重叠 token + 相对位置编码 + 缺失即丢弃；16ch+2ch 联合预训练实证。代码：github.com/ycq091044/BIOT
2. **TimesNet** — Wu et al. *TimesNet: Temporal 2D-Variation Modeling*. ICLR 2023. FFT 周期折叠 1D→2D，多尺度并行建模。
3. **CBraMod** — Wang et al. ICLR 2025. 我们当前骨干：19ch×200Hz×1s patch，criss-cross 时空注意力。
4. **LaBraM** — Jiang et al. ICLR 2024. EEG patch 化预训练，跨数据集时统一重采样。
5. **GRU-D** — Che et al. Scientific Reports 2018. 时间 decay 填充，不规则采样 RNN 基线。
6. **SeFT** — Horn et al. NeurIPS 2020. 观测即集合元素，注意力聚合。
7. **mTAND** — Shukla & Marlin. NeurIPS 2020. 连续时间嵌入 + time-aware 插值注意力。
8. **Neural CDE** — Kidger et al. NeurIPS 2019. 序列作为连续动力系统观测。
9. **Clockwork RNN** — Koutník et al. ICML 2014. 分层时间尺度 RNN。

---

## 8. 一句话结论

"快数据与慢数据"的问题是真实且被充分研究过的；**对均匀采样的多速率数据，首选"以秒为单位分词 + 通道独立 + 时间感知位置编码"（BIOT 路线），慢信号提取窗口统计特征做 epoch 级融合；只有数据真的不规则或含 kHz 级信号时，才上连续时间模型或双分支跨注意力**。对我们项目最务实的下一步是阶段 2+3：把 CBraMod 的 patch 语义改成时间单位，让任意速率数据都能进同一个模型。
