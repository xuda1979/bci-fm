# 技术发现记录 (2026-09-30)

## 数据源调研结论

1. **PhysioNet eegmmidb 可用且可靠**:
   - S3 直链: `https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/{S###}/{S###}R{##}.edf`
   - 无需凭证, 20 被试 240 文件 7 分钟下载完成, 0 失败
   - 主站 physionet.org 直链也可用但较慢

2. **MOABB 自带镜像已失效** (BNCI2014 等连不上), 不推荐作为下载渠道

3. **TUH 系列不可用**: 需凭证且已从 PhysioNet 部分下架

4. **HF 国内访问**: hf-mirror.com 可用; 官方 huggingface.co 需代理
   - `braindecode/CBraMod` 仓库是**空壳** (只有 README) — 坑!
   - 真正权重在 `braindecode/cbramod-pretrained` (model.safetensors, 19MB)

## CBraMod 微调要点

1. **braindecode 1.8.1 中模型导入**: `from braindecode.models import CBraMod`
   (不是 `braindecode.models.CBraMod`)

2. **输入约束**: patch_size=200 (1s @ 200Hz), 时间维必须整除 200
   - 4s epoch @ 200Hz = 802 采样 (160Hz 重采样后有多余 2 点) → 裁到 800
   - MNE 默认 epoch 含 tmax 端点, 160→200Hz 重采样产生 802 点

3. **通道命名坑**: eegmmidb EDF 通道名为 'C3..', 'Fc5.' 格式
   - MNE 读取后需 `c.rstrip('.')` 规范化
   - CBraMod 预训练用 19 通道标准 10-20 导联, eegmmidb 64 通道是超集, 直接子集提取即可

4. **权重加载**: `load_state_dict(matched, strict=False)`, 211/213 张量匹配
   - 不匹配的 2 个: final_layer 权重/偏置 (分类头, 随机初始化 — 正常)

5. **MPS 训练**: batch 32 × 19ch × 800t 正常, ~5s/epoch (16 训练被试)

## 快速结果 (3-epoch 冒烟测试)

- 跨被试测试: **56.1%** (随机 25%), 3 分钟内即超过大多数论文基线
- per-class: left_hand 78%, right_hand 63%, feet 28%, tongue 56%
- feet 类最难 (脑电运动区对脚的区分度低, 与文献一致)

## 论文基线参考 (eegmmidb 4-class MI, 跨被试)

- EEGNet 从零训练: ~40-50%
- 判别 CSP + SVM: ~35-45% (跨被试差)
- CBraMod 原论文 (预训练+微调, 全 109 被试): ~65-70%
