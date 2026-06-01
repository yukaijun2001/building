# Building Energy Forecasting Experiment

本目录实现 `../我的实验.md` 中的建筑能耗预测实验。实验参考了 `参考模型/TimeEmb/TimeEmb-main` 的工程结构，并结合 `参考总结.md`、SRSNet 的 patch 选择思想，以及 gated attention 论文中的“注意力输出后接 sigmoid gate”的做法，搭建了一个可训练、可评估、便于后续消融的框架。

## 目录结构

- `data_provider/`: 数据读取、缺失值处理、异常值处理、时间特征构造和滑动窗口数据集。
- `models/`: 核心模型 `PatchGatedLSTM.py`。
- `exp/`: 训练、验证、测试、指标计算和结果保存。
- `utils/`: 实验配置、随机种子、设备选择、JSON 保存等工具。
- `layers/`: RevIN、频域去噪、patch 选择与重排、gated attention 等可复用层。
- `scripts/`: 预留给批量实验脚本。
- `run_experiment.py`: 实验入口。

## 实验目标

给定每栋建筑的历史气象变量、时间特征和历史用电量，预测未来一段时间的 `electricity`。默认设置为：

- 输入窗口：过去 `168` 个小时。
- 预测窗口：未来 `24` 个小时。
- 数据文件：`../data/*.csv` 中的 9 个建筑 CSV。
- 评估方式：按建筑分别训练和测试，最后汇总平均指标。

## 整体流程

1. **读取建筑数据**
   从 `data` 目录读取每个 CSV。每个文件包含 `timestamp`、气象变量和目标列 `electricity`。

2. **时间排序和特征扩展**
   解析 `timestamp`，按时间排序，并加入周期性时间特征：
   `hour_sin`、`hour_cos`、`dow_sin`、`dow_cos`、`month_sin`、`month_cos`。

3. **缺失值处理**
   对数值列使用基于时间索引的插值，然后前向填充、后向填充；仍缺失的值填为 `0`。

4. **异常值处理**
   只在训练集上拟合预处理参数，计算每个特征的 1% 和 99% 分位数。训练、验证、测试都使用训练集分位数进行裁剪，避免未来信息泄漏。

5. **全局标准化**
   使用训练集裁剪后的均值和标准差，对全部 split 做标准化。模型内部还会使用 RevIN 做样本级归一化。

6. **时间顺序切分**
   按时间顺序划分：
   `70% train`、`10% val`、`20% test`。验证集和测试集起点会向前保留 `seq_len`，保证第一个窗口拥有完整历史上下文。

7. **滑动窗口构造**
   每个样本由历史窗口 `x: [seq_len, features]` 和未来目标 `y: [pred_len, 1]` 组成。

## 模型流程

核心模型位于 `models/PatchGatedLSTM.py`，名称为 `PatchGatedLSTMForecaster`。方案 B 融合版本也在同一文件中，名称为 `PatchGatedSegRNNForecaster`。

1. **RevIN 归一化**
   对每个 batch 样本沿时间维计算均值和标准差，执行实例级归一化。预测输出后只对 `electricity` 目标维度执行反归一化。

2. **频域去噪和门控**
   对输入序列沿时间维做 `rfft`。保留低频部分，并通过可学习的 sigmoid 频域 gate 调节各频率分量，最后用 `irfft` 回到时域。这个阶段对应你的“利用频域时域之间的关系进行去噪”的想法。

3. **Patch 切分**
   将历史序列切成多个 patch。默认：
   `patch_len=24`，`patch_stride=12`。每个 patch 表示一段局部时间模式。

4. **SRSNet 风格 MLP patch 打分**
   将每个候选 patch 的时间片段输入 `Linear -> ReLU -> Dropout -> Linear` 评分网络，对多变量通道分数取平均，得到每个 patch 的重要性分数。

5. **top-k 选择**
   根据 MLP 评分选择 top-k 重要 patch。

6. **LSTM 编码重要 patch**
   将选中的 patch 展平成 token 序列，投影到 `model_dim` 后输入 LSTM 编码器。

7. **Gated Attention**
   对 LSTM 隐状态计算注意力权重，得到上下文向量；随后使用 sigmoid gate 对注意力输出进行动态过滤。该设计对应 gated attention 中“在 SDPA/注意力输出后接 gate”的核心思想。

8. **全连接预测头**
   gated attention 输出经过 `LayerNorm -> Linear -> GELU -> Dropout -> Linear`，生成未来 `pred_len` 步的电耗预测。

## 训练与评估

训练逻辑位于 `exp/exp_main.py`。

- 损失函数：`L1Loss`。
- 优化器：`AdamW`。
- 早停指标：验证集 loss。
- 梯度裁剪：默认 `1.0`。
- 测试指标：`MAE`、`RMSE`、`MAPE`、`R2`。
- 指标计算前会将预测值和真实值从标准化空间还原到原始电耗单位。

## PatchGatedSegRNN

保留原模型的 RevIN、频域去噪、patch 选择、patch 投影和 LSTM 编码。区别是不用 `gated_attention + MLP head` 直接输出整段预测，而是把 LSTM 编码器的最终 hidden state 交给 SegRNN 风格的未来 segment 解码器，按 `seg_len` 逐段生成 residual，再和 NLinear baseline 相加。

约束：

- `pred_len` 必须能被 `seg_len` 整除。

## 快速运行

安装依赖：

```bash
pip install -r code/requirements.txt
```

先跑一个小规模 smoke test：

```bash
python code/run_experiment.py --epochs 1 --max-files 1 --limit-windows 256 --batch-size 32
```

运行默认实验：

```bash
python code/run_experiment.py
```

## 常用参数

```bash
python code/run_experiment.py \
  --model PatchGatedLSTM \
  --seq-len 168 \
  --pred-len 24 \
  --seg-len 24 \
  --patch-len 24 \
  --patch-stride 12 \
  --top-k-patches 8 \
  --hidden-size 64 \
  --model-dim 64 \
  --epochs 20 \
  --batch-size 64
```

运行方案 B：

```bash
MODEL=PatchGatedSegRNN bash code/scripts/building_energy.sh
```

## 输出文件

实验输出默认保存在 `code/outputs/`：

- `config.json`: 本次实验配置。
- `metrics.json`: 每栋建筑的测试指标和整体平均指标。
- `*_predictions.npz`: 每栋建筑的预测值和真实值，包含原始单位和标准化空间两种版本。
- `*_model.pt`: 每栋建筑训练得到的模型权重。

## 代码对应关系

- 数据预处理和滑窗数据集：`data_provider/data_loader.py`
- DataLoader 工厂入口：`data_provider/data_factory.py`
- 模型主体：`models/PatchGatedLSTM.py`
- 模型层实现：`layers/RevIN.py`、`layers/FrequencyDenoiser.py`、`layers/PatchSelector.py`、`layers/GatedAttention.py`
- 实验基类：`exp/exp_basic.py`
- 训练和评估实验：`exp/exp_main.py`
- 参数入口：`run_experiment.py`
- 实验配置：`utils/config.py`
