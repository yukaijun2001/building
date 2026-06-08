# 热插拔实验

这个目录放“可插拔基线 + patch-gated residual correction”的独立实验入口，不改动原来的 `PatchGatedLSTM.py` 主流程。

## 结构

- `hotplug_model.py`: 模型定义。baseline 只接收目标列历史 `[B, seq_len, 1]`，patch-gated residual 分支仍接收完整多变量输入。
- `hotplug_run.py`: 建筑能耗数据训练/验证/测试入口，复用 `data_provider/building_energy.py` 的 CSV 切分和标准化。
- `run_hotplug_tsl.sh`: 跑常规 baseline，包括 `NLinear`、`TimeXer`、`iTransformer`、`TimeMixer`、`PatchTST`。
- `run_hotplug_mapl_llm.sh`: 跑 `jie_project/components/models/times_llama.py` 作为 baseline，并从同名 `.pt` 读取 `x_mark_enc`，和 patch-gated residual 联合训练。

## TSL baseline 热插拔

```bash
cd /home/ykj/build/Time-Series-Library
bash 热插拔实验/run_hotplug_tsl.sh
```

常用覆盖参数：

```bash
BASELINES="NLinear PatchTST iTransformer" SEQ_LEN=672 PRED_LENS="24 48" EPOCHS=30 bash 热插拔实验/run_hotplug_tsl.sh
```

baseline 选择：

```bash
BASELINES="NLinear TimeXer iTransformer TimeMixer PatchTST"
```

输出会写到：

- `热插拔实验/results/`
- `热插拔实验/test_results/`
- `热插拔实验/outputs/source_<baseline>_<seq_len>_<pred_len>/`

只跑 baseline，不加 patch-gated residual：

```bash
DISABLE_RESIDUAL=1 BASELINES="NLinear TimeXer iTransformer TimeMixer PatchTST" bash 热插拔实验/run_hotplug_tsl.sh
```

## MaPL-LLM 桥接

## MaPL-LLM 热插拔

`MaPL_LLM` 会直接导入 `/home/ykj/build/jie_project/components/models/times_llama.py` 里的 `TimesLlama`，baseline 仍然只输入目标列历史 `[B, seq_len, 1]`。额外的 `x_mark_enc` 从同名 `.pt` 读取，例如：

```text
/home/ykj/build/jie_project/dataset/building/Hog_assembly_Colette.csv
/home/ykj/build/jie_project/dataset/building/Hog_assembly_Colette.pt
```

运行：

```bash
cd /home/ykj/build/Time-Series-Library
bash 热插拔实验/run_hotplug_mapl_llm.sh
```

默认只跑：

```text
/home/ykj/build/jie_project/dataset/building/Hog_assembly_Colette.csv
/home/ykj/build/jie_project/dataset/building/Hog_assembly_Colette.pt
```

常用覆盖参数：

```bash
DATA_PATH=Hog_assembly_Colette.csv PRED_LENS="24 48" EPOCHS=30 bash 热插拔实验/run_hotplug_mapl_llm.sh
```

注意：`TimesLlama` 会加载 `LLM_CKP_DIR`，默认是 `/home/ykj/build/llama_model`，显存占用会明显高于其他 baseline。

MaPL-LLM 只跑 baseline，不加 patch-gated residual：

```bash
DISABLE_RESIDUAL=1 bash 热插拔实验/run_hotplug_mapl_llm.sh
```
