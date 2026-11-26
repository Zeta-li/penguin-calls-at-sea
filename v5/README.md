# 第五版实验使用说明

这套程序复现非洲企鹅叫声的四问分析。第一问按录音日期比较六项共有声学特征的分类能力，第二问按距离验证传播曲线，第三问按回放实验对评价行为发生概率，第四问求不同传播和行为假设共同支持的通信距离。

所有命令均从仓库根目录执行。与第五版实验包对应的程序、冻结概率和历史对照位于 `v5/`，仓库原有的 `python -m q1` 入口继续用于原研究方案。

## 下载代码和安装依赖

```bash
git clone --branch fivecheck --single-branch https://github.com/Zeta-li/penguin-calls-at-sea.git
cd penguin-calls-at-sea
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r v5/第四版实验/requirements.txt
```

以上为 macOS / Linux 命令。Windows PowerShell 创建环境用 `py -3.13 -m venv .venv`，激活用 `.\.venv\Scripts\Activate.ps1`，再执行 `$env:PYTHONUTF8 = "1"` 以统一中文清单的编码。实验的参考环境是 Python 3.13.5，数值计算与绘图依赖固定在[基础依赖清单](第四版实验/requirements.txt)。

## 准备数据

首次运行可直接从固定版本的 Zenodo 记录下载全部输入。

```bash
python v5/prepare_data.py --download
```

已有比赛附件时，也可以使用本地文件。把下列路径替换成实际存放 `Data01` 至 `Data11` 的 `date` 目录。脚本依据历史清单逐文件核验 SHA-256 并复制到 `v5/date/`。默认实验需要 11 个 CSV，复算第四版还需 3 个 `Supplementary5` WAV。CSV 请保留原始编码、文件名和换行。

```bash
python v5/prepare_data.py --from-dir "/path/to/2026 APMCM Problem C/date"
```

数据由 Andréa Thiebault 发布，许可为 CC BY 4.0。数据准备后，默认实验可在本地离线完成。[原始数据及音频](https://doi.org/10.5281/zenodo.21376649) · [许可](https://creativecommons.org/licenses/by/4.0/)

## 运行四问并核对数值

```bash
python v5/第五版实验/reproduce.py
python v5/verify_results.py
```

默认流程使用已记录、带校验值的 TabICLv2 前向概率，重新训练经典分类器，计算按日配对差、跨年成绩和后三问的数值。首次复现无需下载模型权重。生成的 34 张 CSV 与本次运行清单位于 `v5/第五版实验/results/`，新增图件位于 `v5/第五版实验/figures/`。校验成功时会显示 `Verified 34 fifth-edition result tables (SHA-256).` 字节校验以参考运行环境为准，其他平台如仅浮点末位不同，脚本也会列出相应文件。[参考哈希清单](reference/manifest.json)

| 问题 | `v5/第五版实验/results/` 中先看 | 核对什么 |
| --- | --- | --- |
| 一 · 按日期留出 | `q1v5_scores.csv`、`q1v5_comparisons.csv`、`q1v5_predictions.csv` | 原始 TabICLv2、双视图和嵌套分类器的平衡准确率依次为 **45.05%、56.79%、59.44%**。`raw_balanced` 是对原模型使用相同先验判决的对照。 |
| 一 · 跨年 | `q1t_scores.csv`、`q1c_temporal_scores.csv` | 2021 年训练、2022 年测试，对应三种方法为 28.64%、39.95%、47.82%。 |
| 二 · 传播 | `q2v5_held.csv`、`q2v5_scores.csv` | 平均保真度在 15 个留距点上的 RMSE，逐表征参数模型为 0.04449，混合模型为 0.04860。 |
| 三 · 回放 | `q3v5_probabilities.csv`、`q3v5_scores.csv` | 按 139 个实验对验证的平均 Brier 分数，两部分模型为 0.107034，混合模型为 0.107471。 |
| 四 · 通信距离 | `q4v5_scenarios.csv`、`q4v5_joint_support.csv` | 读取端点与状态。以平均保真度、阈值 0.70、无额外损失为例，双声型联系与招募的共同支持半径为 127.33 m、82.44 m。 |

分类实验的输入和先验判决约定见[比较方案](第五版实验/比较方案.md)。独立分类器的候选集合和选择次序见[预设方案](第五版实验/独立分类器预设方案.md)。

## 需要从头重算时

**重新拟合第四版对照。** 准备全部 CSV 和 3 个 WAV 后运行以下命令。程序在临时目录重建第四版的 35 个数值文件并核对，再运行第五版。

```bash
python v5/第五版实验/reproduce.py --rebuild-v4
```

**从权重重新运行 TabICLv2。** 安装可选依赖，把固定版本的 `tabicl-classifier-v2-20260212.ckpt` 放入 `v5/model_cache/`。程序核验权重，再重新推断 16 个留出日期及跨年测试，逐条与归档概率核对。这一步支持 CPU。[固定版本权重](https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt) · [权重与预测的 SHA-256](第五版实验/inputs/q1f_inputs.json) · [可选依赖](第二版实验/requirements.txt)

```bash
python -m pip install -r v5/第二版实验/requirements.txt
python v5/第五版实验/reproduce.py --foundation-cache v5/model_cache
```

只想检查经典分类器时，可运行 `python v5/第五版实验/classical.py`。它会同时生成逐日验证和跨年测试的结果。第一问的输入始终来自六项共有测量，日期只用于拆分数据。后三问使用实验原始声型标签。
