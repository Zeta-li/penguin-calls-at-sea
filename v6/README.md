# 第六版四问复现指南

本分支随附 11 张原始 CSV、前五版实验用到的历史对照，以及第六版 43 张数值结果表和 6 幅图。以下命令均从仓库根目录执行。依赖安装完成后，四问可离线复算。

## 1. 克隆代码并安装环境

```bash
git clone --branch sixcheck --single-branch https://github.com/Zeta-li/penguin-calls-at-sea.git
cd penguin-calls-at-sea
```

参考运行使用 Python 3.13.5。macOS 或 Linux 可用下列命令创建虚拟环境。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r v6/第四版实验/requirements.txt
```

Windows PowerShell 可直接调用虚拟环境内的 Python。

```powershell
py -3.13 -m venv .venv
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pip install -r v6/第四版实验/requirements.txt
```

后续命令中的 `python`，在上述 PowerShell 环境下使用 `.\.venv\Scripts\python.exe`。六项基础依赖的版本固定在[依赖清单](第四版实验/requirements.txt)。

## 2. 重算并校验

```bash
python v6/第六版实验/reproduce.py
python v6/verify_results.py
```

第一行重新生成第六版的数值表和图，按第一、二问声学分析、第三问行为分析、跨实验检验、第四问功能距离及绘图的顺序运行。输入包含 `v6/date/` 的 Data01（声学测量）、Data02（海上回放）、Data03–11（水库传播）以及 `v6/` 下的历版模型预测和参数。保留整个目录结构即可。原始 CSV 来自 Andréa Thiebault 的公开数据记录，沿用原始文件名和字节，按 CC BY 4.0 使用。[数据记录](https://doi.org/10.5281/zenodo.21376649) · [许可](https://creativecommons.org/licenses/by/4.0/)

第二行逐张核对 43 张 CSV 的 SHA-256。克隆后即可查看归档结果，复算后再运行校验可检验自己的输出。数值表、逐条预测及运行清单位于 [results/](第六版实验/results)，彩色 PNG、SVG 和灰度预览位于 [figures/](第六版实验/figures)。清单记录依赖版本、输入及历史对照、分析代码和数值表的哈希值。字节校验对浮点末位和换行均敏感；如遇差异，可先核对 Python、依赖版本，再用 Git 比较差异文件与提交的参考表。[运行清单](第六版实验/results/manifest.json)

## 3. 按问题核对结果

| 问题 | 运行内容和验证方式 | 表内可核对的参考值 |
| --- | --- | --- |
| 一 · 声型 | 对 108 条录音按 16 个日期留出，仅用持续时间、第一基频、峰频和三个频谱分位数。按第五版各折选定的参数重新拟合极端随机树，与归档的留出概率逐条核对。TabICLv2 与跨年分类得分由归档预测重算。 | 在 `date_holdout` 行读取 `balanced_accuracy`。树模型 `classical_nested` 为 **59.44%**，双视图 TabICLv2 `blend_balanced` 为 **56.79%**。2021 年训练、2022 年测试时，树模型降至 **47.82%**。平直型与双声型第一基频的日期内置换检验经 Holm 校正后 *p*=0.0135。[分类](第六版实验/results/q1v6_classification.csv) · [分布检验](第六版实验/results/q1v6_ks.csv) |
| 二 · 传播 | 对三类声型分别拟合谱图、幅度谱、包络随距离的衰减。三个声型各留出五个内部距离，合计 15 个预测点。 | 选 `criterion=mean`、`call=pooled`，参数曲线 RMSE 为 **0.04449**，保序曲线为 **0.06819**。阈值 0.50、三表征平均达标时，平直、调制和双声型的参数曲线分别给出约 **88、149、182 m**。这五个距离也参与了模型选择。[留距评分](第六版实验/results/q2v6_scores.csv) · [达标距离](第六版实验/results/q2v6_ranges.csv) |
| 三 · 回放行为 | 以自然声和海浪对照组成的 139 对回放为单位做五折验证，预测行为出现概率、出现后的强度及首发时间。一次自然声及其对照始终在同一折。 | 在 `field=HeadsUpDuration`、`population=all` 下，`model=shared` 相对 `reference=stimulus_mean` 将抬头时长 RMSE 从 **12.02 降至 11.26 秒**，差值的 95% 整对区间为 **−1.50 至 −0.03 秒**。首发时间的平均 Brier 为共享斜率 **0.107017**、声型交互 **0.109860**。[四项强度](第六版实验/results/q3v6_intensity_scores.csv) · [首发时间](第六版实验/results/q3v6_timing_scores.csv) · [跨实验增量](第六版实验/results/q3cue_scores.csv) |
| 四 · 功能距离 | 用净增鸣叫度量联系，用确认靠近度量招募。将第三问的行为模型与第二问的衰减曲线相连，固定传播和门控设定，按完整回放对重抽样 400 次。 | 取 8 只、双声型、平均保真度阈值 0.70、无额外损失。只更新传播门控且沿用旧行为参数，招募约 **113 m**；使用第三问选定的行为参数后，主估计约 **102 m**，联系为 **≥150 m**。招募端点在 20–150 m 域内截断后的 2.5% 和 97.5% 分位数约为 **23 和 128 m**。[门控对照](第六版实验/results/q4v6_radii.csv) · [主模型](第六版实验/results/q4v6_primary_specifications.csv) · [重抽样](第六版实验/results/q4v6_primary_paired_intervals.csv) |

第四问主表中可用 `criterion=mean`、`scenario=Threshold 0.70`、`mechanism=shared`、`model=parametric`、`call=tv` 定位上述行。招募行 `endpoint=recruitment_approach` 的 `clipped_m` 为 101.876；联系行的 `status=right_boundary` 表示达标延伸至观测上限 150 m。上述区间反映回放实验对的抽样波动；传播缺失值和模型设定的影响另有独立结果表。[缺失值范围](第六版实验/results/q4v6_primary_pava_completion.csv) · [设定敏感性](第六版实验/results/q4v6_primary_joint_summary.csv)

四问的脚本调用顺序可直接查阅[统一入口](第六版实验/reproduce.py)。需要复查 TabICLv2 权重推断时，可另行使用[第五版冻结模型复算说明](https://github.com/Zeta-li/penguin-calls-at-sea/blob/fivecheck/v5/README.md)。论文逐问采用哪些结果、哪些比较仍有不确定性，见[第六版评审回应](第六版实验/评审回应与结果.md)。
