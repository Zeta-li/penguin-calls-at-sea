# 第七版四问复现指南

本分支包含原始数据、第五和第六版的历史对照，以及第七版的代码、逐条留出预测、汇总结果和图件。可先打开论文，对照下文的表名查数，或按顺序重新计算四问。[中文版 Word 论文](论文/完整论文_中文版_第七版.docx) · [结果表](results/) · [图件](figures/)

## 获取代码和安装环境

```bash
git clone --branch sevencheck --single-branch https://github.com/Zeta-li/penguin-calls-at-sea.git
cd penguin-calls-at-sea
```

后续命令均从仓库根目录执行。参考环境为 Python 3.13.5，依赖版本列于[数值与绘图依赖清单](../v6/第四版实验/requirements.txt)。在 macOS 或 Linux 上创建并激活虚拟环境。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r v6/第四版实验/requirements.txt
```

Windows PowerShell 可直接调用虚拟环境的解释器。后文命令中的 `python` 相应替换为 `.\.venv\Scripts\python.exe`。

```powershell
py -3.13 -m venv .venv
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pip install -r v6/第四版实验/requirements.txt
```

## 重算与校验

```bash
python v7/reproduce.py
python v7/verify_results.py
```

第一行重新生成第七版数值表和图件。第二行将复算文件与随分支发布的参考结果核对。建议在新克隆中操作，以便区分已发布结果和自己的实验修改。默认计算读取仓库中的数据与归档预测，依赖安装完成后即可离线运行。[统一运行入口](reproduce.py) · [结果校验程序](verify_results.py)

| 复算阶段 | 读取的数据与依赖关系 | 入口 |
| --- | --- | --- |
| 一、二问 | Data01 含 108 条录音的声学测量，Data03–11 含水库三声型、三表征的传播矩阵。归档提供前五、六版分类对照。只重算一问时，可运行 `python v7/q12.py --question 1` 或 `--question 2`。 | [声学分类与传播](q12.py) |
| 录音示例 | 三类声型各一段公开 WAV。对原声叠加已知频带和信噪比的噪声，测量表征变化。噪声重复沿用这三段波形，Data01 中的 108 条测量缺少逐行音档标识。 | [遮蔽实验](audio_probe.py) · [录音来源](audio/README.md) · [汇总](results/audio_summary.csv) |
| 第三问 | Data02 给出 139 对自然声与海浪对照的回放。程序读取第二问传播结果，比较外部传播曲线和纯距离弯曲带来的增量。 | [海上行为模型](q3.py) · [传给第四问的行为概率](results/q3_state_grid.csv) |
| 第四问与绘图 | 在第二问传播参数和第三问行为概率就绪后，第四问加入 Stony Point 船噪实验的三张岸上原表。全部数值完成后生成图件。 | [情景模型](q4.py) · [岸上数据与许可](data/noise_stony_point/README.md) · [绘图程序](figures.py) |

原题数据 Data01–11 和历史对照已置于本分支的 `v6/`。默认入口使用归档的 TabICLv2 对照概率；若要从模型权重重跑冻结模型，可按第五版说明单独执行。原题数据和岸上船噪数据均保留来源及开放许可。[海上与水库数据](https://doi.org/10.5281/zenodo.21376649) · [Stony Point 数据](https://doi.org/10.5281/zenodo.23923079) · [第五版独立复现](https://github.com/Zeta-li/penguin-calls-at-sea/blob/fivecheck/v5/README.md) · [第六版独立复现](https://github.com/Zeta-li/penguin-calls-at-sea/blob/sixcheck/v6/README.md)

## 按四问核对数值

**第一问 · 先区分留日成绩与跨年外推。** 在评分表筛选 `validation=date_holdout`，读取 `balanced_accuracy`。固定模型用六项对数测量得到 42.73%，加谐波特征为 64.07%，同时加谐波与谱形为 63.74%，第六版树为 59.44%。将验证改为 `year_forward`，同一双组特征树从 2021 年推到 2022 年仅得 38.99%，第六版树为 47.82%。双组特征提升了本数据的留日点估计；跨年结果仍受训练年份、月份和录音规模限制。[评分表](results/q1v7_scores.csv) · [日期配对区间](results/q1v7_paired.csv) · [匹配训练规模的跨期比较](results/q1v7_matched_calendar.csv) · [方法](evidence_q12.md)

**第二问 · 验证单位是距离。** 同时移除同一距离处的全部声型和表征，五个内部距离给出 15 个声型—距离预测点。在评分表取 `criterion=mean`，第六版独立参数曲线的 RMSE 为 0.04449，训练折内选择汇聚强度的新模型为 0.05213。因此主估计继续用独立曲线。双声型在平均保真度 0.50 的端点约 182 m，实测达标与失效节点夹在 128–256 m，逐一删去内部距离重拟合得到 180–199 m。[留距评分](results/q2v7_scores.csv) · [逐点预测](results/q2v7_held.csv) · [端点与相邻实测距离](results/q2v7_ranges.csv)

**第三问 · 同一事件概率约束发生、时延和强度。** 把每对自然声和海浪对照放入同一验证折，按 278 次刺激评分。第六版中有 83 次的“叫数净增”预测概率高于“出现任意鸣叫”，第七版的 `selected` 模型为 0 次。三行为、四时点的平均 Brier 从 0.107017 变为 0.106165，七减六差值的 95% 配对区间为 −0.003615 至 +0.001666。可以确证的是概率之间恢复了子集关系，现有样本对评分改善的幅度仍有不确定性。[一致性审计](results/q3_coherence.csv) · [逐项评分](results/q3_scores.csv) · [跨实验对照](results/q3_curvature_scores.csv) · [方法](evidence_q3.md)

**第四问 · 分开核对岸上预测与海上情景。** 岸上 67 只企鹅在 18 段视频中有完整的三期观测。视频等权得到船噪播放中的非警戒时间机会相对损失 **0.5491**。留一视频预测播放中的警戒比例，维持原先警戒水平的 RMSE 为 **0.4773**，采用统一损失为 **0.1391**。这两个数评价岸上视频的预测。[岸上时间预算](results/q4v7_shore_budget.csv) · [逐视频评分](results/q4v7_shore_scores.csv)

海上船噪的迁移比例 η 由情景指定。半径表选择 `acoustic=Threshold 0.70`、`call=tv`、`endpoint=recruitment`、`clock=interval_mle`，依次查看 `eta` 为 0、0.5、1 的行。8 只群体的双声型招募连续达标端点约 **101.9、101.9、99.4 m**。η=1 表示把全部岸上机会损失移用至海上；这些米数衡量给定情景下企鹅至播放声源的距离，现场船噪效应尚待海上观测验证。[半径与边界状态](results/q4v7_radii.csv) · [与第六版比较](results/q4v7_v6_comparison.csv) · [条件时钟留出评分](results/q4v7_clock_scores.csv) · [方法](evidence_q4.md)
