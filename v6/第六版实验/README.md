# 第六版实验使用说明

本目录配套中文版第六版论文。先看[评审回应与结果](评审回应与结果.md)可了解哪些结果有折外验证，哪些距离只在参数设定下成立。

## 在 sixcheck 分支复现

从仓库根目录执行下列命令。原始观测、早期版本对照和冻结模型预测已随分支归档。首次安装、Windows 命令和四问结果索引见[第六版复现指南](../README.md)。

```bash
python -m pip install -r v6/第四版实验/requirements.txt
python v6/第六版实验/reproduce.py
python v6/verify_results.py
```

依赖就绪后，整个默认入口可离线运行。第一问重拟合已选的极端随机树、重新计算分布检验与线索替换，基础模型对照读取相同日期的归档概率。第四问既对齐旧行为系数以隔离门控公式，也用第六版共享斜率模型重新拟合和重抽样。要从模型权重另外复查第五版冻结基础模型，可使用[第五版复算说明](../../v5/README.md)。公开观测来自 Andréa Thiebault 发布的数据集，原始值、来源与派生计算可分别追溯。[数据记录](https://doi.org/10.5281/zenodo.21376649) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## 输出和复核入口

完整运行后，程序将数值表与运行清单写入本目录的 `results/`，图写入 `figures/`。清单记录软件版本、输入、历史对照、源码和数值结果的 SHA-256。仓库中的独立参考文件可用于逐张核验 43 张数值表。图像的字体由本机环境决定。[运行清单](results/manifest.json) · [参考哈希](../reference/numeric_outputs.json)

| 想核对的结果 | 程序 | 数据表 |
| --- | --- | --- |
| 第一问的留日、跨年及双声型误报 | [声学与传播分析](acoustic_propagation.py) | [分类结果](results/q1v6_classification.csv)、[日期分层 KS](results/q1v6_ks.csv)、[同日交换线索](results/q1v6_donor_loss.csv) |
| 第二问 9 条曲线和三个声学门槛 | [声学与传播分析](acoustic_propagation.py) | [参数](results/q2v6_parameters.csv)、[留距预测](results/q2v6_held.csv)、[模型评分](results/q2v6_scores.csv)、[半径与实测节点](results/q2v6_ranges.csv)、[缺格完成外界](results/q2v6_completion_pava.csv) |
| 分类线索加权后能否改善海上行为预测 | [跨实验检验](cue_transfer.py) | [传播权重](results/q2v6_weights.csv)、[逐次回放概率](results/q3cue_oof.csv)、[配对差及区间](results/q3cue_scores.csv) |
| 第三问的强度、首发、斜率交互及稀疏格 | [行为总入口](behavior.py)、[时延与状态模型](q3_timing.py) | [强度评分](results/q3v6_intensity_scores.csv)、[强度情景及实测支持](results/q3v6_intensity_scenarios.csv)、[时延评分](results/q3v6_timing_scores.csv)、[时延情景](results/q3v6_timing_scenarios.csv)、[首发与强度关联](results/q3v6_onset_intensity_association.csv) |
| 复核修正门控公式是否还原海上基准 | [功能模型与归因比较](ecology.py) | [逐情景新旧公式](results/q4v6_radii.csv)、[固定公式更换行为设计](results/q4v6_behavior_slopes.csv) |
| 第四问最终主结果（λ=2）及不确定性 | [主模型及重抽样](primary_intervals.py) | [各曲线和机制端点](results/q4v6_primary_specifications.csv)、[整对重抽样](results/q4v6_primary_paired_intervals.csv)、[缺格投影](results/q4v6_primary_pava_completion.csv)、[联合参数敏感性](results/q4v6_primary_joint_summary.csv)、[群体人数情景](results/q4v6_primary_responders.csv) |

复核功能距离时同时查看数值列与 `status`。`crossing` 表示域内有交点，`no_segment_at_lower_bound` 表示从 20 m 起无连续达标段，`right_boundary` 表示至少延至 150 m。数值列在后两种状态分别按 20、150 编码。`q4v6_primary_*` 对应将第三问调参结果贯彻到功能距离的主分析，较早的 `q4v6_radii.csv` 与 `q4v6_paired_intervals.csv` 保留旧行为系数 λ=0.5，用于隔离门控修正。阈值 0.70 的双声型招募在这两组系数下分别约为 102 m 和 113 m。

各专题还配有[声学与传播的详细方法](声学与传播证据.md)、[强度和时延的详细方法](第三问证据.md)及[跨实验比较方案](跨实验比较方案.md)。
