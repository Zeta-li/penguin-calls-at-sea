# 非洲企鹅海上声学通信 · 2026 APMCM C

`sixcheck` 收录第六版四问的代码、数据和实验结果。要复现实验，从[中文复现指南](v6/README.md)开始。运行环境准备好后，从仓库根目录执行以下两行即可重算四问、核对 43 张结果表。

```bash
python v6/第六版实验/reproduce.py
python v6/verify_results.py
```

| 问题 | 验证或估计对象 | 查看结果 |
| --- | --- | --- |
| 一 · 三类叫声如何区分 | 六项共有测量的留日分类、跨年泛化和日期内分布差异 | [分类成绩](v6/第六版实验/results/q1v6_classification.csv) |
| 二 · 声音能传播多远 | 三种表征在留出距离上的预测误差及达标距离 | [留距评分](v6/第六版实验/results/q2v6_scores.csv) · [传播端点](v6/第六版实验/results/q2v6_ranges.csv) |
| 三 · 海上群体怎样响应 | 完整回放实验对留出后的行为强度和首发时间 | [强度误差](v6/第六版实验/results/q3v6_intensity_scores.csv) · [时延评分](v6/第六版实验/results/q3v6_timing_scores.csv) |
| 四 · 联系和招募能维持多远 | 指定传播、行为与噪声情景下的功能半径及重抽样区间 | [主模型半径](v6/第六版实验/results/q4v6_primary_specifications.csv) · [端点区间](v6/第六版实验/results/q4v6_primary_paired_intervals.csv) |

四问的方法改动、比较基准和证据限制见[第六版逐问评审回应](v6/第六版实验/评审回应与结果.md)。此前的设计和运行说明可查阅[四问研究方案](research_plan.md)、[第一问自评](review_q1.md)、[原研究入口及命令](https://github.com/Zeta-li/penguin-calls-at-sea/blob/main/readme.md)和[第五版独立分支](https://github.com/Zeta-li/penguin-calls-at-sea/tree/fivecheck)。
