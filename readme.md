# 非洲企鹅海上声学通信 · 2026 APMCM C

`sevencheck` 收录第七版中文版论文、四问的实验代码、逐条预测和结果图。完整复现使用本分支随附的海上录音测量、水库传播数据、海上回放记录及岸上船噪实验数据。[中文版 Word 论文](v7/论文/完整论文_中文版_第七版.docx) · [中文复现指南](v7/README.md)

在 Python 3.13 环境中，从仓库根目录运行以下命令。

```bash
python -m pip install -r v6/第四版实验/requirements.txt
python v7/reproduce.py
python v7/verify_results.py
```

| 问题 | 可以在结果中核对什么 | 结果入口 |
| --- | --- | --- |
| 一 · 声型分类 | 固定树与判决规则后，谐波和谱形特征怎样改变留日、跨年分类表现 | [分类成绩](v7/results/q1v7_scores.csv) · [同日配对差值](v7/results/q1v7_contrasts.csv) |
| 二 · 传播距离 | 整个距离留出时，独立曲线与跨声型共享曲线各有多准，哪些距离能夹住阈值交点 | [留距评分](v7/results/q2v7_scores.csv) · [达标距离](v7/results/q2v7_ranges.csv) |
| 三 · 海上群体反应 | 由同一组发生概率预测首发时间和强度，检查净增鸣叫概率能否与任意鸣叫概率相容 | [留出评分](v7/results/q3_scores.csv) · [概率一致性](v7/results/q3_coherence.csv) |
| 四 · 噪声与功能距离 | 岸上视频能否预测警戒时间，以及假定部分或全部效应迁移到海上时的联系和招募距离 | [岸上留一视频评分](v7/results/q4v7_shore_scores.csv) · [海上情景半径](v7/results/q4v7_radii.csv) |

第五版和第六版各有独立分支，可分别复核前期实验。[第五版 fivecheck](https://github.com/Zeta-li/penguin-calls-at-sea/tree/fivecheck) · [第六版 sixcheck](https://github.com/Zeta-li/penguin-calls-at-sea/tree/sixcheck)
