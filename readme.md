# 非洲企鹅海上声学通信 · 2026 APMCM C

这个仓库记录四问的研究主线和第 1 问的可复现分析。我们想知道三类叫声靠哪些声学特征区分，这些特征在远处保留多少，以及它们与企鹅群回叫或靠近有什么关系。

后续建模与论文写作以[团队确认的四问研究主线](research_plan.md)为入口。第 1 问已完成，模型成绩、检验结果及局限见[第 1 问自评](review_q1.md)。

## 四问的衔接

| 章节 | 采用的方案 | 读者能得到什么 |
|---|---|---|
| 1 · 声型 | 用六个共同特征、同日秩检验和逐日留出的分类比较 | 哪些测量在同一日期仍有差异，哪些声型换一天就容易认混 |
| 2 · 传播 | 对声谱图、幅度谱、包络分别建距离曲线。统一比较分段直线、指数、幂律、普通平滑、单调平滑、等序与中心等序七种实现 | 宽松和严格条件下的连续达标范围，以及阈值、权重、共享录音对结论的影响 |
| 3 · 回放 | 自然声与海浪对照按群体实验配对，比较八种动作组合。三种行为分别计时，补充首次任意行为，次数与持续时间另答 | 固定群体 8 只和 50、100、150 米时，各情境如何改变回应的类型、机会与等待时间 |
| 4 · 功能 | 将声学门槛与第 3 问的群体回应概率连接，分别用发声和接近作为联系、招募的代理 | 两种目标的条件性半径，噪声情景下的变化，以及参与个体比例假设的影响 |

总体结构借鉴 Loning 等将传播测量与接收者听觉联系起来的研究。该研究的物种为斑胸草雀。企鹅的三档识别标准、联系／招募评分与噪声情景由本队设定。三组实验按声型和条件衔接，缺少逐录音公共键。后续各章将使用附件的实际声型，第一章的预测误差用于解释这套分类能推广到哪里。[框架研究](https://doi.org/10.1093/beheco/arab107) · [企鹅题源研究的 Methods、Table 1 和 Discussion](https://ore.eu/article/view/1538)

Q2 的形状约束平滑和中心等序曲线分别借鉴 Pya 与 Wood、Oron 与 Flournoy 的方法，是否采用以留出距离的成绩决定。Q3 的八组合建模拟比较 Dai 等的联合二元框架与八类回归，发声主口径为播放后出现叫声，另以超过基线的发声作敏感性分析。Q4 相对基准比较噪声下通信范围的思路可参照 Hatch 等的鲸类研究。[形状约束](https://doi.org/10.1007/s11222-013-9448-7) · [中心等序回归](https://doi.org/10.1080/19466315.2017.1286256) · [多变量伯努利](https://doi.org/10.3150/12-BEJSP10) · [通信空间](https://doi.org/10.1111/j.1523-1739.2012.01908.x)

## 第 1 问的选择

1. **先比较声音。** 按三类统计样本量、中位数和四分位数，绘制特征分布。题源用 Kruskal–Wallis 总体检验与两两秩和检验。沿用队员第三版的改进，在同一日期内置换声型检验总体差异，再限定到当天同时出现两类声音的记录作精确两两比较。对六项总体检验和十八项成对检验分别作 Holm 校正。按日期比较还需要同日记录在零假设下可交换。[题源 Table 2](https://ore.eu/article/view/1538) · [条件置换的一般框架](https://doi.org/10.18637/jss.v028.i08) · [Holm 校正](https://www.jstor.org/stable/4615733)
2. **让分类器回答同一个问题。** 使用 Duration、F01、Fpeak、Q25、Q50、Q75。每次留出一整个日期，其他日期用于训练和必要的内部选参。基频 F02 只在双声型有值，平直型也缺少脉冲率，这两个字段作结构描述。日期、地点、结构性缺失标记均不进入六特征分类器。日期分组借鉴分块验证原则，具体折分是本题设计。[Roberts 等，2017](https://doi.org/10.1111/ecog.02881)
3. **六个候选平等比较。** 经典模型为岭惩罚多项逻辑回归、线性判别分析、浅决策树和支持向量机。Favaro 等在圈养非洲企鹅的另一组类别上使用过判别分析，本文以共同特征实现线性判别。队员还采用两个冻结的预训练表格分类器 TabICLv2、Mitra-v2。两篇模型论文说明算法，本题成绩以逐条折外预测为准。[Favaro 等的统计方法和表 4](https://doi.org/10.1371/journal.pone.0103460) · [TabICLv2 论文](https://proceedings.mlr.press/v306/qu26a.html) · [Mitra-v2 技术报告](https://arxiv.org/abs/2609.04540)
4. **先定评价，再解释误判。** 将每条录音恰好一次的折外预测合并，以宏平均 F1 为主排序，平衡准确率、对数损失和混淆矩阵并列。宏 F1 并列时参考平衡准确率。用同一组日期成对重抽样比较固定预测对日期组成的敏感性。另对表现好的模型删去一组声学测量、重复验证，评价在其余特征已知时该组提供多少额外信息。模型分数衡量的是新日期上复现人工声型标签的能力。

## 输入与执行

在 [Zenodo 版本 21376649](https://zenodo.org/records/21376649) 下载 `Data01_AfricanPenguins_seasurface_calls_acoustics.csv`，或使用比赛附件 `date` 中的同名文件。程序核验原始字节的 SHA-256 为 `1328f92601a84d81dbf8e1404fa0eeabc36cdf17a7626d9746aa62794827d02e`。结果中的 `row_id=1` 对应 CSV 表头下面第一条记录。

以下 PowerShell 示例从仓库根目录执行。`$data01` 替换为下载的 CSV 路径，Python 3.12 的隔离环境可以按 `requirements.txt` 安装基础依赖。

```powershell
$data01 = "data/Data01_AfricanPenguins_seasurface_calls_acoustics.csv"
python -m pip install -r requirements.txt
python -m q1 --data $data01
```

基础入口依次生成描述检验和四个经典分类器的结果。`--stage statistics` 或 `--stage classical` 可单独运行。结果目录默认为 `results/q1`，可用 `--out` 替换。冻结模型与合并评分的精确命令见下文。

```powershell
# 在隔离的 Python 3.12 环境中运行。两个冻结模型均支持 CPU。
python -m pip install -r requirements-foundation.txt
$weights = "../q1-model-cache"
python -m q1.foundation --data $data01 --cache $weights
python -m q1.compare --data $data01 results/q1/classical/oof.csv results/q1/foundation/oof.csv
python -m q1.foundation --data $data01 --cache $weights --model TabICLv2 --offline --ablate
python -m q1.svm_check --data $data01
```

首次运行会下载固定版本的检查点，并验证 SHA-256。本次使用 HTTPS 镜像传输模型文件，脚本默认采用官方地址，运行时可选择 `--hf-endpoint https://hf-mirror.com`。Mitra 的四个原生推理模块取自固定哈希的上游 wheel。权重、依赖和原始数据使用外部路径。[TabICLv2 运行清单](results/q1/foundation/run_TabICLv2.json) · [Mitra-v2 运行清单](results/q1/foundation/run_Mitra-v2.json)

若要复现自评中的岭模型删组对照，可从仓库根目录运行：

```powershell
python -c "from pathlib import Path; import sys; from q1.data import load_data; from q1.classifiers import ablate; ablate(load_data(Path(sys.argv[1])),Path('results/q1/classical'),'ridge')" $data01
```

**结果入口。** [六模型成绩](results/q1/comparison/scores.csv) · [各类混淆计数](results/q1/comparison/confusion.csv) · [六模型混淆图](results/q1/comparison/confusion.png) · [日期成对重抽样](results/q1/comparison/paired_date_sensitivity.csv) · [胜出模型删组结果](results/q1/foundation/ablations_TabICLv2.csv)。每个候选的 108 条测试概率见[传统模型逐条预测](results/q1/classical/oof.csv)和[冻结模型逐条预测](results/q1/foundation/oof.csv)。
