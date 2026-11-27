# Stony Point 船噪回放数据

三张 CSV 来自 Andréa Thiebault 在 Zenodo 发布的 *African penguins' behaviour is affected by motor boat airborne noise: an experimental study on wild penguins*。第七版保留数据库的原始字节、列名和分号分隔格式，使用 CC BY 4.0 许可。数据与船噪实验论文可通过 DOI 获取。[Zenodo 数据记录](https://doi.org/10.5281/zenodo.23923079) · [开放许可](https://creativecommons.org/licenses/by/4.0/) · [实验论文](https://doi.org/10.1016/j.scitotenv.2026.182372)

| 文件 | 记录单位 | 在第四问中的用途 |
| --- | --- | --- |
| DataTable_analysis1.csv | 各视频组人数随时点改变 | 与另外两表核对视频和噪声标签 |
| DataTable_analysis2.csv | 各视频中的个体反应得分 | 交叉核对视频与噪声标签 |
| DataTable_analysis3.csv | 播放前、中、后每只个体的行为扫描 | 用七类行为的扫描次数计算警戒时间份额，按个体和视频汇总 |

海上组仍来自比赛的 Data02。模型把岸上的时间预算变化作为可调的外部参照。运行 `python v7/q4.py` 会先核验三张原表的 SHA-256。
