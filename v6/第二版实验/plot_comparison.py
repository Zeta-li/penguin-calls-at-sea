"""Figures for the second edition, drawn from held-out predictions only."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figures_apmcm import arrow, box, save
from utils.plot_style import apply_publication_style


def flow(folder):
    fig, ax = plt.subplots(figsize=(6.55, 3.62), layout="none")
    fig.subplots_adjust(left=.006, right=.994, top=.996, bottom=.004)
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.axis("off")
    lanes = [
        ("108 条叫声测量", "问题一　声型分类\n六项共测声学特征", "逐日期留出\n岭模型与两种基础模型"),
        ("9 张传播相关表", "问题二　结构传播\n三种表征的单调拟合", "留出距离节点\n工作分块重抽样"),
        ("139 对海上回放", "问题三　群体行为\n八种组合与条件潜伏期", "整对分折\n基础模型复核任一响应"),
    ]
    centers = [.16,.50,.84]
    for center, lane in zip(centers, lanes):
        for (y,h,kind), label in zip([(.83,.115,"data"),(.61,.15,"model"),(.39,.15,"model")],lane):
            box(ax,center-.145,y,.29,h,label,kind)
        arrow(ax,(center,.83),(center,.76))
        arrow(ax,(center,.61),(center,.54))
    box(ax,.04,.185,.24,.105,"声型的可区分性","terminal")
    arrow(ax,(.16,.39),(.16,.29))
    box(ax,.36,.16,.57,.15,"问题四　同距离下综合保真度与行为\n按情境比较自然声及海浪声")
    arrow(ax,(.50,.39),(.50,.31))
    arrow(ax,(.84,.39),(.81,.31))
    box(ax,.36,.012,.57,.091,"功能半径、群体响应概率与个体数情景","terminal")
    arrow(ax,(.645,.16),(.645,.103))
    save(fig,folder,"flow_overall_model")


def comparison(folder):
    data = pd.read_csv(HERE/"results/q1_oof.csv")
    model_names = ["reference","TabICLv2","Mitra-v2"]
    legend = ["岭模型","TabICLv2","Mitra-v2"]
    palette = ["#565F67","#006C91","#A44015"]
    markers = ["o","s","D"]
    y = data.actual.to_numpy()
    recalls = []
    for name in model_names:
        p = data[[f"{name}_{k}" for k in range(3)]].to_numpy()
        data[name] = -np.log(p[np.arange(len(y)),y])
        recalls.append([np.mean(p.argmax(1)[y==k] == k) for k in range(3)])
    dates = data.groupby("group",sort=True)[model_names].mean()
    n_per_date = data.groupby("group",sort=True).size()
    fig, (left,right) = plt.subplots(1,2,figsize=(6.55,3.6),layout="none",
                                      gridspec_kw={"width_ratios":[.98,1.22]})
    fig.subplots_adjust(left=.085,right=.98,bottom=.16,top=.80,wspace=.57)
    for i in range(3):
        left.scatter(np.arange(3)+(i-1)*.13,recalls[i],s=25,marker=markers[i],color=palette[i],zorder=3)
    left.set(xticks=np.arange(3),xticklabels=["平直型","调制型","双声型"],ylim=(-.035,1.04),
             yticks=[0,.25,.5,.75,1],yticklabels=["0","25","50","75","100"],ylabel="召回率（%）",xlim=(-.5,2.5))
    left.set_title("a　各声型的识别率",loc="left",fontweight="bold")
    left.grid(axis="y",alpha=.22)
    for i in [1,2]:
        diff = dates[model_names[i]] - dates.reference
        right.scatter(diff,np.arange(len(dates))+(i-1.5)*.22,s=16,
                      marker=markers[i],color=palette[i],zorder=3)
    right.axvline(0,color=palette[0],lw=.85)
    show = sorted(set(range(0,len(dates),2)) | {len(dates)-1})
    right.set(yticks=show,
              yticklabels=[f"{d[2:]}（{n_per_date[d]}）" for d in dates.index[show]],
              xlabel="日期内平均对数损失差",ylim=(len(dates)-.35,-.65))
    right.tick_params(axis="y", labelsize=6.8)
    right.set_title("b　逐日期与岭模型的差值",loc="left",fontweight="bold")
    right.grid(axis="x",alpha=.22)
    keys=[Line2D([],[],marker=markers[i],linestyle="none",color=palette[i],label=legend[i]) for i in range(3)]
    left.legend(handles=keys,ncol=1,loc="upper right",frameon=False,handletextpad=.3)
    save(fig,folder,"result_q1_foundation_models")


if __name__ == "__main__":
    apply_publication_style("en","report")
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS","Noto Sans CJK SC","Hiragino Sans GB","DejaVu Sans"]
    destination = HERE/"figures"
    destination.mkdir(exist_ok=True)
    flow(destination)
    comparison(destination)
