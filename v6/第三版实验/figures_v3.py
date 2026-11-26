"""Figures for sound examples, test-set errors and propagation sensitivity."""
import sys
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import spectrogram

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT))
from figures_apmcm import arrow, box, save
from utils.plot_style import apply_publication_style

CALLS = ["flat","mod","tv"]
TYPES = ["平直型","调制型","双声型"]
MODELS = ["pava","spline","parametric"]
LABELS = ["保序插值","单调样条","三参数曲线"]
COLORS = ["#364B5D","#007992","#AC481F"]
MARKERS = ["o","s","D"]


def samples(folder):
    fig,axes=plt.subplots(2,3,figsize=(6.55,3.03),sharex=True,layout="none",
                          gridspec_kw={"height_ratios":[.55,1.]})
    # Set global layout once; use a common spectral scale and a dedicated colour-bar axis.
    fig.subplots_adjust(left=.095,right=.915,top=.88,bottom=.18,wspace=.22,hspace=.13)
    for j,suffix in enumerate(["flat","modulated","two-voice"]):
        f=next((ROOT/"date").glob(f"*_{suffix}.wav"))
        with wave.open(str(f)) as clip:
            assert (clip.getsampwidth(),clip.getnchannels(),clip.getframerate())==(2,1,12000)
            rate=clip.getframerate()
            signal=np.frombuffer(clip.readframes(clip.getnframes()),dtype="<i2").astype(float)
        signal/=np.max(np.abs(signal))
        sec=np.arange(len(signal))/rate
        axes[0,j].plot(sec,signal,color="#185C78",lw=.4)
        axes[0,j].set(ylim=(-1.05,1.05),yticks=[-1,0,1],title=TYPES[j])
        f,t,power=spectrogram(signal,fs=rate,window="hann",nperseg=256,noverlap=192,
                              nfft=256,detrend=False,scaling="spectrum",mode="psd")
        db=10*np.log10(np.maximum(power/power.max(),1.e-10))
        low=f<=3000
        mesh=axes[1,j].pcolormesh(t,f[low]/1000,db[low],shading="auto",cmap="cividis",vmin=-50,vmax=0)
        axes[1,j].set(xlim=(0,1.22),ylim=(0,3.),yticks=[0,1,2,3],xticks=[0,.5,1])
        if j:
            for row in axes: row[j].tick_params(labelleft=False)
    axes[0,0].set_ylabel("相对幅值")
    axes[1,0].set_ylabel("频率（kHz）")
    axes[1,1].set_xlabel("片段时间（s）")
    cb=fig.colorbar(mesh,cax=fig.add_axes([.935,.18,.013,.415]),ticks=[-50,-25,0])
    cb.ax.tick_params(labelsize=6)
    cb.ax.set_title("dB",fontsize=7)
    save(fig,folder,"raw_q1_example_waves")


def confusions(folder):
    counts=pd.read_csv(HERE/"results/q1v3_confusion.csv")
    fig,axs=plt.subplots(1,3,figsize=(6.55,2.62),layout="none")
    fig.subplots_adjust(left=.105,right=.986,top=.83,bottom=.24,wspace=.17)
    for j,(ax,model,label) in enumerate(zip(axs,["reference","TabICLv2","Mitra-v2"],["岭模型","TabICLv2","Mitra-v2"])):
        c=counts[counts.model==model].set_index("actual").loc[CALLS,CALLS].to_numpy()
        rates=c/c.sum(axis=1,keepdims=True)
        ax.pcolormesh(rates,vmin=0,vmax=.75,cmap="Blues",edgecolors="white",linewidths=1)
        ax.set(xticks=np.arange(3)+.5,xticklabels=TYPES,yticks=np.arange(3)+.5,
               yticklabels=TYPES if j==0 else [],ylim=(3,0),aspect="equal")
        ax.set_title(label,fontsize=9)
        for r,s in np.ndindex(c.shape):
            ax.text(s+.5,r+.5,str(c[r,s]),ha="center",va="center",fontsize=10,
                    color="white" if rates[r,s]>.48 else "#172838")
    axs[0].set_ylabel("实测声型")
    axs[1].set_xlabel("预测声型")
    save(fig,folder,"result_q1_confusion")


def ranges(folder):
    q2=pd.read_csv(HERE/"results/q2v3_ranges.csv").query("threshold==0.7").set_index("call")
    par=pd.read_csv(HERE/"results/q2v3_parametric_ranges.csv").query("threshold==0.7").set_index("call")
    q4=pd.read_csv(HERE/"results/q4v3_shape_transfer.csv").query("scenario=='Threshold 0.70'")
    fig,axs=plt.subplots(1,2,figsize=(6.55,2.91),layout="none")
    fig.subplots_adjust(left=.09,right=.986,bottom=.16,top=.735,wspace=.30)
    x=np.arange(3)
    for a,(ax,title) in enumerate(zip(axs,["a　声学有效距离","b　功能通信半径"])):
        for j,(model,txt) in enumerate(zip(MODELS,LABELS)):
            if a==0:
                series=(q2.pava_m,q2.spline_m,par.radius_m)[j]
            else:
                subset=q4[q4.model==model].set_index("call")
                assert (subset.status=="crossing").all()
                series=subset.radius_m
            ax.scatter(x+(j-1)*.15,series.loc[CALLS],s=30,marker=MARKERS[j],color=COLORS[j],label=txt,zorder=3)
        ax.set(xlim=(-.45,2.45),ylim=(0,150),xticks=x,xticklabels=TYPES,yticks=[0,30,60,90,120,150])
        ax.set_title(title,loc="left",fontsize=8)
        ax.grid(axis="y",alpha=.25)
    axs[0].set_ylabel("距离（m）")
    keys=[Line2D([],[],marker=m,color=c,linestyle="none",label=l) for m,c,l in zip(MARKERS,COLORS,LABELS)]
    fig.legend(handles=keys,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.52,.995))
    save(fig,folder,"result_q4_curve_shape")


def flow(folder):
    fig,ax=plt.subplots(figsize=(6.55,3.65),layout="none")
    fig.subplots_adjust(left=.01,right=.99,bottom=.012,top=.995)
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    lanes=[("108 条叫声测量","问题一　声型差异\n同日秩检验、三分类","留出录音日期\n核对每类的误判去向"),
           ("9 张传播相关表","问题二　传播劣化\n按三种表征拟合保真度","留出距离节点\n比较三种曲线的阈值交点"),
           ("139 对海上回放","问题三　群体行为\n八状态与首次响应时间","整对分折\n比较两部分模型与 Cox")]
    centers=[.16,.5,.84]
    for x,labels in zip(centers,lanes):
        for (y,h,kind),s in zip([(.83,.115,"data"),(.60,.16,"model"),(.38,.15,"model")],labels):
            box(ax,x-.15,y,.30,h,s,kind)
        arrow(ax,(x,.83),(x,.76)); arrow(ax,(x,.60),(x,.53))
    box(ax,.034,.17,.252,.12,"声型与声学线索","terminal")
    arrow(ax,(.16,.38),(.16,.29))
    box(ax,.36,.16,.57,.15,"问题四　结合传播与八状态概率\n相同距离、群体规模及行为情境")
    arrow(ax,(.50,.38),(.50,.31)); arrow(ax,(.84,.38),(.81,.31))
    box(ax,.36,.008,.57,.09,"功能半径、响应概率及噪声情景","terminal")
    arrow(ax,(.645,.16),(.645,.098))
    save(fig,folder,"flow_overall_model")


if __name__=="__main__":
    apply_publication_style("en","report")
    matplotlib.rcParams["figure.constrained_layout.use"]=False
    matplotlib.rcParams["figure.autolayout"]=False
    matplotlib.rcParams["font.sans-serif"]=["Arial Unicode MS","Noto Sans CJK SC","Hiragino Sans GB","DejaVu Sans"]
    folder=HERE/"figures"
    folder.mkdir(exist_ok=True)
    flow(folder)
    samples(folder)
    confusions(folder)
    ranges(folder)
