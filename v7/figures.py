"""Rebuild the quantitative figures used in the seventh-edition manuscript."""
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1795737600")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "v6"
sys.path.insert(0, str(BASE))
from utils.export_figure import export_figure
from penguin_data import read_inputs

RES = HERE / "results"
OUT = HERE / "figures"
COLORS = ("#176B75", "#B35F2B", "#525A91")
plt.rcParams.update({"svg.hashsalt": "penguin-seven", "font.family": "DejaVu Sans",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.alpha": .16})


def finish(fig, name, size):
    fig.tight_layout(pad=1.1)
    export_figure(fig, str(OUT / name), formats=("png", "svg"), dpi=300,
                  size_inches=size, grayscale_preview=True, tight=False)
    plt.close(fig)


def study_design():
    fig,ax=plt.subplots(figsize=(7.2,2.45))
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    style=dict(boxstyle="round,pad=.35",edgecolor="#3F656D",facecolor="#F4F8F7",lw=.8)
    boxes=[(.11,.76,"108 calls\n16 dates"),(.425,.76,"Reservoir\n7 distances"),
           (.76,.76,"At-sea playback\n139 trial pairs"),
           (.11,.26,"Call structure\nheld-out dates"),(.425,.26,"Signal fidelity\nheld-out distances"),
           (.76,.26,"Response & timing\nheld-out trial pairs")]
    for x,y,t in boxes:
        ax.text(x,y,t,ha="center",va="center",fontsize=9,bbox=style)
    for x in (.11,.425,.76):
        ax.annotate("",(x,.42),(x,.59),arrowprops=dict(arrowstyle="->",color="#3F656D"))
    ax.annotate("",(.30,.26),(.23,.26),arrowprops=dict(arrowstyle="->",color="#3F656D"))
    ax.annotate("",(.634,.26),(.553,.26),arrowprops=dict(arrowstyle="->",color="#3F656D"))
    ax.text(.98,.76,"Shore\nnoise",ha="center",va="center",fontsize=8,color="#914A28")
    ax.annotate("",(.91,.42),(.97,.6),arrowprops=dict(arrowstyle="->",color="#914A28"))
    ax.text(.94,.2,"Functional\nrange",ha="center",va="center",fontsize=8,color="#914A28")
    ax.annotate("",(.91,.24),(.88,.24),arrowprops=dict(arrowstyle="->",color="#914A28"))
    finish(fig,"flow_overall_model",(7.2,2.45))


def raw_features():
    obs, _, _ = read_inputs()
    fig, axes = plt.subplots(2, 3, figsize=(7, 3.7))
    for k, (ax, feature) in enumerate(zip(axes.flat, ("Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"))):
        for i, (call, color) in enumerate(zip(("flat", "mod", "tv"), COLORS)):
            y = obs.loc[obs.CallType.eq(call), feature].to_numpy()
            x = np.random.default_rng(20 + i).uniform(i-.14, i+.14, size=len(y))
            ax.scatter(x, y, s=6, color=color, alpha=.5, linewidths=0)
            ax.hlines(np.median(y), i-.23, i+.23, color="#222222", linewidth=1.3)
        ax.set(xticks=[0, 1, 2], xticklabels=["Flat", "Mod.", "Two-voice"], title=feature,
               ylabel="s" if k == 0 else "Hz")
    finish(fig,"raw_q1_calls",(7,3.7))


def classification_and_coverage():
    scores = pd.read_csv(RES / "q1v7_scores.csv")
    coverage = pd.read_csv(RES / "q1v7_selective.csv").query("method=='inner_plain'")
    fig, axes = plt.subplots(1, 2, figsize=(7,2.9))
    models = ["v6_tree","fixed_both","inner_plain","inner_aug"]
    for k,(regime,label) in enumerate([("date_holdout","Held-out date"),("year_forward","Later year")]):
        data=scores.query("validation==@regime").set_index("method").loc[models]
        axes[0].plot(range(len(models)),data.balanced_accuracy,"o-" if k==0 else "s--",
                     color=COLORS[k],label=label)
        group=coverage.query("validation==@regime").sort_values("coverage")
        axes[1].plot(group.coverage,group.accuracy_accepted,"o-" if k==0 else "s--",color=COLORS[k],label=label)
    axes[0].set(ylabel="Balanced accuracy",xticks=range(4),xticklabels=["V6","Fixed","Nested","Aug."],ylim=(.25,.8))
    axes[1].set(xlabel="Fraction classified automatically",ylabel="Accuracy on accepted calls",xlim=(.25,1.05),ylim=(.25,.9))
    axes[0].legend(fontsize=8)
    finish(fig,"result_q1_classification",(7,2.9))


def waveform():
    df=pd.read_csv(RES/"audio_summary.csv")
    frame=df.query("call=='two-voice'")
    fig,axes=plt.subplots(1,3,figsize=(7,2.35),sharey=True)
    for ax,view in zip(axes,("spectrogram","spectrum","envelope")):
        for i,(band,sub) in enumerate(frame.query("view==@view").groupby("band_hz")):
            sub=sub.sort_values("snr_db")
            ax.plot(sub.snr_db,sub["median"],"o-" if i==0 else "s--",label=f"{band} Hz",color=COLORS[i],ms=3)
            ax.fill_between(sub.snr_db,sub.p10,sub.p90,color=COLORS[i],alpha=.12)
        ax.set(title=view.capitalize(),xlabel="Digital SNR (dB)",xticks=[-6,0,6,12],ylim=(0,1.03))
    axes[0].set_ylabel("Correlation with original")
    axes[2].legend(fontsize=6.4,loc="lower right")
    finish(fig,"process_q2_masking",(7,2.35))


def propagation():
    nodes=pd.read_csv(RES/"q2v7_nodes.csv").query("model=='v6_parametric'")
    fig,axes=plt.subplots(1,3,figsize=(7,2.6),sharey=True)
    for ax,call,title in zip(axes,("flat","mod","tv"),("Flat","Modulated","Two-voice")):
        for i,(key,label) in enumerate([("mean","Mean"),("all","Minimum")]):
            z=nodes.query("call==@call and criterion==@key").sort_values("distance_m")
            ax.plot(z.distance_m,z.fitted,color=COLORS[i],label=label)
            ax.scatter(z.distance_m,z.measured,s=13,marker="o" if i==0 else "s",facecolor="white",edgecolor=COLORS[i],zorder=3)
        for tau in (.5,.6,.7):ax.axhline(tau,color="gray",ls=":",lw=.6)
        ax.set(xscale="log",xticks=[4,16,64,256],xticklabels=["4","16","64","256"],
               xlabel="Distance (m)",title=title,ylim=(-.06,1.02))
        ax.minorticks_off()
    axes[0].set_ylabel("Correlation");axes[2].legend(fontsize=7)
    finish(fig,"result_q2_nodes",(7,2.6))


def baseline_states():
    _,obs,_=read_inputs()
    fig,axes=plt.subplots(1,2,figsize=(7,2.65))
    bits=obs[["HeadsUpYesNo","VocalYesNo","ApproachYesNo"]].to_numpy(dtype=int)@np.array([4,2,1])
    counts=pd.crosstab(bits,obs.Type).reindex(range(8),fill_value=0)
    for i,(k,label) in enumerate([("call","Natural call"),("control","Wave control")]):
        options=obs.Type.unique()
        key=k if k in options else next(x for x in options if x!="call")
        axes[0].bar(np.arange(8)+(i-.5)*.37,counts[key],.36,color=COLORS[i],label=label)
    axes[0].set(xticks=range(8),xticklabels=[f"{x:03b}" for x in range(8)],xlabel="State (head / net call / approach)",ylabel="Stimuli")
    axes[0].tick_params(axis="x",labelsize=7);axes[0].legend(fontsize=7)
    for i,(field,label) in enumerate([("VocalNcalls","Calls after"),("VocalRelNCalls","Net change > 0")]):
        cats=np.where(obs.VocalNcallsBefore>0,"Active","Silent")
        vals=(obs[field]>0).groupby(cats).mean().reindex(["Silent","Active"])
        axes[1].bar(np.arange(2)+(i-.5)*.34,vals,.33,color=COLORS[i],label=label)
    axes[1].set(xticks=[0,1],xticklabels=["Silent before","Active before"],ylabel="Fraction observed",ylim=(0,1))
    axes[1].legend(fontsize=7)
    finish(fig,"raw_q3_joint",(7,2.65))


def vigilance():
    video=pd.read_csv(RES/"q4v7_shore_videos.csv").dropna(subset=["Playback"])
    clocks=pd.read_csv(RES/"q4v7_clocks.csv")
    fig,axes=plt.subplots(1,2,figsize=(7,2.8))
    for row in video.itertuples():
        axes[0].plot([0,1,2],[row.Pre_playback,row.Playback,row.Post_playback],color="#A6B8BC",lw=.7)
    mean=video[["Pre_playback","Playback","Post_playback"]].mean().to_numpy()
    axes[0].plot(range(3),mean,"o-",color=COLORS[0],lw=2,ms=4)
    axes[0].set(xticks=[0,1,2],xticklabels=["Before","Noise","After"],ylim=(0,1.03),
                ylabel="Vigilance share",title="Onshore: 18 videos")
    for i,(k,label) in enumerate([("contact","Net call"),("recruitment","Approach")]):
        g=clocks.query("endpoint==@k").sort_values("time_s")
        axes[1].step(g.time_s,g.cdf,where="post",label=label,color=COLORS[i])
        axes[1].fill_between(g.time_s,g.cdf_low,g.cdf_high,step="post",color=COLORS[i],alpha=.13)
    axes[1].axvline(27.05,ls=":",color="#333333")
    axes[1].set(xlim=(0,60),ylim=(0,1.03),xlabel="Available time (s)",ylabel="Conditional cumulative fraction",
                title="At sea: events within 60 s")
    axes[1].legend(fontsize=7,loc="lower right")
    finish(fig,"result_q4_time_budget",(7,2.8))


def main():
    OUT.mkdir(exist_ok=True)
    study_design()
    raw_features()
    classification_and_coverage()
    waveform()
    propagation()
    baseline_states()
    vigilance()


if __name__=="__main__":
    main()
