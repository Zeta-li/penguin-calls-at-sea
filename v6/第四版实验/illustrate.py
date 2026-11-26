"""Figures for the four-question design and adjusted playback contrasts."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from figures_apmcm import save
from penguin_data import CALLS, CONTEXTS, STATE_NAMES

def main():
    folder=HERE/"figures"
    folder.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7.3,3.35))
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    x0,x1,x2=.16,.49,.835
    rows=[(.81,"108 acoustic records","Date-held-out classification","Acoustic discriminability"),
          (.52,"9 correlation tables","Monotone component curves","Mean / weakest feature"),
          (.23,"139 playback pairs","Joint states + first-event times","Contact / recruitment")]
    for y,left,middle,right in rows:
        for x,s in [(x0,left),(x1,middle),(x2,right)]:
            ax.text(x,y,s,ha="center",va="center",fontsize=9,bbox=dict(boxstyle="round,pad=.50",fc="#eaf2f3" if x!=x2 else "#e5efe8",ec="#6b858c",lw=.7))
        for start,end in ((.30,.325),(.653,.683)):
            ax.annotate("",(end,y),(start,y),arrowprops=dict(arrowstyle="->",color="#3c6265",lw=1.3))
    ax.annotate("",(.84,.31),(.84,.44),arrowprops=dict(arrowstyle="->",color="#3c6265",lw=1.3))
    ax.text(.50,.033,"Validation units: recording date   /   held-out distance   /   experiment pair or natural track",ha="center",fontsize=8,color="#324b57")
    fig.subplots_adjust(left=.015,right=.985,top=.98,bottom=.04)
    save(fig,folder,"flow_overall_model")
    t=pd.read_csv(HERE/"results/q3v4_state_shifts.csv",dtype={"state":str})
    q=t[(t.endpoint=="net_response")&(t.stimulus!="control")]
    fig,axes=plt.subplots(1,3,figsize=(8,2.65),layout="constrained",sharey=True)
    titles=["Commuting","Searching","After feeding"]
    for j,(c,a) in enumerate(zip(CONTEXTS,axes)):
        z=q[q.context==c].pivot(index="stimulus",columns="state",values="adjusted_difference").loc[list(CALLS),list(STATE_NAMES)].to_numpy()*100
        im=a.imshow(z,vmin=-75,vmax=75,cmap="RdBu_r",aspect="auto")
        a.set(xticks=range(8),xticklabels=STATE_NAMES,yticks=range(3),yticklabels=["Flat","Modulated","Two-voice"],title=titles[j])
        a.tick_params(axis="x",labelrotation=90,labelsize=8)
        for (r,k),v in np.ndenumerate(z):
            a.text(k,r,f"{v:.0f}",ha="center",va="center",fontsize=7,color="white" if abs(v)>41 else "black")
    fig.colorbar(im,ax=axes,shrink=.85,label="Difference (pp)")
    save(fig,folder,"result_q3_state_shifts")

if __name__=="__main__":
    main()
