"""Draw the observed-data and model dependencies used in the sixth edition."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from figures_apmcm import save


def flow():
    folder=Path(__file__).resolve().parent/'figures'; folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial Unicode MS','Noto Sans CJK SC','Hiragino Sans GB','DejaVu Sans'],'svg.fonttype':'none','svg.hashsalt':'six-structure'})
    fig,ax=plt.subplots(figsize=(6.48,2.65)); fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.015)
    ax.set(xlim=(0,10),ylim=(0,5.05)); ax.axis('off')
    def box(x,y,w,h,line1,line2,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=.11',lw=.8,edgecolor=color,facecolor='#f7faf9'))
        ax.text(x+w/2,y+h*.68,line1,ha='center',va='center',fontsize=9,color=color,fontweight='bold')
        ax.text(x+w/2,y+h*.26,line2,ha='center',va='center',fontsize=7.1,color='#343d43')
    def arrow(a,b,style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color='#4c636c',lw=1,linestyle=style))
    for y,left,right in [(3.85,('叫声测量','108 条，16 个日期'),('跨日声型分类','六项共测特征、谐波和谱形')),(2.16,('水库传播','9 张表，7 个实测距离'),('逐表征衰减','拟合曲线、门槛交点及缺格界')),(.46,('海上回放','139 对自然声与对照'),('群体行为预测','联合状态、首发时间及强度'))]:
        col=('#195f50' if y>3 else '#235677' if y>1 else '#734c2d'); box(.15,y,2.2,.92,*left,col);box(3.3,y,3.15,.92,*right,col);arrow((2.44,y+.46),(3.2,y+.46))
    box(7.36,1.44,2.45,1.1,'功能距离','联系与招募分别评价','#513c67')
    arrow((4.88,3.77),(4.88,3.18),'--');arrow((6.42,2.21),(7.27,2.06));arrow((6.52,1.02),(7.26,1.8));
    ax.text(5.01,3.48,'类间判别权重',ha='left',va='center',fontsize=6.7,color='#435b65')
    ax.text(7.45,.89,'基准取海上净响应',fontsize=7.15,color='#513c67');ax.text(7.45,.52,'噪声只改变相对保留量',fontsize=7.15,color='#513c67')
    save(fig,folder,'flow_overall_model')

if __name__=='__main__':flow()
