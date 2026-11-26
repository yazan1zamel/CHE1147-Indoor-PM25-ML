"""Recreate publication figures from completed paper-aligned results.

Requires results at $CHE1147_DATA_ROOT/Results/PM25_10min_Colleague_Aligned_cap100.
Does not fit any model or alter the test labels.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('CHE1147_DATA_ROOT', str(Path.home()/'Desktop/PhD/Courses/CHE1147/Data')))
RUN = ROOT/'Results'/'PM25_10min_Colleague_Aligned_cap100'
T = RUN/'tables'
X = RUN/'xgboost_only'
OUT = RUN/'report_figures'
OUT.mkdir(parents=True, exist_ok=True)

main = pd.read_csv(T/'08_held_out_predictions.csv', dtype={'household_id':str})
xgb = pd.read_csv(X/'xgboost_predictions.csv', dtype={'household_id':str})
if len(main)!=len(xgb) or not main['household_id'].equals(xgb['household_id']) or not np.allclose(main.actual_pm,xgb.actual_pm,equal_nan=True) or not np.allclose(main.pred_persistence,xgb.persistence_prediction,equal_nan=True):
    raise ValueError('Main and XGBoost predictions do not match on household, target, and persistence. Refusing to combine them.')
main['pred_xgboost']=xgb.xgboost_prediction.values
models={'Persistence':'pred_persistence','LightGBM':'pred_lightgbm','XGBoost':'pred_xgboost'}
colors={'Persistence':'#64748B','LightGBM':'#2563EB','XGBoost':'#10B981'}
labels=['<5','5–<10','10–<30','≥30']
main['stratum']=pd.cut(main.actual_pm,[-np.inf,5,10,30,np.inf],labels=labels,right=False)
rows=[]
for label in labels:
    subset=main[main.stratum==label]
    for name,col in models.items():
        rows.append({'range_ug_m3':label,'model':name,'count':len(subset),'MAE_ug_m3':(subset.actual_pm-subset[col]).abs().mean()})
df=pd.DataFrame(rows)
df.to_csv(OUT/'mae_by_concentration.csv',index=False)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.3,4.6),gridspec_kw={'width_ratios':[2.35,1]})
counts=main.stratum.value_counts().reindex(labels)
for i,name in enumerate(models):
    series=df[df.model==name].set_index('range_ug_m3').MAE_ug_m3
    bs=ax1.bar(np.arange(3)+(i-1)*.24,series.reindex(labels[:3]).values,.24,color=colors[name],label=name)
    for b in bs:ax1.text(b.get_x()+b.get_width()/2,b.get_height()+.02,f'{b.get_height():.3f}',fontsize=8,ha='center')
    b=ax2.bar((i-1)*.24,float(series.loc['≥30']),.24,color=colors[name])[0]
    ax2.text(b.get_x()+b.get_width()/2,b.get_height()+.15,f'{b.get_height():.2f}',fontsize=9,ha='center')
ax1.set_xticks(range(3),[f'{s}\n(n={counts[s]:,})' for s in labels[:3]]);ax1.set_ylim(0,1.53)
ax2.set_xticks([0],[f'≥30\n(n={counts["≥30"]:,})']);ax2.set_xlim(-.55,.55);ax2.set_ylim(0,15.5)
for ax in [ax1,ax2]:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
ax1.set_ylabel('MAE (µg/m³)');ax1.set_title('(a) Below 30 µg/m³');ax2.set_title('(b) Elevated concentrations')
fig.suptitle('Prediction error by indoor PM₂.₅ concentration',y=1.02,fontsize=13,fontweight='bold')
fig.legend(*ax1.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1),ncol=3,frameon=False)
fig.text(.5,.004,'Observed concentration (µg/m³); separate y-axis scales',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.88]);fig.savefig(OUT/'Figure_3_Improved.png',dpi=300,bbox_inches='tight');plt.close(fig)
print('Wrote:',OUT/'Figure_3_Improved.png')
