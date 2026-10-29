"""Render paired evaluation evidence from the current versioned artifact."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
ev=json.loads((ROOT/'runs/eval.json').read_text())
keys=['random','greedy','pro','rookie','grandmaster']
names=['Random','Greedy','Rookie · DQN','Pro · Q-learning','Grandmaster · MCTS']
fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
fig.set_facecolor('#0b1912')
for ax,vals,title in [(axes[0],[ev['elo'][k] for k in keys],'Paired tournament ELO'),(axes[1],[ev['paired']['summary'][k]['mean'] for k in keys],'Paired mean food score')]:
 ax.set_facecolor('#0b1912');ax.barh(names,vals,color=['#698b80','#83b29a','#c79b75','#a7c8a4','#d9bf75']);ax.set_title(title,color='#ece4cc',pad=18);ax.tick_params(colors='#ece4cc');ax.spines[['top','right']].set_visible(False);ax.set_xlim(0,max(vals)*1.2)
 for i,v in enumerate(vals):ax.text(v,i,' '+str(v),va='center',color='#ece4cc')
fig.suptitle('DeepArcade · shared 50-seed evaluation pool',color='#ece4cc',fontsize=19)
fig.savefig(ROOT/'docs/results.png',dpi=150)
