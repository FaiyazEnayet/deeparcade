from pathlib import Path
import json
import html
import re
import plotly.graph_objects as go


def render_dashboard(root):
    load = lambda name: json.loads((root / 'runs' / name).read_text(encoding='utf-8'))
    comparison = load('da8_comparison.json')
    families = ['mlp_control', 'cnn_dqn_plain', 'cnn_dqn_residual']
    names = ['MLP control', 'Plain CNN', 'Residual CNN']
    colors = ['#879b93', '#98d7af', '#e9bb61']
    means = [[5.61,5.16,5.60],[12.16,13.19,12.97],[16.82,20.42,21.86]]
    training = [[4.24,4.90,4.36],[9.85,10.29,9.89],[11.59,11.61,13.64]]
    for i,family in enumerate(families):
        for j,seed in enumerate((7,8,9)):
            key=f'{family}_seed{seed}'
            actual=sum(row['score'] for row in comparison['rows'][key])/500
            assert abs(actual-means[i][j])<.005
            assert load(key+'_meta.json')['final_mean100']==training[i][j]
    
    def chart(figure, first=False):
        figure.update_layout(paper_bgcolor='#101c18',plot_bgcolor='#101c18',font=dict(family='Space Mono, monospace',color='#eae8d9',size=12),margin=dict(l=62,r=25,t=34,b=64),height=410,legend=dict(orientation='h',y=-.22),hovermode='closest')
        figure.update_xaxes(gridcolor='#293c33',zerolinecolor='#668073',automargin=True)
        figure.update_yaxes(gridcolor='#293c33',zerolinecolor='#668073',automargin=True)
        return figure.to_html(full_html=False,include_plotlyjs=first,config={'displaylogo':False,'responsive':True})
    
    held=go.Figure()
    for i,name in enumerate(names):
        held.add_trace(go.Bar(name=name,x=['Seed 7','Seed 8','Seed 9'],y=means[i],text=[f'{v:.2f}' for v in means[i]],textposition='outside',cliponaxis=False,marker_color=colors[i],hovertemplate='%{x}<br>%{y:.2f} mean food score<extra>'+name+'</extra>'))
    for value,name,color in [(5.28,'Frozen MLP', '#b5919b'),(20.78,'Rookie / tabular Q','#b8d6ff'),(20.61,'Greedy','#c4c9c1')]:
        held.add_hline(y=value,line_dash='dot',line_color=color,annotation_text=f'{name} {value:.2f}',annotation_position='bottom right' if name=='Greedy' else 'top left')
    held.update_layout(barmode='group',yaxis_title='Mean score<br>keyed test',yaxis_range=[0,26])
    
    paired=go.Figure()
    for baseline,label in [('mlp_frozen','Frozen MLP'),('rookie','Rookie / tabular Q'),('greedy','Greedy')]:
        for family,color in zip(families[1:],colors[1:]):
            rows=[comparison['comparisons'][f'{family}_seed{s}']['vs_'+baseline] for s in (7,8,9)]
            values=[r['mean_diff'] for r in rows]
            paired.add_trace(go.Scatter(x=values,y=[f'{names[families.index(family)]} · {s}' for s in (7,8,9)],mode='markers',name=label+' / '+family,marker=dict(color=color,size=11),error_x=dict(type='data',array=[r['ci_high']-r['mean_diff'] for r in rows],arrayminus=[r['mean_diff']-r['ci_low'] for r in rows]),visible=baseline=='mlp_frozen',customdata=[[r['ci_low'],r['ci_high']] for r in rows],hovertemplate='%{y}<br>Difference %{x:+.2f}<br>95% CI [%{customdata[0]:+.2f}, %{customdata[1]:+.2f}]<extra></extra>'))
    paired.add_vline(x=0,line_color='#eae8d9',line_dash='dot')
    paired.update_layout(showlegend=False,xaxis_title='Paired difference<br>CNN minus baseline',updatemenus=[dict(buttons=[dict(label=label,method='update',args=[{'visible':[i//2==b for i in range(6)]}]) for b,label in enumerate(['Frozen MLP','Rookie / tabular Q','Greedy'])],x=0,y=1.17,xanchor='left')])
    
    curves=go.Figure()
    for i,family in enumerate(families):
        for j,seed in enumerate((7,8,9)):
            values=load(f'{family}_seed{seed}_curve.json')['mean_score']
            curves.add_trace(go.Scatter(x=list(range(1,len(values)+1)),y=values,name=f'{names[i]} · {seed}',line=dict(color=colors[i],dash=['solid','dash','dot'][j],width=2),hovertemplate='Logged checkpoint %{x}<br>Mean-100 score %{y:.2f}<extra>%{fullData.name}</extra>'))
    curves.update_layout(xaxis_title='Logged training checkpoint',yaxis_title='Training mean-100 score',height=490)
    
    def table(headers,rows):
        return '<div class="scroll" tabindex="0"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    
    rows=[]
    for i,name in enumerate(names):
        for j,seed in enumerate((7,8,9)):
            rows.append([name,seed,f'{means[i][j]:.2f}',f'{training[i][j]:.2f}'])
    legacy=table(['Agent / archived legacy protocol','Mean score'],[['Random','0.16'],['Greedy','19.89'],['Rookie · tabular Q','20.78'],['Previous Pro · flat MLP','5.39'],['Grandmaster · MCTS','44.1']])
    fonts=json.loads((root/'portfolio_ui/fonts.json').read_text(encoding='utf-8'))
    css="""
    :root{color-scheme:dark;--bg:#08110e;--surface:#101c18;--ink:#eae8d9;--muted:#aac0b3;--gold:#e9bb61;--line:#304238}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 'Space Mono',monospace}a{color:var(--gold)}p{overflow-wrap:anywhere}button{font:inherit;color:var(--ink);background:#20372b;border:1px solid #7fa48d;padding:12px 18px;cursor:pointer}a:focus-visible,button:focus-visible,.scroll:focus-visible{outline:3px solid var(--gold);outline-offset:4px}header,main,footer{max-width:1220px;margin:auto;padding:36px 28px}.overline{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.18em}.mast{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:20px;gap:15px;flex-wrap:wrap}.mast a{text-decoration:none}.hero{display:grid;grid-template-columns:1.3fr 1fr;gap:48px;padding:50px 0 36px}h1{font:normal clamp(3.6rem,8vw,6.5rem)/1.05 'Pirata One',Georgia,serif;margin:16px 0 26px;letter-spacing:-.025em}h2{font:600 clamp(1.35rem,2.5vw,2rem)/1.3 'Space Mono',monospace;margin:0 0 16px}h3{font-size:1.05rem;margin:0 0 12px}.hero p{color:#c8d6cd;max-width:630px}.range{border:1px solid var(--line);padding:28px;background:var(--surface);align-self:end}.range strong{font-size:clamp(1.5rem,2.5vw,2.2rem);color:var(--gold);display:block;line-height:1.3}.range p{margin-bottom:0}.seed-results{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:22px}.seed-results span{font-size:.72rem;color:var(--muted)}.seed-results>div{border-top:2px solid var(--gold);padding-top:12px}.hero-actions{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-top:28px}.hero-actions a{text-decoration:none}.play-link{display:inline-block;background:var(--gold);color:#08110e;padding:13px 22px;font-weight:bold;border:1px solid var(--gold)}.play-link:hover{background:#f4d28c}.tradeoff{display:grid;grid-template-columns:1fr 1fr;gap:22px;border-top:3px solid var(--gold);padding:25px;background:#18251d}.tradeoff strong{color:var(--gold);font-size:1.6rem}.tradeoff p{margin:6px 0}.anchor-nav{display:flex;gap:20px;flex-wrap:wrap;padding:24px 0;border-bottom:1px solid var(--line)}section{margin:54px 0}.section-title{display:flex;align-items:baseline;justify-content:space-between;gap:20px;flex-wrap:wrap}.panel{border:1px solid var(--line);background:var(--surface);padding:24px;min-width:0}.caption{font-size:.75rem;color:var(--muted);overflow-wrap:anywhere}.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}.scroll{overflow:auto;max-width:100%}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:left;border-bottom:1px solid var(--line);padding:12px 14px;font-size:.85rem}th{color:var(--gold)}.protocol{display:grid;grid-template-columns:repeat(3,1fr);gap:0}.protocol div{padding:18px;border:1px solid var(--line)}.protocol span{display:block;color:var(--muted);font-size:.75rem}.protocol strong{font-size:1rem}.notice{border-left:3px solid var(--gold);padding:16px 22px;background:#1d2c22}.game iframe{width:100%;height:940px;border:1px solid var(--line);margin-top:22px}footer{color:var(--muted);font-size:.75rem;border-top:1px solid var(--line)}@media(max-width:760px){header,main,footer{padding:24px 16px}.hero,.two,.tradeoff{grid-template-columns:1fr}.hero{gap:22px;padding-top:28px}.protocol{grid-template-columns:1fr 1fr}.panel{padding:14px}h1{font-size:4rem}.js-plotly-plot{min-width:0}.modebar{max-width:100%;display:none!important}.range{padding:20px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
    """
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepArcade | Snake agent research</title><style>{fonts['Space Mono']}{fonts['Pirata One']}{css}</style><header><div class="mast"><span>DEEPARCADE / AGENT LAB</span><span>v1.3.0 · <a href="game.html">Enter the arena ↗</a></span></div><div class="hero"><div><p class="overline">Snake agents / learning and search</p><h1>DeepArcade</h1><p>Play the trained agents, then inspect the evidence behind their decisions. Pro uses a residual CNN that improves on the flat MLP under matched training conditions. Its three seeds reach the tabular and heuristic baselines with substantial variation.</p><div class="hero-actions"><a class="play-link" href="game.html">Play Snake ↗</a><a href="#comparison">Explore the results ↓</a></div></div><aside class="range"><span class="overline">Residual CNN · all three held-out means</span><div class="seed-results"><div><span>Seed 7</span><strong>16.82</strong></div><div><span>Seed 8</span><strong>20.42</strong></div><div><span>Seed 9</span><strong>21.86</strong></div></div><p>Seeds 7 / 8 / 9 · 500 keyed test episodes per policy. Pro plays the seed 9 export.</p></aside></div><div class="tradeoff"><div><span class="overline">Representation budget</span><p><strong>13,795 / 32,291</strong><br>Plain / residual encoder parameters</p></div><div><span class="overline">Decision-time tradeoff</span><p><strong>≈20× slower</strong><br>MLP ≈21,600 · plain CNN ≈2,000 · residual CNN ≈1,000 decisions/s</p><p class="caption">Approximate measured inference rates from the same comparison run. Training throughput is a different measurement.</p></div></div><nav class="anchor-nav" aria-label="Sections"><a href="#comparison">Held-out results</a><a href="#paired">Paired evidence</a><a href="#training">Training</a><a href="#protocol">Protocol</a><a href="#arena">Play</a><a href="#legacy">Legacy evaluation</a></nav></header><main>
    <section id="comparison"><div class="section-title"><h2>Held-out scores by seed</h2><span class="overline">Keyed test namespace</span></div><p>All policies face the same episode keys. The test partition is disjoint from training by construction. Bars show per-policy means; dotted lines identify the frozen and simple baselines.</p><div class="panel">{chart(held,True)}</div><p class="caption">Source · runs/da8_comparison.json · 500 paired episodes per policy. Frozen MLP 5.28, Rookie 20.78, Greedy 20.61.</p></section>
    <section id="paired"><h2>Paired baseline comparisons</h2><div class="panel">{chart(paired)}</div><div class="notice">Against Rookie, residual seed 7 trails by −3.96 [−4.83, −3.09], seed 8 is −0.36 [−1.34, +0.62], and seed 9 edges ahead by +1.08 [+0.07, +2.09]. Seed 9 versus Greedy is +1.25 [+0.22, +2.27]. Intervals are paired 95% confidence intervals.</div><p>Spatial structure is a large, real improvement over the flat MLP under matched conditions. The residual CNN reaches the tabular and heuristic baselines but does not clearly beat them. Its seed means span 16.82 to 21.86.</p><p class="caption">Source · runs/da8_comparison.json · use the chart selector to compare every CNN seed with each baseline. Every CNN-versus-frozen-MLP interval is far from zero.</p></section>
    <section id="training"><h2>Training trajectories</h2><div class="panel">{chart(curves)}</div><p class="caption">Source · runs/*_seed{{7,8,9}}_curve.json · recorded checkpoint order, with the original mean-100 training scores. These are training trajectories, separate from the held-out evaluation above.</p><div class="panel">{table(['Encoder','Seed','Held-out mean','Final training mean-100'],rows)}</div></section>
    <section id="protocol"><h2>Matched training protocol</h2><div class="protocol">'''+''.join(f'<div><span>{k}</span><strong>{v}</strong></div>' for k,v in [('Environment steps','300,000'),('Seeds','7 / 8 / 9'),('Optimizer','Adam · 1e-3'),('Discount gamma','0.95'),('Batch / replay','64 / 50,000'),('Update / target sync','8 / 2,000 steps'),('Epsilon','1.0 → 0.05'),('Decay / warmup','120,000 / 2,000 steps'),('Hardware','CPU')])+f'''</div><p>Reward shaping and the episode-key training partition are held constant with the MLP. The frozen MLP remains the reference baseline, alongside three retrained MLP controls.</p><p class="caption">Sources · runs/da8_comparison.json, runs/*_meta.json, scripts/train_cnn_dqn.py. Reported suite status at release · 116 passed, 5 skipped.</p></section>
    <section id="arena" class="game"><h2>Meet Pro in the arena.</h2><div class="two"><article class="panel"><h3>Pro · residual CNN, seed 9</h3><p>The playable export is runs/cnn_dqn_residual_seed9_weights.json. It is the strongest held-out seed and the only residual seed whose paired intervals edge both simple baselines.</p><p>The selector also includes the frozen MLP as Previous Pro. Rookie plays tabular Q-learning. Grandmaster uses MCTS.</p><a href="game.html">Open the standalone game ↗</a></article><article class="panel"><h3>From export to action</h3><p>Head, body and food occupy spatial channels. Heading conditions the pooled features. The browser uses the exported convolutions, residual block and dense head.</p><button id="load-game">Load playable arena</button><p class="caption">The arena loads when requested. Keyboard controls remain inside the game.</p></article></div><div id="game-host"></div></section>
    <section id="legacy"><h2>Legacy evaluation records</h2><p>The archived 500-episode evaluation uses a separate namespace from the keyed comparison above. The old Pro in this table is the frozen flat MLP.</p><div class="panel">{legacy}</div><p class="caption">Source · runs/eval_legacy.json · 500 episodes for the non-MCTS policies, 50 for MCTS.</p><div class="notice">Current paired-pool record · Grandmaster mean 46.14, median 47.0 over 50 episodes. Source · runs/eval.json. This is separate from the archived Grandmaster mean 44.1 above.</div></section></main><footer>DeepArcade · Faiyaz Enayet · Read the committed artifacts to reproduce the comparison. Browser play illustrates behavior; the reported comparisons come from the recorded evaluation protocols.</footer><script>document.getElementById('load-game').onclick=function(){{const frame=document.createElement('iframe');frame.src='game.html';frame.title='DeepArcade playable Snake arena';document.getElementById('game-host').append(frame);this.disabled=true;this.textContent='Arena loaded';}};</script></html>'''
    # Lead with play and the agent evaluation. The encoder study follows
    # as its own research section with its protocol and cost tradeoff.
    seed_start = page.index('<aside class="range">')
    seed_end = page.index('</aside>', seed_start) + len('</aside>')
    seed_card = page[seed_start:seed_end]
    grandmaster_card = '''<aside class="range">
      <span class="overline">Grandmaster / MCTS · current paired pool</span>
      <div class="seed-results">
        <div><span>Mean score</span><strong>46.14</strong></div>
        <div><span>Median</span><strong>47.0</strong></div>
        <div><span>Episodes</span><strong>50</strong></div>
      </div>
      <p>Current evaluation record · runs/eval.json. Grandmaster is the strongest agent in the evaluated lineup.</p>
    </aside>'''
    page = page[:seed_start] + grandmaster_card + page[seed_end:]
    page = page.replace(
        'Play the trained agents, then inspect the evidence behind their decisions. Pro uses a residual CNN that improves on the flat MLP under matched training conditions. Its three seeds reach the tabular and heuristic baselines with substantial variation.',
        'Trained policies and Monte Carlo tree search in a playable Snake arena. Watch an agent or challenge the tournament ladder, then explore the recorded evaluations. Grandmaster leads the evaluated lineup; Pro uses a residual CNN.'
    )
    tradeoff_start = page.index('<div class="tradeoff">')
    tradeoff_end = page.index('<nav class="anchor-nav"', tradeoff_start)
    tradeoff = page[tradeoff_start:tradeoff_end]
    page = page[:tradeoff_start] + page[tradeoff_end:]
    nav_start = page.index('<nav class="anchor-nav"')
    nav_end = page.index('</nav>', nav_start) + len('</nav>')
    page = page[:nav_start] + '''<nav class="anchor-nav" aria-label="Sections">
      <a href="#arena">Play</a><a href="#legacy">Agent results</a>
      <a href="#comparison">CNN research</a><a href="#paired">Paired comparisons</a>
      <a href="#training">Training</a><a href="#protocol">Protocol</a>
    </nav>''' + page[nav_end:]
    page = page.replace('href="#comparison">Explore the results', 'href="#legacy">Explore the results')
    arena = re.search(r'<section id="arena".*?</section>', page, re.S).group()
    results = re.search(r'<section id="legacy".*?</section>', page, re.S).group()
    page = page.replace(arena, '').replace(results, '')
    arena = '''<section id="arena" class="game">
      <h2>Play DeepArcade</h2>
      <p>Play a round, watch an agent, or climb the tournament ladder.
      Rookie uses tabular Q-learning, Pro uses a residual CNN, and Grandmaster uses MCTS.
      The frozen MLP is also available as Previous Pro.</p>
      <iframe src="game.html" loading="lazy" title="DeepArcade playable Snake arena"></iframe>
      <p class="caption">The recorded evaluations use the classic board.
      Browser MCTS adapts its search budget for responsiveness. Use the standalone game for a full-window arena.</p>
    </section>'''
    results = results.replace('Legacy evaluation records', 'Agent results and evaluation records')
    notice = re.search(r'<div class="notice">.*?</div>', results, re.S).group()
    results = results.replace(notice, '').replace('<div class="panel">', notice + '<div class="panel">', 1)
    page = page.replace('<main>', '<main>' + arena + results, 1)
    page = page.replace('<section id="comparison">', '<section id="comparison">' + '<p class="overline">Research / convolutional DQN under matched conditions</p>' + seed_card + tradeoff, 1)
    page = re.sub(r"<script>document.getElementById\('load-game'\).*?</script>", '', page, flags=re.S)
    return page
