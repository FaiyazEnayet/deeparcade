from pathlib import Path
import json
from bs4 import BeautifulSoup
HERE=Path(__file__).parent
fonts=json.loads((HERE/"fonts.json").read_text(encoding="utf8"))
fontcss=fonts["Space Mono"]+"\n"+fonts["Pirata One"]
css='''
#replaybar{display:none!important}
.play-area{position:relative;isolation:isolate}#overlay{z-index:2}#overlay .btnrow{align-items:center}

*{box-sizing:border-box}body{margin:0}#stage{position:relative}canvas{display:block}.hidden{display:none!important}#side{display:flex;flex-direction:column}.btn{display:block;width:100%;cursor:pointer;margin:6px 0}select,input[type=text]{width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left}.elo-up{color:var(--accent)}.elo-dn{color:var(--danger)}#overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:16px;overflow:auto}#overlay h2,#overlay p{margin:0}#overlay .btnrow{display:grid;grid-template-columns:1fr 1fr;gap:12px;width:min(100%,320px)!important}#overlay .btn{margin:0;min-width:0;width:100%}#replaybar{display:flex;align-items:center;justify-content:center}#replaybar .btn{width:auto}
:root{--bg:#0b1410;--panel:#13251b;--accent:#9ad7ac;--accent2:#d5c17a;--danger:#ee9d94;--gold:#e4cb78;--text:#edf0db;--dim:#b5c6b4;color-scheme:dark}body{background:var(--bg);font-family:'Space Mono',monospace;display:block;min-height:100vh;padding:0 24px 32px;color:var(--text);line-height:1.6}h1{font-family:'Pirata One',Georgia,serif;background:none;-webkit-text-fill-color:var(--text);color:var(--text);letter-spacing:.02em;font-size:76px;font-weight:400;margin:8px 0}h2{font-family:'Pirata One',Georgia,serif;font-weight:400}.game-masthead{max-width:1000px;margin:0 auto 24px;border-bottom:1px solid #3a5740;padding:18px 0 24px}.game-topline{display:flex;justify-content:space-between;gap:20px;font-size:11px;color:var(--dim);border-bottom:2px solid var(--gold);padding-bottom:12px;flex-wrap:wrap}.game-topline a{color:var(--gold);text-decoration:none}.game-title{display:flex;justify-content:space-between;align-items:center;gap:28px}.game-edition{font-size:11px;color:var(--gold);text-align:right;line-height:1.8}.game-tagline{font-size:13px;color:var(--dim);margin:0;max-width:800px}.game-facts{display:flex;gap:24px;font-size:10px;color:var(--dim);margin-top:18px;flex-wrap:wrap}#wrap{display:grid;grid-template-columns:minmax(0,600px) 320px;gap:24px;max-width:1000px;width:100%;margin:auto;align-items:start}#stage{min-width:0;width:100%}canvas{max-width:100%;width:100%;height:auto;border-radius:3px;border:2px solid #5c8261;box-shadow:none;background:#0c1b13}#hud{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;justify-content:start;margin:14px 0}#side{width:auto;min-width:0;gap:20px}.card{border-radius:2px;background:var(--panel);border:1px solid #3a5740;padding:20px}.card h3{font-size:12px;color:var(--gold);font-weight:400;letter-spacing:.07em;line-height:1.5;margin-bottom:12px}.pill{border-radius:2px;background:transparent;border:1px solid #3a5740;font-size:11px;padding:8px 10px}.pill b{display:block;font-size:21px;font-weight:400}.btn{font-family:'Space Mono',monospace;background:#294d36;color:#edf0db;border:1px solid #719a77;border-radius:3px;box-shadow:none;font-weight:400;font-size:12px;padding:12px;transition:background-color 140ms ease-out,border-color 140ms ease-out;min-height:44px}.btn.green{background:#214a32;color:#edf0db}.btn.gold{background:#d5c17a;color:#102017;border-color:#d5c17a}.btn.ghost{background:#182c20;color:#dce6d5}.btn:hover{transform:none;box-shadow:none;background:#365e41;border-color:#b1d3af}.btn.gold:hover{background:#e4d596}.btn:disabled{opacity:.7;cursor:default}.btn:focus-visible,button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid var(--gold);outline-offset:4px}select,input[type=text]{font-family:'Space Mono',monospace;border:1px solid #719a77;border-radius:3px;background:#0d1d14;color:var(--text);font-size:12px;min-height:44px;padding:10px;max-width:100%}table{font-size:12px}th{font-size:11px;color:var(--dim)}td,th{border-color:#3a5740;padding:7px 4px}.tag{font-size:10px;padding:2px 6px;border-radius:2px}.tag.ai{background:#233d2d;color:#d5c17a}.tag.you{background:#233d2d;color:#a5ddb0}#overlay{background:rgba(8,20,13,.94);border-radius:3px;backdrop-filter:none;padding:20px;text-align:center}#overlay h2{font-size:46px}#overlay p{color:var(--dim);font-size:13px}#overlay .btnrow{max-width:100%}#replaybar{flex-wrap:wrap;gap:8px}#replaybar .btn{min-width:48px}kbd{font-family:'Space Mono',monospace;background:#223c2c;border-color:#5c8261;border-radius:2px;font-size:12px;padding:3px 7px}.touch-controls{display:grid;grid-template-columns:repeat(3,48px);grid-template-rows:repeat(2,44px);gap:8px;justify-content:center;margin:12px auto}.touch-controls button{border:1px solid #719a77;border-radius:3px;background:#182e20;color:var(--text);font-size:18px;touch-action:manipulation;cursor:pointer}.touch-controls button:first-child{grid-column:2}.touch-controls button:nth-child(2){grid-column:1;grid-row:2}.touch-controls button:nth-child(3){grid-column:2;grid-row:2}.touch-controls button:nth-child(4){grid-column:3;grid-row:2}.control-caption{text-align:center;font-size:11px;color:var(--dim);margin-top:12px}footer{margin:24px auto 0;max-width:1000px;border-top:1px solid #3a5740;padding-top:20px;font-size:11px;line-height:1.8;color:var(--dim)}footer a{color:var(--gold)}
@media(max-width:800px){body{padding:0 16px 28px}.game-masthead{padding-top:16px;margin-bottom:20px}.game-title{gap:14px}h1{font-size:57px}.game-edition{font-size:9px}.game-tagline{font-size:12px}.game-topline{font-size:10px}.game-facts{gap:12px;font-size:10px;margin-top:14px}#wrap{grid-template-columns:1fr;max-width:600px;gap:20px}#side{display:grid;grid-template-columns:1fr 1fr;gap:16px}#menuCard{grid-column:1/-1}.card{padding:16px}#hud{gap:6px}.pill{font-size:10px;padding:7px}.pill b{font-size:20px}.card h3{font-size:11px}.btn{font-size:12px}#overlay h2{font-size:36px}footer{font-size:11px}}
@media(max-width:420px){#side{grid-template-columns:1fr}#menuCard{grid-column:auto}.game-edition{display:none}.game-title h1{font-size:57px}.game-facts{display:grid;grid-template-columns:1fr 1fr;gap:6px}.game-topline{font-size:9px}.pill b{font-size:18px}.pill{font-size:10px}#overlay p{font-size:11px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
'''
def theme(raw):
 if 'class="game-masthead"' in raw:return raw
 soup=BeautifulSoup(raw,'html.parser')
 # Replace only presentation; scripts remain intact apart from paint colors and reduced-motion effects.
 for st in soup.find_all('style'):st.decompose()
 st=soup.new_tag('style');st.string=fontcss+'\n'+css;soup.head.append(st)
 soup.title.string='DeepArcade | Play the trained agents'
 old=soup.find('h1');sub=soup.select_one('.sub')
 header=BeautifulSoup('''<header class="game-masthead"><div class="game-topline"><span>DEEPARCADE / PLAYABLE RESEARCH</span><a href="dashboard.html" target="_top">Explore the agent evaluation ↗</a></div><div class="game-title"><h1>DeepArcade</h1><div class="game-edition">HUMAN × LEARNED POLICY × SEARCH<br>ONE CLASSIC ARENA</div></div><p class="game-tagline">Two learned agents. One search agent. Watch their decisions, play against their scores, and play the same arena.</p><div class="game-facts"><span>Classic board · 12 × 12</span><span>Pro · tabular Q</span><span>Rookie · DQN</span><span>Grandmaster · MCTS</span></div></header>''','html.parser')
 old.replace_with(header);sub.decompose() if sub else None
 labels={'btnHuman':'Play as human','btnAI':'Watch AI play','btnTourn':'Start tournament run','rpPlay':'Pause / resume','rpBack':'Step back','rpFwd':'Step forward','rpExit':'Exit replay'}
 for ident,txt in labels.items():
  b=soup.find(id=ident)
  if b:b.string=txt;b['aria-label']=txt
 for ident,label in [('modeSel','Game mode'),('aiSel','AI opponent')]:soup.find(id=ident)['aria-label']=label
 replay=soup.find(id='ovReplay')
 if replay:replay.decompose()
 cv=soup.find(id='cv')
 area=soup.new_tag('div');area['class']='play-area';cv.wrap(area);area.append(soup.find(id='overlay'))
 cv=soup.find(id='cv');cv['aria-label']='Snake game board. Use arrow keys or WASD.';cv['role']='img'

 replays=soup.find(id='rpList')
 if replays:replays.find_parent(class_='card')['hidden']='hidden'

 footer=soup.find('footer');footer.clear();footer.append('Recorded evaluation uses classic boards; browser MCTS adjusts its search budget for responsiveness. Human ELO is a rough single-score estimate. ');a=soup.new_tag('a',href='dashboard.html',target='_top');a.string='Return to evaluation';footer.append(a)
 for sc in soup.find_all('script'):
  text=sc.string or ''
  text=text.replace("document.getElementById('ovReplay').classList.remove('hidden');",'').replace("document.getElementById('ovReplay').onclick=()=>{playReplay(0);};",'')
  # These substitutions affect canvas paint only, not game state or AI arithmetic.
  for a,b in {'#0d1226':'#0c1b13','rgba(80,120,200,0.07)':'rgba(150,190,140,0.12)','#2b3a67':'#466249','#4ade80':'#9ad7ac','#bbf7d0':'#efe2ad','#22c55e':'#d7bf72','#166534':'#285b39','#0b0e1a':'#0b1410'}.items():text=text.replace(a,b)
  text=text.replace('function spawnBurst(r,c,color,n=14){','function spawnBurst(r,c,color,n=14){\n  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;')
  text=text.replace('if(shake>0){ ctx.translate','if(shake>0 && !matchMedia("(prefers-reduced-motion: reduce)").matches){ ctx.translate')
  text=text.replace('const pr=CELL*0.28+Math.sin(t)*3;','const pr=CELL*0.28+(matchMedia("(prefers-reduced-motion: reduce)").matches?0:Math.sin(t)*3);')
  sc.string=text
 extra=soup.new_tag('script');extra.string='document.querySelectorAll("[data-direction]").forEach(b=>b.addEventListener("pointerdown",e=>{e.preventDefault();dispatchEvent(new KeyboardEvent("keydown",{code:b.dataset.direction,key:b.dataset.direction,bubbles:true}));}));';soup.body.append(extra)
 return str(soup)