from pathlib import Path
import base64
import io
import json
import re
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

base=Path(__file__).resolve().parents[1]
def font(repo,family,size):
    fonts=json.loads((base/repo/'portfolio_ui/fonts.json').read_text(encoding='utf-8'))
    path=base/(family.replace(' ','')+'.ttf')
    if not path.exists():
        raw=base64.b64decode(re.findall(r'base64,([^\)]+)',fonts[family])[-1])
        f=TTFont(io.BytesIO(raw));f.flavor=None;f.save(path)
    return ImageFont.truetype(str(path),size)

def arcade():
    root=base
    sys.path.insert(0,str(root/'src'))
    from narcade.snake import SnakeEnv
    from narcade.models import load_cnn_checkpoint,encode_cnn_dqn
    net,_=load_cnn_checkpoint(root/'runs/cnn_dqn_residual_seed9_weights.json')
    torch.set_num_threads(1)
    env=SnakeEnv(seed=7)
    states=[]
    while not env.done:
        spatial,heading=encode_cnn_dqn(env)
        with torch.no_grad():
            action=int(net(torch.from_numpy(spatial).unsqueeze(0),torch.from_numpy(heading).unsqueeze(0)).argmax(1))
        states.append(dict(body=list(env.body),food=env.food,score=env.score,action=action,step=env.steps))
        env.step(action)
    states.append(dict(body=list(env.body),food=env.food,score=env.score,action=None,step=env.steps))
    
    title=font('','Pirata One',62)
    small=font('','Space Mono',17)
    medium=font('','Space Mono',24)
    large=font('','Space Mono',62)
    frames=[]
    stride=max(1,len(states)//140)
    chosen=states[::stride]
    if chosen[-1]!=states[-1]:chosen.append(states[-1])
    for index,state in enumerate(chosen):
        im=Image.new('RGB',(1120,740),'#08110e');d=ImageDraw.Draw(im)
        d.text((38,19),'DeepArcade',font=title,fill='#eae8d9')
        d.text((42,94),'PRO / RESIDUAL CNN / SEED 9',font=small,fill='#e9bb61')
        d.line((42,128,1078,128),fill='#304238',width=2)
        ox,oy,cell=42,155,43
        for r in range(12):
            for c in range(12):d.rectangle((ox+c*cell,oy+r*cell,ox+(c+1)*cell-2,oy+(r+1)*cell-2),fill='#172a21' if (r+c)%2 else '#102219')
        for i,(r,c) in enumerate(state['body']):
            d.rounded_rectangle((ox+c*cell+3,oy+r*cell+3,ox+(c+1)*cell-5,oy+(r+1)*cell-5),radius=7,fill='#e9bb61' if i==len(state['body'])-1 else '#78bb91')
        if state['food']:
            r,c=state['food'];x,y=ox+c*cell+21,oy+r*cell+21
            d.ellipse((x-10,y-10,x+10,y+10),fill='#f2886d')
        d.text((620,157),'FOOD SCORE',font=small,fill='#aac0b3')
        d.text((615,189),str(state['score']).zfill(2),font=large,fill='#e9bb61')
        d.text((620,290),'BOARD → FEATURES',font=small,fill='#aac0b3')
        d.text((620,327),'Head · body · food',font=medium,fill='#eae8d9')
        d.text((620,369),'Heading conditioning',font=small,fill='#aac0b3')
        d.line((620,420,1060,420),fill='#304238')
        d.text((620,452),'NEXT ACTION',font=small,fill='#aac0b3')
        label='The End' if state['action'] is None else ['STRAIGHT','TURN LEFT','TURN RIGHT'][state['action']]
        d.text((620,488),label,font=medium,fill='#e9bb61')
        d.text((620,570),'Actual policy rollout',font=small,fill='#eae8d9')
        d.text((620,601),'Illustrative classic episode',font=small,fill='#aac0b3')
        d.text((42,699),'PLAY THE EXPORTED POLICY  /  docs/game.html',font=small,fill='#aac0b3')
        frames.append(im.quantize(colors=128))
    durations=[95]*len(frames);durations[0]=700;durations[-1]=1800
    frames[0].save(root/'docs/demo.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True)
    print('Arcade GIF',len(frames),'frames; actual final score',env.score)


if __name__=='__main__':
    arcade()
