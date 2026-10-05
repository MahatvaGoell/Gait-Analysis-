"""Reference-style presentation; no smoothing, time warping or signal edits."""
import json
import math
import time
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from generate_qs_plots import font
from subject08_review import COLORS, runs


def render(path, title, grid, z, markers, info, quality, phases=None,
           segmentation_note='', imputed=None):
    assert np.allclose(np.diff(grid), .01, atol=1e-5), '100 Hz axis requires a verified 100 Hz grid'
    imputed = np.zeros_like(z, dtype=bool) if imputed is None else imputed
    im = Image.new('RGB', (2600, 900), 'white')
    d = ImageDraw.Draw(im)
    left, right, top, bottom = 140, 2330, 200, 690
    xmax = math.ceil(grid[-1] * 100 / 200) * 200
    xp = lambda seconds: left + seconds * 100 / xmax * (right-left)
    yp = lambda value: bottom - value / 500 * (bottom-top)
    def centered(x, y, text, size=22, bold=False, color='black'):
        f = font(size, bold)
        box = d.textbbox((0, 0), text, font=f)
        d.text((x-(box[2]-box[0])/2, y), text, font=f, fill=color)
    d.text((left, 20), title, font=font(25, True), fill='black')
    d.text((left, 60), 'Original time retained | solid: source-grid values | pale: estimated values | phase cuts provisional', font=font(18), fill='#744600')
    d.line((left,top,left,bottom,right,bottom),fill='#333333',width=2)
    for value in range(0,501,100):
        d.line((left,yp(value),left+12,yp(value)), fill='#333333', width=2)
        d.text((left-58,yp(value)-14),str(value),font=font(23),fill='#333333')
    step = max(200, math.ceil(xmax/3000)*200)
    for sample in range(0,xmax+1,step):
        x = xp(sample/100)
        d.line((x,bottom,x,bottom+12),fill='#333333',width=2)
        centered(x,bottom+18,str(sample),22)
    centered((left+right)/2,750,'Samples @ 100 Hz (original trial timeline)',25,True)
    lab = Image.new('RGBA',(510,40),(255,255,255,0))
    ImageDraw.Draw(lab).text((0,0),'Display amplitude (scaled and offset)',font=font(24,True),fill='black')
    lab = lab.rotate(90,expand=True)
    im.paste(lab,(25,200),lab)
    transforms=[]
    vgrf_display=None
    for k in range(10):
        x=z[:,k]; valid=np.isfinite(x)
        name=f'Channel {k+1}' if k<8 else ('CoP','vGRF')[k-8]
        if not valid.any():
            transforms.append(dict(signal=name,missing=True))
            continue
        anchor=[200,170,145,120,100,80,60,40,400,380][k]
        baseline=grid<.75
        base=float(np.nanmedian(x[baseline])) if np.any(np.isfinite(x[baseline])) else float(np.nanmedian(x))
        delta=x-base
        low,high=(10,340) if k<8 else ((365,435) if k==8 else (345,425))
        up=max(0,float(np.nanmax(delta))); down=max(0,float(-np.nanmin(delta)))
        gain=min(1.0 if k<8 else (2.5 if k==8 else .08),(high-anchor)/max(up,1e-9),(anchor-low)/max(down,1e-9))
        display=anchor+delta*gain
        if k==9:vgrf_display=display
        rgb=ImageColor.getrgb(COLORS[k]); pale=tuple(int(.55*c+.45*255) for c in rgb)
        # Draw only finite runs. Missing intervals are never bridged.
        for a,b in runs(valid):
            if b-a>1:d.line([(xp(grid[j]),yp(display[j])) for j in range(a,b)],fill=pale,width=2)
        for a,b in runs(valid & ~imputed[:,k]):
            if b-a>1:d.line([(xp(grid[j]),yp(display[j])) for j in range(a,b)],fill=COLORS[k],width=2)
        y=top+15+k*32
        d.line((2360,y+10,2400,y+10),fill=COLORS[k],width=3)
        d.text((2410,y),name+(' [est.]' if imputed[:,k].all() else ''),font=font(18),fill='black')
        transforms.append(dict(signal=name,baseline=base,gain=gain,offset=anchor))
    labels={'QS':'QS','GI':'GI','SSSW':'Steady state short step (SSSW)',
            'SLT':'SLT','SSLW':'Steady state long step (SSLW)','GT':'GT'}
    for phase in phases or []:
        a,b=phase['start_seconds'],phase['end_seconds']; name=phase['phase']
        level=120 if name in ('GI','SLT') else 165
        d.line((xp(a)+5,level,xp(b)-5,level),fill='#333333',width=3)
        label=labels[name]+'*'
        center=xp((a+b)/2)
        f=font(23,True); box=d.textbbox((0,0),label,font=f); width=box[2]-box[0]
        if name in ('GI','SLT'):
            centered(center,level-32,label,23,True)
        else:
            d.rectangle((center-width/2-10,level-7,center+width/2+10,level+25),fill='white')
            centered(center,level-7,label,23,True)
        if a>grid[0]+.001:
            xx=xp(a)
            for yy in range(top,bottom,20):d.line((xx,yy,xx,min(yy+12,bottom)),fill='#333333',width=2)
            d.polygon([(xx,top-10),(xx-6,top),(xx+6,top)],fill='#333333')
    for v,label in zip(info.get('valleys',[]),['S1','S2','S3','S4','S5','T','L1','L2','L3','L4','L5']):
        centered(xp(v),yp(340)+5,label,13,color='#666666')
    d.text((left,810),'* Model-completed inputs; estimates are not recovered measurements. Phase labels are not validated ground truth.',font=font(18,True),fill='#744600')
    d.text((left,840),'Dashed lines: reviewed phase cuts | S1-S5, T, L1-L5: selected valley labels',font=font(17),fill='#555555')
    d.text((left,868),quality[:240],font=font(16),fill='#555555')
    temporary=path.with_suffix('.pending.png'); im.save(temporary)
    for attempt in range(10):
        try:temporary.replace(path);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.3)
    return transforms
