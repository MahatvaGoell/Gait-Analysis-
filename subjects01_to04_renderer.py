"""Subject 01-04 renderer: same style, adaptive ticks for long unmatched context."""
import math
import time
import numpy as np
from PIL import Image, ImageDraw
from generate_qs_plots import font
from subject08_review import COLORS, runs

def render(path, title, grid, z, markers, info, quality, phases=None, segmentation_note='', imputed=None):
    if imputed is None:imputed=np.zeros_like(z,dtype=bool)
    completed=info.get('view_mode')=='model_completed'
    w,h=2200,880 if phases is not None else 850
    im=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(im)
    left,right,top,bottom=120,1940,160,650
    xmax=math.ceil(grid[-1]/2)*2
    def xp(t): return left+t/xmax*(right-left)
    def yp(v): return bottom-v/500*(bottom-top)
    d.text((left,16),title,font=font(25,True),fill='black')
    subtitle='MODEL-COMPLETED | dark lines: source grid | pale lines: estimates (may include wholly absent sensors)' if completed else 'Full trial | original elapsed time | dashed: short-gap estimates | blank: unrecoverable data'
    d.text((left,51),subtitle,font=font(18,completed),fill='#744600' if completed else '#333333')
    d.line((left,top,left,bottom,right,bottom),fill='#333333',width=2)
    for y in range(0,501,100):
        d.line((left,yp(y),right,yp(y)),fill='#dddddd',width=1)
        d.text((left-48,yp(y)-11),str(y),font=font(17),fill='black')
    tick_step=max(2,2*math.ceil(xmax/60))
    for t in np.arange(0,xmax+.1,tick_step):
        d.line((xp(t),bottom,xp(t),bottom+9),fill='black',width=2)
        d.text((xp(t)-10,bottom+15),f'{t:g}',font=font(17),fill='black')
    d.text((760,700),'Elapsed time from trial-start trigger (seconds)',font=font(21),fill='black')
    lab=Image.new('RGBA',(480,32),(255,255,255,0)); ImageDraw.Draw(lab).text((0,0),'Scaled display amplitude (offset per signal)',font=font(19),fill='black')
    lab=lab.rotate(90,expand=True); im.paste(lab,(25,220),lab)
    # Display-only affine transforms; no clipping, polarity flip or per-phase reset.
    onset=info['onset_seconds']; base_end=float(onset) if onset!='' else .75
    base_mask=grid<max(.25,min(base_end,.75))
    transforms=[]
    for k in range(10):
        x=z[:,k]; valid=np.isfinite(x)
        if not valid.any():
            name=f'Channel {k+1}' if k<8 else ['CoP','vGRF'][k-8]
            transforms.append({'signal':name,'baseline':None,'gain':None,'offset':None})
            d.text((1970,top+k*30),name+' (missing)',font=font(14),fill='#999999')
            continue
        anchor=[200,170,145,120,100,80,60,40,400,380][k]
        base=float(np.nanmedian(x[base_mask])) if np.any(np.isfinite(x[base_mask])) else float(np.nanmedian(x))
        delta=x-base
        low,high=(10,340) if k<8 else ((365,435) if k==8 else (345,425))
        up=max(0,float(np.nanmax(delta))); down=max(0,float(-np.nanmin(delta)))
        gain=min(1.0 if k<8 else (2.5 if k==8 else .08), (high-anchor)/max(up,1e-9),(anchor-low)/max(down,1e-9))
        display=anchor+delta*gain
        if completed:
            from PIL import ImageColor
            rgb=ImageColor.getrgb(COLORS[k])
            pale=tuple(int(.55*c+.45*255) for c in rgb)
            d.line([(xp(float(t)),yp(float(v))) for t,v in zip(grid,display)],fill=pale,width=2)
        for a,b in runs(valid & ~imputed[:,k]):
            if b-a>=2:
                d.line([(xp(float(grid[j])),yp(float(display[j]))) for j in range(a,b)],fill=COLORS[k],width=2)
        for a,b in runs(imputed[:,k] if not completed else np.zeros(len(grid),bool)):
            # Dash by screen distance, so small repairs remain distinguishable.
            aa=max(0,a-1);bb=min(len(grid),b+1)
            distance=0.
            for j in range(aa,bb-1):
                p=(xp(float(grid[j])),yp(float(display[j])))
                q=(xp(float(grid[j+1])),yp(float(display[j+1])))
                if int(distance/4)%2==0:d.line((*p,*q),fill=COLORS[k],width=2)
                distance+=math.hypot(q[0]-p[0],q[1]-p[1])
        transforms.append({'signal':f'Channel {k+1}' if k<8 else ['CoP','vGRF'][k-8], 'baseline':base,'gain':gain,'offset':anchor})
        yy=top+k*30
        d.line((1970,yy+10,2000,yy+10),fill=COLORS[k],width=3)
        suffix=' [est.]' if imputed[:,k].all() else ''
        d.text((2010,yy),transforms[-1]['signal']+suffix,font=font(15),fill='black')
    for idx,(t,label) in enumerate(markers if phases is None else []):
        xx=xp(t)
        for yy in range(top,bottom,18): d.line((xx,yy,xx,min(yy+10,bottom)),fill='#333333',width=2)
        d.text((max(left,min(xx-100,right-220)),88+idx*28),f'{label} {t:.2f} s',font=font(17,True),fill='black')
    if phases is None and info['onset_seconds']!='':
        start=float(info['onset_seconds']); end=float(info['settled_seconds']) if info['settled_seconds']!='' else grid[-1]
        d.text((left+12,130),'Initial standing',font=font(16),fill='#444444')
        d.text((xp((start+end)/2)-125,130),'Walking / transitions: review',font=font(16),fill='#444444')
    if phases is None:
        d.text((left,754),'* Signal-derived candidates, not validated gait-event labels. Short/long-step transitions require trial annotations.',font=font(17),fill='#444444')
        d.text((left,783),quality[:215],font=font(15),fill='#555555')
    else:
        if phases:
            for phase in phases:
                a,b=phase['start_seconds'],phase['end_seconds']
                label=phase['phase']+'*'
                center=xp((a+b)/2)
                level=108 if phase['phase'] in ('GI','SLT') else 140
                d.line((xp(a)+3,level,xp(b)-3,level),fill='#333333',width=2)
                box=d.textbbox((0,0),label,font=font(22,True))
                d.rectangle((center-(box[2]+12)/2,level-25,center+(box[2]+12)/2,level-2),fill='white')
                d.text((center-box[2]/2,level-28),label,font=font(22,True),fill='black')
                if a>0:
                    for yy in range(top,bottom,18):
                        d.line((xp(a),yy,xp(a),min(yy+10,bottom)),fill='#444444',width=2)
                    stamp=f'{a:.2f}s'
                    tb=d.textbbox((0,0),stamp,font=font(13))
                    d.text((xp(a)-tb[2]/2,top-18 if phase['phase'] in ('GI','SLT','GT') else top+3),stamp,font=font(13),fill='#555555')
        else:
            d.text((left,110),'UNRESOLVED - this recording does not support a complete six-phase estimate',font=font(23,True),fill='#9a4c00')
        warning='* MODEL-ESTIMATED reconstruction, not recovered measurements. Phase labels are provisional; not classification ground truth.' if completed else '* PROVISIONAL sequence fit. Short-to-long order is assumed as requested; step length and heel strikes are not verified.'
        d.text((left,754),warning,font=font(18,True),fill='#744600')
        d.text((left,783),'QS: quiet standing | GI: initiation | SSSW: short-step walking | SLT: transition | SSLW: long-step walking | GT: stopping + ending',font=font(16),fill='#444444')
        d.text((left,811),segmentation_note[:210],font=font(16),fill='#744600')
        d.text((left,839),quality[:215],font=font(15),fill='#555555')
    assert path.parent.is_dir(), 'Refinement must use existing folders only'
    temporary=path.with_suffix('.pending.png')
    im.save(temporary)
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except OSError:
            if attempt==4: raise
            time.sleep(.2)
    return transforms
