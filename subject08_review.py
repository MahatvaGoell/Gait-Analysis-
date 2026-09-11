"""Full timestamp-based Subject 08 plots with explicitly provisional events.

No clinical gait labels are inferred from amplitude alone. All raw channels
retain their polarity. Per-stream times are relative to corresponding start
triggers; missing intervals are gaps, not compressed or filled in plots.
"""
from pathlib import Path
import csv
import json
import math
import time
import numpy as np
from PIL import Image, ImageDraw
from generate_qs_plots import ROOT, DATASET_ROOT, load_stream, split_source_files, unit_trial_starts, font

OUT = ROOT / 'tempgraphs'
DT = .01
COLORS = ['#4e79a7','#c27d3b','#d4b640','#9f77b0','#64a343','#42a6b2','#bd627b','#28758a','#925a37','#d4a333']

def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def grid_values(t, values, grid):
    # First occurrence of a timestamp wins; nonmonotonic packets are audited.
    order = np.argsort(t, kind='stable')
    times, indices = np.unique(t[order], return_index=True)
    values = values[order[indices]]
    z = np.column_stack([np.interp(grid, times, values[:, k]) for k in range(values.shape[1])])
    z[(grid < times[0]) | (grid > times[-1])] = np.nan
    for a, b in zip(times[:-1], times[1:]):
        if b-a > .10:
            z[(grid>a) & (grid<b)] = np.nan
    return z

def activity(z):
    # Local range measures ongoing movement rather than sustained force level.
    filled = np.where(np.isfinite(z), z, 0)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(filled, ((15,15),(0,0)), mode='edge'),31,axis=0)
    span = np.percentile(windows,90,axis=2)-np.percentile(windows,10,axis=2)
    head = span[:min(75,len(span))]
    base = np.median(head,axis=0)
    noise = np.median(np.abs(head-base),axis=0)*1.4826
    threshold = np.maximum(base+6*np.maximum(noise,.15), np.maximum(np.percentile(span,90,axis=0)*.20, 1.0))
    score = np.percentile(span/threshold,75,axis=1)
    score[~np.all(np.isfinite(z),axis=1)] = np.nan
    return score

def events(z, grid):
    f = activity(z[:,:8]); p = activity(z[:,8:])
    score = np.maximum(f,p)
    mask = np.isfinite(score) & (score>1)
    # Join brief pauses inside a moving sequence, not long recording gaps.
    for a,b in runs(~mask):
        if a>0 and b<len(mask) and b-a<=35 and np.all(np.isfinite(score[a:b])):
            mask[a:b]=True
    bouts = [(a,b) for a,b in runs(mask) if b-a>=30]
    info={'onset_seconds':'','settled_seconds':'','gt_start_seconds':'', 'review':'', 'walk_stage_labels':'unverified: no per-trial step-length event annotations'}
    markers=[]
    if not bouts:
        info['review']='movement not reliably resolved; manual review needed'
        return markers,info,score
    # Ignore isolated brief activity if not followed by sustained movement.
    substantial=[(a,b) for a,b in bouts if b-a>=100]
    if not substantial:
        info['review']='only brief activity; manual review needed'
        return markers,info,score
    first,last=substantial[0][0],bouts[-1][1]
    onset=max(0,first-15)
    if onset<50:
        info['review']='initial standing not resolved; recording may begin during movement'
    else:
        markers.append((float(grid[onset]),'Movement onset*'))
        info['onset_seconds']=round(float(grid[onset]),2)
    # Settling is supported only with >=0.8 s of observed quiet at the tail.
    end=min(len(grid)-1,last+15)
    if end<len(grid)-80 and np.all(np.isfinite(score[end:])) and np.mean(score[end:]<1)>.9:
        markers.append((float(grid[end]),'Standing resumes*'))
        info['settled_seconds']=round(float(grid[end]),2)
    else:
        info['review']+='; final standing not confirmed within recorded trial'
    info['review']=info['review'].strip('; ') or 'provisional onset and settling; inspect both sensor groups'
    return markers,info,score

def render(path, title, grid, z, markers, info, quality, phases=None, segmentation_note=''):
    w,h=2200,880 if phases is not None else 850
    im=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(im)
    left,right,top,bottom=120,1940,160,650
    xmax=math.ceil(grid[-1]/2)*2
    def xp(t): return left+t/xmax*(right-left)
    def yp(v): return bottom-v/500*(bottom-top)
    d.text((left,16),title,font=font(25,True),fill='black')
    d.text((left,51),'Full trial | original elapsed time | 8 FMG channels + CoP + vGRF',font=font(18),fill='#333333')
    d.line((left,top,left,bottom,right,bottom),fill='#333333',width=2)
    for y in range(0,501,100):
        d.line((left,yp(y),right,yp(y)),fill='#dddddd',width=1)
        d.text((left-48,yp(y)-11),str(y),font=font(17),fill='black')
    for t in np.arange(0,xmax+.1,2):
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
        anchor=[200,170,145,120,100,80,60,40,400,380][k]
        base=float(np.nanmedian(x[base_mask])) if np.any(np.isfinite(x[base_mask])) else float(np.nanmedian(x))
        delta=x-base
        low,high=(10,340) if k<8 else ((365,435) if k==8 else (345,425))
        up=max(0,float(np.nanmax(delta))); down=max(0,float(-np.nanmin(delta)))
        gain=min(1.0 if k<8 else (2.5 if k==8 else .08), (high-anchor)/max(up,1e-9),(anchor-low)/max(down,1e-9))
        display=anchor+delta*gain
        for a,b in runs(valid):
            if b-a>=2:
                d.line([(xp(float(grid[j])),yp(float(display[j]))) for j in range(a,b)],fill=COLORS[k],width=2)
        transforms.append({'signal':f'Channel {k+1}' if k<8 else ['CoP','vGRF'][k-8], 'baseline':base,'gain':gain,'offset':anchor})
        yy=top+k*30
        d.line((1970,yy+10,2000,yy+10),fill=COLORS[k],width=3)
        d.text((2010,yy),transforms[-1]['signal'],font=font(16),fill='black')
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
        d.text((left,754),'* PROVISIONAL sequence fit. Short-to-long order is assumed as requested; step length and heel strikes are not verified.',font=font(18,True),fill='#744600')
        d.text((left,783),'QS: quiet standing | GI: initiation | SSSW: short-step walking | SLT: transition | SSLW: long-step walking | GT: stopping + ending',font=font(16),fill='#444444')
        d.text((left,811),segmentation_note[:210],font=font(16),fill='#744600')
        d.text((left,839),quality[:215],font=font(15),fill='#555555')
    path.parent.mkdir(parents=True,exist_ok=True)
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

def write_rows(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r})); writer.writeheader(); writer.writerows(rows)

def generate(record_filter=None,trial_filter=None,overwrite=False,side_filter=None):
    fm,ins=split_source_files(DATASET_ROOT/'Sub08_H')
    rows=[]; transforms=[]
    for record,fp in fm.items():
        if record_filter and record!=record_filter: continue
        f=load_stream(fp,18,'FMG')
        for side in ('L','R'):
            if side_filter and side!=side_filter: continue
            i=load_stream(ins[(record,side)],4,'insole',side)
            for a,b,trial in unit_trial_starts(f):
                if trial_filter and trial!=trial_filter: continue
                edge=2*(trial-1)
                if b is None or edge+1>=len(i.rising_edges): continue
                c,e=i.rising_edges[edge:edge+2]
                ft=f.values[a:b+1,0]-f.values[a,0]; it=i.values[c:e+1,0]-i.values[c,0]
                fv=f.values[a:b+1,2:10] if side=='L' else f.values[a:b+1,10:18]
                iv=i.values[c:e+1,2:4]
                grid=np.arange(0,max(ft[-1],it[-1])+.005,DT)
                z=np.column_stack((grid_values(ft,fv,grid),grid_values(it,iv,grid)))
                markers,info,score=events(z,grid)
                error=float(it[-1]-ft[-1]); gaps_f=int(np.sum(np.diff(ft)>.10)); gaps_i=int(np.sum(np.diff(it)>.10))
                backwards=int(np.sum(np.diff(it)<=0))+int(np.sum(np.diff(ft)<=0))
                quality=f'Trigger-end difference {error:+.3f} s | gaps >0.10 s: FMG {gaps_f}, insole {gaps_i} | repeated/reversed timestamps {backwards}'
                if abs(error)>.25: info['review']+='; trigger timing mismatch'
                if gaps_f or gaps_i or backwards: info['review']+='; timestamp quality review'
                filename=f'Sub08_H/{record}/trial_{trial:02d}_{side.lower()}_complete.png'
                tr=render(OUT/filename,f'Sub08_H | {record} | Trigger-pair trial {trial:02d} | {side}',grid,z,markers,info,quality)
                for t in tr: transforms.append(dict(record=record,side=side,trial=trial,**t))
                rows.append(dict(record=record,side=side,trial=trial,png=filename,full_fmg_seconds=float(ft[-1]),full_insole_seconds=float(it[-1]),trigger_end_error_seconds=error,fmg_gaps=gaps_f,insole_gaps=gaps_i,nonmonotonic_or_duplicate_timestamps=backwards,**info))
    OUT.mkdir(exist_ok=True)
    if not record_filter and not trial_filter and not side_filter:
        write_rows(OUT/'complete_trial_manifest.csv',rows)
        write_rows(OUT/'display_transforms.csv',transforms)
        audit_keys=('record','side','trial','trigger_end_error_seconds','fmg_gaps','insole_gaps','nonmonotonic_or_duplicate_timestamps')
        write_rows(OUT/'synchronization_report.csv',[{k:r[k] for k in audit_keys} for r in rows])
        review=OUT/'review_sheets'; review.mkdir(exist_ok=True)
        for page in range(math.ceil(len(rows)/8)):
            sheet=Image.new('RGB',(1600,4*330),'white')
            for j,row in enumerate(rows[page*8:page*8+8]):
                with Image.open(OUT/row['png']) as im:
                    im.thumbnail((800,330)); sheet.paste(im,((j%2)*800,(j//2)*330))
            sheet.save(review/f'page_{page+1:02d}.png')
    print(json.dumps({'graphs':len(rows),'onset_candidates':sum(r['onset_seconds']!='' for r in rows),'settling_candidates':sum(r['settled_seconds']!='' for r in rows),'shortest_seconds':min(r['full_fmg_seconds'] for r in rows),'longest_seconds':max(r['full_fmg_seconds'] for r in rows)},indent=2))
    return rows
