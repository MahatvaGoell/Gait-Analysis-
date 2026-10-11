"""Noise-aware, source-preserving CoP review for subjects 03 and 04 only.

This extends the CoP valley-to-peak convention, not physiological ground truth.
All thresholds are detection heuristics. Raw exports are never overwritten.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import hashlib
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from PIL import Image, ImageDraw

import refine_sub08_h_valleys as old

SUBJECTS = ("Sub03_H", "Sub04_H")


def clean_signals(values, flags):
    """Symmetric despiking and 3-sample smoothing; preserve real unloaded lows.

    No reference-shaped valleys, resampling, time shifts, or invented cycles.
    Repaired isolated spikes are tracked separately from existing estimates.
    """
    z = old.finite_fill(values)
    repaired = ~np.isfinite(values)
    for k in range(10):
        raw = z[:, k].copy()
        local = median_filter(raw, size=9, mode="nearest")
        mad = median_filter(np.abs(raw-local), size=9, mode="nearest")
        span = max(float(np.percentile(raw, 95)-np.percentile(raw, 5)), 1e-6)
        suspect = np.abs(raw-local) > np.maximum(8*1.4826*mad, .12*span)
        # Sustained changes and true unloaded intervals must survive despiking.
        for a, b in old.runs(suspect):
            if b-a <= 2 and a>0 and b<len(raw) and abs(raw[a-1]-raw[b]) < .12*span:
                if k in (8,9) and np.min(raw[a:b]) <= .02*max(float(np.percentile(raw,95)),1):
                    continue
                z[a:b,k] = local[a:b]
                repaired[a:b,k] = True
        if k == 8:
            # Existing model/short-gap estimates cannot create new oscillations.
            z[:,k] = old.bridge_short_gaps(z[:,k], flags[:,k], longest=20)
        z[:,k] = old.smooth(z[:,k],3)
    return z, repaired


def regular_run(events, amplitudes, stride=None):
    """Select a measured cadence train, never a requested number of steps."""
    if len(events)<3:
        return list(range(len(events)))
    diff=np.diff(events)
    plausible=diff[(diff>=40)&(diff<=250)]
    if not len(plausible):
        return []
    stride=float(np.median(plausible)) if stride is None else float(stride)
    groups=[]; group=[0]
    for k, delta in enumerate(diff,1):
        if .50*stride <= delta <= 1.70*stride:
            group.append(k)
        else:
            groups.append(group); group=[k]
    groups.append(group)
    return max(groups,key=lambda g:(len(g),sum(amplitudes[i] for i in g)))


def cadence(z):
    """Dominant repeating CoP period; no timing borrowed from another trial."""
    cop=z[:,8]
    detrended=cop-old.smooth(cop,min(301,len(cop)//2*2-1))
    energy=max(float(np.dot(detrended,detrended)),1e-9)
    lags=np.arange(40,min(251,len(cop)//3))
    corr=np.array([np.dot(detrended[:-k],detrended[k:])/energy for k in lags])
    p,_=find_peaks(corr,distance=20)
    if not len(p):return 120.,0.
    best=int(p[np.argmax(corr[p])])
    return float(lags[best]),float(corr[best])


def load_train(z):
    load=old.smooth(median_filter(z[:,9],size=5,mode="nearest"),5)
    span=max(float(np.percentile(load,95)-np.percentile(load,5)),1e-6)
    peaks, prop=find_peaks(load,prominence=.20*span,distance=40)
    group=regular_run(peaks,prop['prominences'])
    return peaks[group].tolist()


def find_stances(z, flags, first, last, stride, window=None):
    """Find narrow CoP minima and following CoP peaks within a walking train."""
    cop=z[:,8]
    a,b=(1,len(cop)-1) if window is None else window
    a=max(1,int(a));b=min(len(cop)-1,int(b))
    portion=cop[a:b]
    span=max(float(np.percentile(portion,95)-np.percentile(portion,5)),1e-6)
    # 3 samples, not the old 11-sample median that erases narrow true lows.
    s=old.smooth(median_filter(cop,size=3,mode="nearest"),3)
    # Measure prominence on the complete trace before applying the walking
    # window: cropping first can erase a genuine shallow initial valley.
    valleys,_=find_peaks(-s,prominence=.18*span,distance=max(25,int(.65*stride)))
    valleys=valleys[(valleys>=a)&(valleys<b)].tolist()
    stances=[]
    for k,v in enumerate(valleys):
        following=valleys[k+1] if k+1<len(valleys) else min(b,int(v+1.2*stride))
        limit=min(following-3,int(v+1.15*stride),len(cop)-1)
        if limit<=v+4:
            continue
        peak=v+int(np.argmax(s[v:limit+1]))
        climb=float(s[peak]-s[v])
        if climb<.23*span or peak-v<4:
            continue
        load=z[:,9]
        loadspan=max(float(np.percentile(load,95)-np.percentile(load,1)),1e-6)
        local_load=load[max(0,v-10):min(len(load),peak+10)]
        if np.ptp(local_load)<.10*loadspan:
            continue  # standing CoP jitter is not a walking stance
        # A zero/low plateau marks an unloaded interval. Place at the end of
        # that low immediately before the measured CoP climb, not its centre.
        threshold=s[v]+.08*climb
        end=v
        while end+1<peak and s[end+1]<=threshold:
            end+=1
        v=end
        stances.append((int(v),int(peak),'filtered_cop_valley'))
    if len(stances)>=3:
        group=regular_run(np.array([v for v,_,_ in stances]),np.array([cop[p]-cop[v] for v,p,_ in stances]),stride)
        stances=[stances[i] for i in group]
        # The initial weight shift can make a small CoP climb before the first
        # full stride. Use the same 0.7-stride exclusion as the original method.
        while len(stances)>3 and stances[1][0]-stances[0][0]<.70*np.median(np.diff([v for v,_,_ in stances[1:]])):
            stances.pop(0)
    # The first limb may climb out of quiet standing without an enclosed dip.
    if stances and stances[0][0]-first>.55*stride:
        end=stances[0][0]-int(.25*stride)
        start=max(a,int(first-1.3*stride))
        if end>start+8:
            peak=start+int(np.argmax(s[start:end]))
            v=start+int(np.argmin(s[start:peak+1]))
            climb=s[peak]-s[v]
            if climb>=.25*span:
                low=np.flatnonzero(s[v:peak+1]<=s[v]+.08*climb)
                v+=int(low[-1])
                if .65*stride<=stances[0][0]-v<=1.5*stride and peak>v+4:
                    stances.insert(0,(int(v),int(peak),'filtered_first_cop_climb'))
    # Recheck after first-climb insertion: it must not reintroduce a small
    # preparatory weight shift that was rejected above. A weak, short initial
    # climb is not a full stance merely because it falls inside the window.
    while len(stances)>3:
        v,p,_=stances[0]
        gaps=np.diff([x[0] for x in stances[1:]])
        typical=float(np.median(gaps))
        climbs=[s[pp]-s[vv] for vv,pp,_ in stances[1:]]
        weak=s[p]-s[v]<.45*float(np.median(climbs))
        gap=stances[1][0]-v
        if gap<.70*typical or (weak and gap<1.65*typical):
            stances.pop(0)
        else:
            break
    return stances,span


def local_onset(left,right,first,stride):
    """Last locally quiet sample before the first recurring walking event."""
    a=max(1,int(first-1.0*stride)); end=max(a+1,int(first))
    before=max(0,a-int(.7*stride))
    scores=[]
    for z in (left,right):
        # Robust across channels: a single faulty sensor cannot set GI.
        for cols in (range(8),[9]):
            columns=[]
            for k in cols:
                x=old.smooth(z[:,k],7)
                base=x[before:a+1]
                noise=max(float(1.4826*np.median(np.abs(base-np.median(base)))),1e-6)
                span=max(float(np.percentile(x,95)-np.percentile(x,5)),1e-6)
                columns.append(np.abs(x-np.median(base))/max(5*noise,.10*span))
            scores.append(np.median(columns,axis=0))
    score=np.max(scores,axis=0)
    active=score[a:end]>1
    for x,y in old.runs(active):
        if y-x>=10:
            return max(1,a+x-3),True
    return max(1,int(first-.35*stride)),False


def final_contact(z,stances,stride):
    """A terminal boundary needs an actual loading change, not cadence alone."""
    expected=int(stances[-1][0]+np.median(np.diff([v for v,_,_ in stances[-6:]])))
    reach=int(.30*stride)
    slope=np.gradient(old.smooth(z[:,9],5))
    a,b=max(1,expected-reach),min(len(z)-2,expected+reach)
    if b<=a:
        return None
    contact=a+int(np.argmax(slope[a:b]))
    regular=[];leads=[]
    for v,_,_ in stances[-6:]:
        x,y=max(0,v-int(.2*stride)),min(len(z),v+int(.2*stride))
        k=x+int(np.argmax(slope[x:y]))
        regular.append(slope[k]);leads.append(v-k)
    if slope[contact]<.18*max(float(np.median(regular)),1e-6):
        return None
    contact+=int(np.median(leads))
    if not stances[-1][1]<contact<len(z)-5:
        return None
    return contact,'terminal_loading_contact_candidate'


def segment(subject,record,trial,left,right,fl,fr):
    """Independent per-window timing, extra steps after the protocol sequence."""
    if subject not in SUBJECTS:
        raise ValueError('This refinement only applies to Sub03_H and Sub04_H')
    with (old.SUBJECTS[subject][1]/'recording_windows.csv').open(encoding='utf-8',newline='') as f:
        window=next(r for r in csv.DictReader(f) if r['record']==record and int(r['trial'])==trial)
    cleaned={}; repaired={}; flags={'L':fl,'R':fr}
    for side,values in [('L',left),('R',right)]:
        cleaned[side],repaired[side]=clean_signals(values,flags[side])
    result={'source_subject':subject,'source_window_method':window['boundary_method'],'rise':0,'rest':len(left)-1}
    if 'unpaired' in window['boundary_method'].lower():
        for side in ('L','R'):
            result[side]=dict(stances=[],layout=None,reason='unpaired trigger context; protocol boundaries unverified',estimated_cuts=[],review_notes=[],extra_steps=[],display_values=cleaned[side],repaired=repaired[side])
        return result
    trains={s:load_train(z) for s,z in cleaned.items()}
    reference=max(cleaned,key=lambda s:cadence(cleaned[s])[1])
    stride,_=cadence(cleaned[reference])
    ref_support=trains[reference]
    ref_stances=[]
    if len(ref_support)>=4:
        ref_stances,_=find_stances(cleaned[reference],flags[reference],ref_support[0],ref_support[-1],stride)
    main_window=None
    if len(ref_stances)>=4:
        main_window=(ref_stances[0][0]-.80*stride,ref_stances[-1][0]+.75*stride)
    for side,other in [('L','R'),('R','L')]:
        own=trains[side]; partner=trains[other]
        support=own if len(own)>=4 else partner
        seg=dict(stances=[],layout=None,reason='',estimated_cuts=[],review_notes=[],extra_steps=[],display_values=cleaned[side],repaired=repaired[side])
        result[side]=seg
        if len(support)<4:
            seg['reason']='fewer than four recurring loading events; walking train uncertain'
            continue
        first,last=support[0],support[-1]
        stances,span=find_stances(cleaned[side],flags[side],first,last,stride,main_window)
        seg['stances']=stances
        seg['stride_samples']=stride
        seg['load_peaks']=own
        seg['cop_span']=span
    # Shared physical onset, localized around supported CoP and load movement.
    good=[result[s]['stances'][0][0] for s in ('L','R') if len(result[s]['stances'])>=4]
    if not good:
        for s in ('L','R'):
            result[s]['reason']=result[s]['reason'] or 'no recurring CoP stance train'
        return result
    stride=float(np.median([result[s]['stride_samples'] for s in ('L','R') if 'stride_samples' in result[s]]))
    anchor=ref_stances[0][0] if ref_stances else min(good)
    rise,onset_supported=local_onset(cleaned['L'],cleaned['R'],anchor,stride)
    result['rise']=rise
    # Reject standing minima preceding the sustained walking train.
    for side in ('L','R'):
        seg=result[side]; stances=seg['stances']
        if not stances:
            continue
        stances=[x for x in stances if x[0]>=rise]
        seg['stances']=stances
    for side,other in [('L','R'),('R','L')]:
        seg=result[side]; stances=seg['stances']; z=cleaned[side]
        if len(stances)<11:
            seg['reason']=f'{len(stances)} supported CoP stances; fewer than 11 for 5-1-5'
            continue
        end_cut=None
        if len(stances)>11:
            end_cut=(stances[11][0],'filtered_cop_valley')
        else:
            terminal=final_contact(z,stances,stride)
            if terminal is None:
                seg['reason']='final landing cannot be established inside recording'
                continue
            end_cut=terminal
        lay=old.layout(stances,end_cut,rise,len(left))
        if not lay:
            seg['reason']='CoP candidates overlap or do not form an ordered 5-1-5 layout'
            continue
        seg['layout']=lay
        assert not old.check(lay)
        # All actual additional complete cycles remain after L5, before GT.
        # The last CoP notch may be a stopping adjustment, not an extra stride.
        extras=[]
        typical_climb=float(np.median([z[p,8]-z[v,8] for v,p,_ in stances[:11]]))
        for k in range(11,len(stances)-1):
            v,p,kind=stances[k]; nxt=stances[k+1][0]
            if v<p<nxt and .65*stride<=nxt-v<=1.6*stride and z[p,8]-z[v,8]>=.5*typical_climb:
                extras.append(('EXTRA',len(extras)+1,v,p,kind,nxt))
            else:
                break  # do not bridge an unsupported closing adjustment
        seg['extra_steps']=extras
        if extras:
            seg['GT_start_sample']=extras[-1][5]
        else:
            seg['GT_start_sample']=lay[0][5]
        valleys=[x[2] for x in lay[1]]+[lay[0][5]]
        seg['estimated_cuts']=[f'v{k}' for k,v in enumerate(valleys) if old.near(flags[side][:,8],v)]
        if len(seg['estimated_cuts'])>=6:
            seg['layout']=None; seg['extra_steps']=[]
            seg['reason']='most CoP cut locations are estimated, not measured'
        elif seg['estimated_cuts']:
            seg['review_notes'].append('Some boundaries intersect existing CoP estimates')
        if seg['layout']:
            if not onset_supported or lay[0][1]!=rise:
                seg['review_notes'].append('GI onset is estimated; verify against foot-lift evidence')
            durs=np.diff([x[2] for x in lay[1]]+[lay[0][5]])
            if max(durs)>1.5*np.median(durs):
                seg['review_notes'].append('An interval is unusually long; inspect stance evidence')
            climbs=np.array([z[p,8]-z[v,8] for v,p,_ in stances[:11]])
            if np.min(climbs)<.35*np.median(climbs):
                seg['review_notes'].append('A CoP climb is weak relative to this trial')
            if lay[0][2]-lay[0][1]>1.5*stride:
                seg['review_notes'].append('GI onset remains uncertain')
    if all(result[s]['layout'] for s in ('L','R')):
        offset=abs(result['L']['layout'][0][3]-result['R']['layout'][0][3])
        if not .25*stride<=offset<=.80*stride:
            for s in ('L','R'):
                result[s]['review_notes'].append('Left/right phase counts need review against limb timing')
    return result


def draw(path,title,times,seg,flags):
    """Complete timeline; additional observed cycles have their own closing span."""
    z=seg['display_values']; n=len(z)
    w,h=2200,800
    im=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(im)
    left,right,top,bottom=138,2015,190,620
    xmax=max(2200,math.ceil(n/200)*200)
    xp=lambda x:left+x/xmax*(right-left)
    yp=lambda y:bottom-y/500*(bottom-top)
    d.text((left,18),title,font=old.font(23,True),fill='#111111')
    d.text((left,55),'Light smoothing / isolated-spike removal | original sample times | pale: existing estimates or spike repairs',font=old.font(16),fill='#795f27')
    d.line((left,top,left,bottom,right,bottom),fill='#555555',width=2)
    for y in range(0,501,100):
        d.line((left-7,yp(y),left,yp(y)),fill='#555555',width=2)
        text=str(y);box=d.textbbox((0,0),text,font=old.font(18))
        d.text((left-15-box[2],yp(y)-11),text,font=old.font(18),fill='#444444')
    target=xmax/12;mag=10**math.floor(math.log10(target))
    ticks=200 if xmax<=3000 else int(next(v*mag for v in (1,2,5,10) if v*mag>=target))
    for x in range(0,xmax+1,ticks):
        d.line((xp(x),bottom,xp(x),bottom+7),fill='#555555',width=2)
        text=str(x);box=d.textbbox((0,0),text,font=old.font(18))
        d.text((xp(x)-box[2]/2,bottom+13),text,font=old.font(18),fill='#444444')
    text='Samples @ 100 Hz (original trial timeline)';box=d.textbbox((0,0),text,font=old.font(20,True))
    d.text(((left+right-box[2])/2,bottom+46),text,font=old.font(20,True),fill='#111111')
    label=Image.new('RGBA',(490,35),(255,255,255,0))
    ImageDraw.Draw(label).text((0,0),'Display amplitude (normalized and offset)',font=old.font(19,True),fill='#111111')
    label=label.rotate(90,expand=True);im.paste(label,(35,int((top+bottom-490)/2)),label)
    for k in range(10):
        x=z[:,k];anchor=old.ANCHORS[k];low,high,max_gain=old.BANDS[k]
        base=float(np.median(x[:min(n,75)]))
        up=max(0.,float(np.max(x)-base));down=max(0.,float(base-np.min(x)))
        gain=min(max_gain,(high-anchor)/max(up,1e-9),(anchor-low)/max(down,1e-9))
        display=anchor+(x-base)*gain
        pts=[(xp(j),yp(v)) for j,v in enumerate(display)]
        pale_mask=flags[:,k]|seg['repaired'][:,k]
        for mask,col in [(pale_mask,old.pale(old.CHANNEL_COLORS[k])),(~pale_mask,old.CHANNEL_COLORS[k])]:
            for a,b in old.runs(mask):
                if b-a>=1:d.line(pts[max(0,a-1):min(n,b+1)],fill=col,width=2)
        yy=204+k*28
        d.line((2042,yy+8,2074,yy+8),fill=old.CHANNEL_COLORS[k],width=2)
        d.text((2086,yy),old.CHANNEL_NAMES[k],font=old.font(15),fill='#222222')
    lay=seg['layout']
    if lay:
        bounds,steps,_=lay
        gt=seg['GT_start_sample']
        phases=list(zip(old.PHASES[:-1],bounds[:-2],bounds[1:-1]))
        if gt>bounds[5]:phases.append(('EXTRA',bounds[5],gt))
        phases.append(('GT',gt,n))
        for name,a,b in phases:
            x0,x1=xp(a),xp(min(b,n-1));centre=(x0+x1)/2
            text=('Extra step' if len(seg['extra_steps'])==1 else 'Extra steps') if name=='EXTRA' else old.PHASE_TITLES[name]
            text+='*';box=d.textbbox((0,0),text,font=old.font(19,True))
            if name in ('GI','SLT','EXTRA'):
                yy=83 if name!='EXTRA' else 117
                d.text((centre-box[2]/2,yy),text,font=old.font(19,True),fill='#111111')
                half=max((x1-x0)/2-4,box[2]/2+2)
                d.line((centre-half,yy+30,centre+half,yy+30),fill='#222222',width=2)
            else:
                d.text((centre-box[2]/2,151),text,font=old.font(19,True),fill='#111111')
                gap=box[2]/2+12
                if centre-gap>x0+8:
                    d.line((x0+5,162,centre-gap,162),fill='#222222',width=2)
                    d.line((centre+gap,162,x1-5,162),fill='#222222',width=2)
            if a>0:
                x=xp(a)
                for y in range(top+6,bottom,15):d.line((x,y,x,min(y+8,bottom)),fill='#333333',width=2)
                d.polygon([(x,top-7),(x-5,top+4),(x+5,top+4)],fill='#333333')
        for phase,num,v,p,kind,nxt in steps+seg['extra_steps']:
            text=f'E{num}' if phase=='EXTRA' else old.step_name(phase,num)
            box=d.textbbox((0,0),text,font=old.font(13))
            d.text((xp((p+nxt)/2)-box[2]/2,yp(330)),text,font=old.font(13),fill='#666666')
        notes=seg['review_notes']
        if notes:d.text((left,687),'REVIEW: '+'; '.join(notes)[:180],font=old.font(16,True),fill='#955000')
        d.text((left,719),'* CoP-derived candidate cuts. SSSW: 5 intervals; SLT: 1; SSLW: 5. Additional observed intervals are labelled E1, E2 ... before GT.',font=old.font(16),fill='#66502d')
        d.text((left,746),'Saved stance candidates run CoP valley to peak. Preprocessing and display normalization do not change source measurements. Events are not ground truth.',font=old.font(16),fill='#555555')
    else:
        d.text((left,112),'UNRESOLVED: '+seg['reason'],font=old.font(21,True),fill='#955000')
        d.text((left,719),'Complete recording shown. Unsupported phase boundaries are not invented to reach a requested step count.',font=old.font(17),fill='#66502d')
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.png',delete=False) as f:
        pending=Path(f.name);im.save(f,format='PNG')
    try:pending.replace(path)
    finally:pending.unlink(missing_ok=True)


def sha256(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def process(subject,backup_root):
    """Replace only subject graphs and their derived event/stance summaries.

    Source CSVs (including their previous phase columns) remain byte-identical.
    Previous derived stances are retained in the private recoverable backup.
    """
    if subject not in SUBJECTS:raise ValueError(subject)
    source,root=old.SUBJECTS[subject]
    root=root.resolve();workspace=Path(__file__).resolve().parent
    assert root.is_relative_to(workspace/'tempgraphs') and root.name==subject.lower()
    assert not root.is_symlink() and not root.is_junction()
    backup=Path(backup_root).resolve()/subject
    backup.mkdir(parents=True,exist_ok=False)
    for name in ('phase_cuts.csv','phase_intervals.csv','stance_segments.csv','README_cop_segmentation.md','cop_verification.json'):
        if (root/name).exists():shutil.copy2(root/name,backup/name)
    # Prepare new stance exports separately before replacing the old derived set.
    staged=backup/'new_stances';staged.mkdir()
    cuts=[];intervals=[];stances=[];audit=[]
    for lp in sorted(source.rglob('trial_*_l_segmented.csv')):
        rp=lp.with_name(lp.name.replace('_l_','_r_'))
        t,l,fl=old.parse_csv(lp);tr,r,fr=old.parse_csv(rp)
        assert np.array_equal(t,tr) and np.allclose(np.diff(t),.01,atol=1e-9)
        record=lp.parent.name;trial=int(lp.name.split('_')[1])
        result=segment(subject,record,trial,l,r,fl,fr)
        for side,values,flags,path in [('L',l,fl,lp),('R',r,fr,rp)]:
            seg=result[side];lay=seg['layout'];key=dict(record=record,trial=trial,side=side)
            row={**key,'status':'cop_valley_cut' if lay else 'unresolved',
                 'cop_stances_found':len(seg['stances']),'extra_steps':len(seg['extra_steps']),
                 'review_required':int(not lay or bool(seg['review_notes'])),
                 'review_notes':'; '.join(seg['review_notes']) if lay else seg['reason'],
                 'source_window_method':result['source_window_method'],
                 'preprocessing':'symmetric_3_sample_mean_and_isolated_1_2_sample_spike_repair',
                 'source_labels_updated':0}
            input_hash=sha256(path)
            with path.open(encoding='utf-8',newline='') as f:raw=list(csv.DictReader(f))
            if lay:
                bounds,steps,end_kind=lay;gt=seg['GT_start_sample']
                phases=list(zip(old.PHASES[:-1],bounds[:-2],bounds[1:-1]))
                if gt>bounds[5]:phases.append(('EXTRA',bounds[5],gt))
                phases.append(('GT',gt,len(t)))
                for name,a,b in phases:
                    row[name+'_start_sample']=a
                    row[name+'_start_s']=round(a/100,2)
                    intervals.append({**key,'phase':name,'start_sample':a,'end_sample_exclusive':b,
                                      'start_seconds':round(a/100,2),'end_seconds_exclusive':round(b/100,2)})
                row['SSLW_end_sample']=bounds[5]
                row['protocol_end_sample']=bounds[5]
                row['GT_start_kind']='after_observed_extra_cycles' if seg['extra_steps'] else end_kind
                row['cuts_on_estimated_cop']=' '.join(seg['estimated_cuts'])
                all_steps=steps+seg['extra_steps']
                for phase,num,v,p,kind,nxt in all_steps:
                    name=f'E{num}' if phase=='EXTRA' else old.step_name(phase,num)
                    filename=f'trial_{trial:02d}_{side.lower()}_{phase}_{name}.csv'
                    target=staged/record/filename;target.parent.mkdir(parents=True,exist_ok=True)
                    chunk=[]
                    for i in range(v,p+1):
                        chunk.append({'time_seconds':raw[i]['time_seconds'],'sample':i,'time_from_valley_seconds':f'{(i-v)/100:.2f}',
                                      **{k:raw[i][k] for k in old.VALUE_COLUMNS},
                                      'FMG_estimated':int(flags[i,:8].any()),'CoP_estimated':int(flags[i,8]),'vGRF_estimated':int(flags[i,9]),
                                      'spike_repaired_for_detection':int(seg['repaired'][i].any())})
                    old.write_dicts(target,chunk)
                    f=flags[v:p+1]
                    stances.append({**key,'phase':phase,'step':name,'cop_valley_sample':v,'cop_peak_sample':p,'next_valley_sample':nxt,
                                    'cop_valley_s':v/100,'cop_peak_s':p/100,'next_valley_s':nxt/100,'samples':p-v+1,
                                    'valley_kind':kind,'cop_estimated_samples':int(f[:,8].sum()),
                                    'fmg_estimated_samples':int(f[:,:8].any(1).sum()),'vgrf_estimated_samples':int(f[:,9].sum()),
                                    'fully_measured':int(not f.any()),'values':'original_export_not_smoothed',
                                    'file':f'stance_segments/{record}/{filename}'})
            else:row['unresolved_reason']=seg['reason']
            cuts.append(row)
            graph=root/record/f'trial_{trial:02d}_{side.lower()}_complete.png'
            if graph.exists():
                saved=backup/'previous_graphs'/record/graph.name
                saved.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(graph,saved)
            draw(graph,f'{subject} | {record} | Recording interval {trial:02d} | {side}',t,seg,flags)
            assert sha256(path)==input_hash
            audit.append({**key,'source_csv':path.relative_to(workspace).as_posix(),'source_csv_sha256':input_hash,
                          'graph':graph.relative_to(root).as_posix(),'graph_sha256':sha256(graph),
                          'repaired_samples_by_channel':{k:int(seg['repaired'][:,j].sum()) for j,k in enumerate(old.VALUE_COLUMNS)},
                          'previously_estimated_samples_by_channel':{k:int(flags[:,j].sum()) for j,k in enumerate(old.VALUE_COLUMNS)},
                          'source_unchanged':True,'sample_times_unchanged':True,
                          'status':row['status'],'review_required':row['review_required'],
                          'extra_steps':row['extra_steps'],'reason':seg['reason'],'review_notes':seg['review_notes']})
    old_stances=root/'stance_segments'
    if old_stances.exists():
        assert old_stances.resolve().parent==root and not old_stances.is_junction() and not old_stances.is_symlink()
        shutil.move(str(old_stances),str(backup/'previous_stance_segments'))
    shutil.move(str(staged),str(old_stances))
    for name,rows in [('phase_cuts.csv',cuts),('phase_intervals.csv',intervals),('stance_segments.csv',stances)]:old.write_dicts(root/name,rows)
    (root/'preprocessing_audit.json').write_text(json.dumps({'method':'CoP-derived noise-aware event candidates; display-only amplitude normalization',
        'filter':'3-sample centred moving mean; isolated 1-2 sample Hampel-style repairs using 9-sample median and 8 scaled MAD / 12% span threshold',
        'preserved':'Raw exports, timestamps, flags and prior source phase columns are byte-identical. No synthetic cycles or forced flat floors.',
        'graphs':audit},indent=2),encoding='utf-8')
    resolved=sum(r['status']=='cop_valley_cut' for r in cuts)
    summary={'subject':subject,'graphs':len(cuts),'provisional_5_1_5_layouts':resolved,'unresolved_graphs':len(cuts)-resolved,
             'layouts_additionally_flagged_for_review':sum(r['status']=='cop_valley_cut' and r['review_required'] for r in cuts),
             'graphs_with_extra_steps':sum(r['extra_steps']>0 for r in cuts),'saved_stance_candidates':len(stances),
             'source_csvs_byte_identical':len(cuts),'visual_review_passes_completed':0,
             'limitation':'Signal-derived event candidates, not independently validated foot contacts or step-length labels.'}
    (root/'cop_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (root/'README_cop_segmentation.md').write_text(
        f'# {subject}: noise-aware CoP graph refinement\n\n'
        f'{len(cuts)} complete limb graphs; {resolved} provisional layouts; {len(cuts)-resolved} unresolved. '
        f'{summary["graphs_with_extra_steps"]} graphs contain supported additional closing intervals.\n\n'
        'Each trial is evaluated independently. The original CoP valley-to-peak convention is retained, '
        'but narrow minima use a 3-sample detection smoother rather than an 11-sample median. '
        'The dominant CoP cadence and recurring loading changes restrict candidates to the main walking train. '
        'A short initial weight-shift interval is excluded when it is under 0.7 of this trial\'s typical stride. '
        'GI uses a local departure from quiet standing before that train, not a generic time.\n\n'
        'SSSW contains the first five candidate intervals, SLT the sixth, and SSLW the next five. '
        'A candidate step runs valley-to-valley; its saved stance is valley-to-peak. '
        'Supported extra intervals are shown after SSLW as Extra step(s), E1, E2 ... . '
        'GT starts after these complete. Small stopping adjustments are not automatically counted as extra steps. '
        'The final contact fallback requires an actual loading change near the per-trial cadence prediction. '
        'If evidence is insufficient or trial triggers are unpaired, the full graph remains visible with unresolved cuts.\n\n'
        'Display preprocessing: isolated 1-2 sample spikes are repaired only when surrounding samples agree; '
        'all traces receive a centred 3-sample moving average. True unloaded lows are preserved. '
        'Amplitude gains and offsets keep FMG below the CoP/vGRF display bands. '
        'No reference-shaped waveform, artificial valley floor, time stretching or invented step is used. '
        'Pale samples denote existing estimates or spike repairs.\n\n'
        'All raw and segmented_data CSV bytes, including their previous label columns, are unchanged. '
        'Those old provisional_phase values may no longer match the graphs: current phase_cuts.csv and '
        'phase_intervals.csv are authoritative for this refinement. Saved stance CSVs contain original, '
        'not smoothed, measurements and existing estimate flags. preprocessing_audit.json records input/output hashes '
        'and repair counts for every graph. Prior derived stances and summaries are backed up outside the repository.\n\n'
        'These are operational signal heuristics, not verified heel strikes, push-off or measured step lengths. '
        'Matching 5-1-5 counts do not establish physiological accuracy. Review warnings and missing measurements '
        'must be resolved against independent event evidence before model training. Other subjects are unchanged.\n',encoding='utf-8')
    print(json.dumps(summary))
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subjects',nargs='+',choices=SUBJECTS)
    parser.add_argument('--backup-root',type=Path,required=True)
    args=parser.parse_args()
    for subject in args.subjects:process(subject,args.backup_root)
