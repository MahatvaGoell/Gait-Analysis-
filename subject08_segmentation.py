"""Provisional, trial-specific sequence fits. NOT ground-truth study labels.

The user-selected short-to-long order is an assumption, not a measured step
length. Pressure landmarks are recurring load extrema, not heel strikes.
Only timestamp-aligned original signals are plotted. No waveform warping.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from generate_qs_plots import DATASET_ROOT, load_stream, split_source_files, unit_trial_starts
from subject08_review import OUT, DT, grid_values, render, runs, write_rows


def smooth(x, width=9):
    # Detection only. A missing sample invalidates its entire local window.
    p = np.pad(x, (width//2, width//2), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(p, width)
    return np.mean(windows, axis=1)


def load_peaks(v, prominence_fraction=.20):
    """Find a sustained sequence of prominent load peaks, not foot contacts."""
    v = smooth(v)
    finite = np.isfinite(v)
    if finite.sum() < 200:
        return [], []
    span = np.nanpercentile(v, 95) - np.nanpercentile(v, 5)
    noise = np.nanmedian(np.abs(np.diff(v[:75])))
    minimum = max(span*prominence_fraction, 12*noise, 30.)
    candidates = []
    for k in range(60, len(v)-60):
        if not np.all(finite[k-60:k+61]):
            continue
        if v[k] > v[k-1] and v[k] >= v[k+1]:
            prominence = v[k] - max(np.min(v[k-60:k]), np.min(v[k+1:k+61]))
            if prominence >= minimum:
                candidates.append((k, float(prominence)))
    selected = []
    for k, prominence in sorted(candidates, key=lambda q: -q[1]):
        if all(abs(k-a) >= 60 for a, _ in selected):
            selected.append((k, prominence))
    selected.sort()
    chains = []
    for k, p in selected:
        if not chains or k-chains[-1][-1][0] > 230:
            chains.append([])
        chains[-1].append((k,p))
    if not chains:
        return [], selected
    chain = max(chains, key=lambda c: (len(c), sum(p for _,p in c)))
    return [k for k,_ in chain], selected


def local_activity(z):
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(z, ((15,15),(0,0)), mode='edge'), 31, axis=0)
    span = np.percentile(windows,90,axis=2)-np.percentile(windows,10,axis=2)
    base = np.nanmedian(span[:75],axis=0)
    mad = 1.4826*np.nanmedian(np.abs(span[:75]-base),axis=0)
    threshold = np.maximum(base+6*np.maximum(mad,.15), np.maximum(np.nanpercentile(span,90,axis=0)*.20,1.))
    return np.percentile(span/threshold,75,axis=1)


def movement_onset(z, first_peak, typical_stride):
    scores = np.column_stack([local_activity(z[:,:16]), local_activity(z[:,16:18]), local_activity(z[:,18:20])])
    good = np.any(np.isfinite(scores),axis=1)
    score = np.max(np.where(np.isfinite(scores),scores,-np.inf),axis=1)
    active = good & (score>1)
    for a,b in runs(~active):
        if a>0 and b<len(active) and b-a<=35 and np.all(good[a:b]):
            active[a:b]=True
    possible = [(a,b) for a,b in runs(active) if b-a>=75 and a<=first_peak and b>=first_peak-typical_stride]
    if not possible:
        return None
    onset = max(0, possible[0][0]-15)
    return onset if onset>=50 else None


def cycle_features(z, peaks):
    features=[]
    for a,b in zip(peaks[:-1],peaks[1:]):
        block=z[a:b]
        if np.any(np.mean(np.isfinite(block),axis=0)<.90):
            features.append(None)
            continue
        med=np.nanmedian(block,axis=0)
        span=np.nanpercentile(block,90,axis=0)-np.nanpercentile(block,10,axis=0)
        features.append(np.r_[med[:16],span[:16],med[16:],span[16:],(b-a)*DT])
    return features


def change_window(features, keep=None):
    """Two constant regimes separated by one unlabelled transition stride.

    Minimum two observed strides on each side; no preferred split time.
    Improvement and alternative split uncertainty are exported, not confidence
    probabilities. Feature groups are balanced to avoid 16 FMG channels
    overwhelming two insoles.
    """
    if len(features)<6 or any(x is None for x in features):
        return None
    f=np.array(features)
    center=np.median(f,axis=0)
    scale=np.maximum(np.percentile(f,90,axis=0)-np.percentile(f,10,axis=0),
                     np.r_[np.full(32,2.),np.array([.2,30,.2,30,.2,30,.2,30]),.05])
    x=(f-center)/scale
    weights=np.r_[np.full(32,1/96), np.full(8,1/24), 1/3]
    if keep is not None:
        weights=weights*keep
        weights/=weights.sum()
    def cost(v):
        return float(np.sum((v-np.mean(v,axis=0))**2*weights))
    total=cost(x)
    trials=[]
    for k in range(2,len(x)-2):
        # Residual from the one transitional stride gets half weight rather
        # than being discarded, preventing an arbitrary outlier from winning.
        a,b=x[:k],x[k+1:]
        mid=(a.mean(axis=0)+b.mean(axis=0))/2
        residual=cost(a)+cost(b)+.5*float(np.sum((x[k]-mid)**2*weights))
        trials.append((k,residual))
    k,best=min(trials,key=lambda q:q[1])
    improvement=1-best/max(total,1e-9)
    alternatives=[j for j,c in trials if c<=best+max(.10*total,.02)]
    return dict(index=k,improvement=improvement,alternatives=alternatives)


def segment_trial(grid,z):
    peaks_l,_=load_peaks(z[:,17]); peaks_r,_=load_peaks(z[:,19])
    info=dict(status='unresolved',sequence_assumption='QS-GI-SSSW-SLT-SSLW-GT; short-to-long requested, not verified',
              load_peaks_left=len(peaks_l),load_peaks_right=len(peaks_r),onset_seconds='',gi_end_seconds='',
              slt_start_seconds='',slt_end_seconds='',gt_start_seconds='',transition_support='unresolved',
              transition_improvement='',alternative_transition_start_seconds='',alternative_transition_end_seconds='',
              flags='All events provisional; pressure extrema are not heel-strike labels')
    landmarks=[dict(side=s,time_seconds=round(k*DT,2),event='recurring_load_peak_candidate')
               for s,pk in [('L',peaks_l),('R',peaks_r)] for k in pk]
    if min(len(peaks_l),len(peaks_r))<5 or max(len(peaks_l),len(peaks_r))<8:
        info['flags']+='; insufficient bilateral repeated cycles for a six-phase fit'
        return [],info,landmarks
    reference='L' if len(peaks_l)>=len(peaks_r) else 'R'
    peaks=peaks_l if reference=='L' else peaks_r
    typical=float(np.median(np.diff(peaks)))
    first=min(peaks_l[0],peaks_r[0])
    onset=movement_onset(z,first,typical)
    if onset is None:
        info['flags']+='; sustained onset with initial standing not resolved'
        return [],info,landmarks
    # Operational GI end: first prominent loading peak of a repetitive bout.
    # Operational GT start: beginning of the final bilateral peak pair.
    gi_end=first
    gt_start=min(peaks_l[-1],peaks_r[-1])
    if grid[-1]-gt_start*DT<.8:
        info['flags']+='; final halt not adequately recorded'
        return [],info,landmarks
    usable=[p for p in peaks if gi_end<=p<=gt_start]
    features=cycle_features(z,usable)
    result=change_window(features)
    if result is None:
        info['flags']+='; insufficient complete strides or missing data for transition estimate'
        return [],info,landmarks
    k=result['index']; slt_start,slt_end=usable[k],usable[k+1]
    assert onset<gi_end<slt_start<slt_end<gt_start
    # Stability check: remove FMG, remove insoles, and remove stride duration.
    changes=[]
    for a,b in ((0,32),(32,40),(40,41)):
        keep=np.ones(41); keep[a:b]=0
        changes.append(change_window(features,keep)['index'])
    agreement=max(abs(v-k) for v in changes)
    support='moderate pattern evidence' if result['improvement']>=.30 and agreement<=1 and len(result['alternatives'])<=2 else 'weak/ambiguous pattern evidence'
    boundaries=[0,onset,gi_end,slt_start,slt_end,gt_start,len(grid)]
    phases=[dict(phase=label,start_index=int(a),end_index=int(b),start_seconds=round(a*DT,2),
                 end_seconds=round(float(grid[-1]) if b==len(grid) else b*DT,2),status='provisional')
            for label,a,b in zip(('QS','GI','SSSW','SLT','SSLW','GT'),boundaries[:-1],boundaries[1:])]
    alt=result['alternatives']+changes
    info.update(status='provisional six-phase fit',onset_seconds=round(onset*DT,2),gi_end_seconds=round(gi_end*DT,2),
                slt_start_seconds=round(slt_start*DT,2),slt_end_seconds=round(slt_end*DT,2),
                gt_start_seconds=round(gt_start*DT,2),transition_support=support,
                transition_improvement=round(result['improvement'],4),
                alternative_transition_start_seconds=round(usable[min(alt)]*DT,2),
                alternative_transition_end_seconds=round(usable[max(alt)+1]*DT,2),reference_limb=reference)
    if support.startswith('weak'): info['flags']+='; SLT timing ambiguous'
    if abs(peaks_l[-1]-peaks_r[-1])>typical:
        info['flags']+='; terminal bilateral peak mismatch'
    return phases,info,landmarks


def iter_trials(record_filter=None,trial_filter=None):
    fm,ins=split_source_files(DATASET_ROOT/'Sub08_H')
    for record,fp in fm.items():
        if record_filter and record_filter!=record: continue
        f=load_stream(fp,18,'FMG')
        streams={s:load_stream(ins[(record,s)],4,'insole',s) for s in ('L','R')}
        for a,b,trial in unit_trial_starts(f):
            if trial_filter and trial_filter!=trial: continue
            if b is None: continue
            ft=f.values[a:b+1,0]-f.values[a,0]
            raw={}; quality={}
            for side,i in streams.items():
                c,e=i.rising_edges[2*(trial-1):2*trial]
                it=i.values[c:e+1,0]-i.values[c,0]
                raw[side]=(it,i.values[c:e+1,2:4])
                quality[side]=dict(full_fmg_seconds=float(ft[-1]),full_insole_seconds=float(it[-1]),
                    trigger_end_error_seconds=float(it[-1]-ft[-1]),fmg_gaps=int(np.sum(np.diff(ft)>.10)),
                    insole_gaps=int(np.sum(np.diff(it)>.10)),nonmonotonic_or_duplicate_timestamps=int(np.sum(np.diff(it)<=0)+np.sum(np.diff(ft)<=0)))
            duration=max(ft[-1],*(t[-1] for t,_ in raw.values()))
            grid=np.arange(0,duration+.005,DT)
            z=np.column_stack([grid_values(ft,f.values[a:b+1,2:18],grid),
                               *(grid_values(*raw[s],grid) for s in ('L','R'))])
            yield record,trial,grid,z,quality


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',choices=('sir_1','sir_21'))
    parser.add_argument('--trial',type=int)
    parser.add_argument('--side',choices=('L','R'),help='Render one limb; boundaries still use both limbs.')
    parser.add_argument('--overwrite',action='store_true',help='Compatibility option: selected plots are regenerated, with a first-run backup.')
    parser.add_argument('--diagnose',action='store_true',help='Read-only estimate, without writing outputs.')
    args=parser.parse_args()
    records=[]; segments=[]; events=[]; plots=[]; transforms=[]
    raw_paths=sorted(p for p in (DATASET_ROOT/'Sub08_H').iterdir() if p.is_file())
    before={str(p.relative_to(DATASET_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_paths}
    if not args.diagnose:
        backup=OUT/'_before_sequence_segmentation'
        backup.mkdir(exist_ok=True)
        for name in ('complete_trial_manifest.csv','display_transforms.csv','synchronization_report.csv'):
            source=OUT/name; target=backup/name
            if source.exists() and not target.exists(): shutil.copy2(source,target)
    for record,trial,grid,z,quality in iter_trials(args.record,args.trial):
        phases,info,landmarks=segment_trial(grid,z)
        if any(q['fmg_gaps'] or q['insole_gaps'] for q in quality.values()):
            info['flags']+='; observed timestamp gaps retained'
        records.append(dict(record=record,trial=trial,**info))
        segments.extend(dict(record=record,trial=trial,**p,sequence_assumption=info['sequence_assumption']) for p in phases)
        events.extend(dict(record=record,trial=trial,**p) for p in landmarks)
        if args.diagnose: continue
        labels=np.full(len(grid),'UNRESOLVED',dtype='<U10')
        for p in phases: labels[p['start_index']:p['end_index']]=p['phase']
        for side,columns in [('L',list(range(8))+[16,17]),('R',list(range(8,16))+[18,19])]:
            if args.side and side!=args.side: continue
            local=z[:,columns]
            filename=f'Sub08_H/{record}/trial_{trial:02d}_{side.lower()}_complete.png'
            destination=OUT/filename
            previous=backup/filename
            if destination.exists() and not previous.exists():
                previous.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(destination,previous)
            q=quality[side]
            message=(f'Transition: {info["transition_support"]}. Alternative fits: '
                     f'{info["alternative_transition_start_seconds"]}-{info["alternative_transition_end_seconds"]} s; not a confidence interval.') if phases else info['flags']
            quality_text=f'Trigger-end difference {q["trigger_end_error_seconds"]:+.3f} s | gaps >0.10 s: FMG {q["fmg_gaps"]}, insole {q["insole_gaps"]} | repeated/reversed timestamps {q["nonmonotonic_or_duplicate_timestamps"]}'
            tr=render(destination,f'Sub08_H | {record} | Trigger-pair trial {trial:02d} | {side}',grid,local,[],info,quality_text,phases,message)
            transforms.extend(dict(record=record,trial=trial,side=side,**t) for t in tr)
            sample_path=OUT/'segmented_data'/f'Sub08_H/{record}/trial_{trial:02d}_{side.lower()}_segmented.csv'
            sample_path.parent.mkdir(parents=True,exist_ok=True)
            with sample_path.open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f)
                writer.writerow(['time_seconds',*[f'FMG_channel_{k+1}' for k in range(8)],'CoP','vGRF','provisional_phase','label_status','signal_missing'])
                for t,x,label in zip(grid,local,labels):
                    writer.writerow([f'{t:.2f}',*[f'{v:.17g}' if np.isfinite(v) else '' for v in x],label,
                                     'unresolved' if label=='UNRESOLVED' else 'provisional_assumed_short_to_long',int(not np.all(np.isfinite(x)))])
            plots.append(dict(record=record,trial=trial,side=side,png=filename,segmented_csv=str(sample_path.relative_to(OUT)),**q,**info))
    after={str(p.relative_to(DATASET_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_paths}
    assert before==after, 'Raw source file changed during generation'
    if args.diagnose:
        print(json.dumps(records,indent=2)); return
    if not args.record and not args.trial and not args.side:
        write_rows(OUT/'phase_boundaries_provisional.csv',records)
        write_rows(OUT/'phase_intervals_provisional.csv',segments)
        write_rows(OUT/'load_landmarks_provisional.csv',events)
        write_rows(OUT/'complete_trial_manifest.csv',plots)
        write_rows(OUT/'display_transforms.csv',transforms)
        review=OUT/'segmentation_review_sheets'; review.mkdir(exist_ok=True)
        for page in range(math.ceil(len(plots)/4)):
            sheet=Image.new('RGB',(1600,640),'white')
            for j,row in enumerate(plots[page*4:page*4+4]):
                with Image.open(OUT/row['png']) as im:
                    im.thumbnail((800,320)); sheet.paste(im,((j%2)*800,(j//2)*320))
            sheet.save(review/f'page_{page+1:02d}.png')
        with (OUT/'segmentation_provenance.json').open('w',encoding='utf-8') as f:
            json.dump(dict(method_version='1.0-provisional',raw_sha256=before,
                           script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           assumption='User-requested short-to-long sequence; not verified by step-length measurements',
                           validation='Signal/layout consistency only; no ground-truth event validation',
                           completed_trials=len(records),completed_plots=len(plots)),f,indent=2)
    print(json.dumps(dict(graphs=len(plots),provisional_six_phase_trials=sum(bool(r['slt_start_seconds']) for r in records),
                          unresolved_trials=[f'{r["record"]}/{r["trial"]:02d}' for r in records if r['status']=='unresolved'],raw_files_unchanged=before==after),indent=2))


if __name__=='__main__': main()
