"""Subjects 01-04 only: audited clocks, complete views, provisional labels.

The A/H datasets are independent. Existing Subjects 05-08 and raw files are
protected. Missing measurements are explicit estimates, not recovered data.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
from generate_additional_subjects import ROOT, DATASET_ROOT, catalog
from subject08_review import DT, grid_values, write_rows
from complete_reconstruction import complete_trial, complete_segments
from subject08_segmentation import load_peaks, local_activity, cycle_features, change_window
from subject08_review import runs
from subjects01_to04_renderer import render

SUBJECTS=('Sub01_A','Sub01_H','Sub02_A','Sub02_H','Sub03_H','Sub04_H')
OUTPUTS=[ROOT/'tempgraphs'/s.lower() for s in SUBJECTS]


def protected_state():
    paths=[p for p in (ROOT/'tempgraphs').rglob('*') if p.is_file() and not any(o in p.parents for o in OUTPUTS)]
    paths += [p for p in DATASET_ROOT.rglob('*') if p.is_file()]
    paths += [p for p in ROOT.glob('*.py') if p.name not in ('generate_subjects01_to04.py','subjects01_to04_renderer.py')]
    return {str(p.relative_to(ROOT)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
             size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(paths)}


def align_times(reference,other):
    """Robust trigger-offset consensus; tolerate missing and noisy pulses."""
    if len(other)<2:raise ValueError('At least two trigger anchors are required')
    deltas=(other[:,None]-reference[None,:]).ravel()
    bins,counts=np.unique(np.round(deltas/.05).astype(int),return_counts=True)
    candidates=[float(np.median(deltas[np.round(deltas/.05).astype(int)==b]))
                for b in bins[np.argsort(counts)[-30:]]]
    results=[]
    for initial in candidates:
        offset=initial
        for _ in range(3):
            nearest=np.argmin(abs((reference+offset)[:,None]-other[None,:]),axis=1)
            good=abs(other[nearest]-reference-offset)<=.25
            if not good.any():break
            offset=float(np.median(other[nearest[good]]-reference[good]))
        ids=np.flatnonzero(good)
        if len(set(nearest[ids]))!=len(ids):continue
        residual=abs(other[nearest[ids]]-reference[ids]-offset)
        results.append((len(ids),-float(np.median(residual)) if len(ids) else -1,offset,ids,nearest,residual))
    if not results:raise ValueError('No unambiguous trigger correspondence')
    count,_,offset,ids,nearest,residual=max(results,key=lambda v:(v[0],v[1]))
    if count<max(2,.70*min(len(reference),len(other))):
        raise ValueError(f'Insufficient trigger consensus: {count}/{min(len(reference),len(other))}')
    return offset,dict(matched_reference_indices=ids.tolist(),matched_source_indices=nearest[ids].tolist(),
                       matched_edges=count,max_residual_seconds=float(max(residual)),
                       median_residual_seconds=float(np.median(residual)))


def normalize_record(subject,record,streams):
    audits=[];mapped={};trigger_times={}
    # Subject 04 left trigger lines chatter; they are NEVER a master timeline.
    choices=('F','R') if subject=='Sub04_H' else ('F','L','R')
    master=max(choices,key=lambda s:len(streams[s]['edges']) if streams[s] else -1)
    # Subject 04 R clock resets, so FMG is the stable clock reference.
    if subject=='Sub04_H':master='F'
    base=streams[master];reference=base['values'][base['edges'],0]
    for side,st in streams.items():
        if st is None:
            mapped[side]=None;trigger_times[side]=np.array([]);continue
        v=st['values'];cuts=np.r_[0,np.flatnonzero(np.diff(v[:,0])<-5)+1,len(v)]
        pieces=[];events=[]
        for epoch,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
            ids=st['edges'][(st['edges']>=a)&(st['edges']<b)]
            times=v[ids,0]
            if side==master:
                if len(cuts)>2:raise ValueError('Master clock has a reset')
                offset=0.;detail=dict(matched_edges=len(times),max_residual_seconds=0.,median_residual_seconds=0.)
            elif len(times)<2:
                audits.append(dict(record=record,side=side,epoch=epoch,status='unmapped epoch without trigger anchors',
                    source_rows=int(b-a),source_start=float(v[a,0]),source_end=float(v[b-1,0])))
                continue
            else:offset,detail=align_times(reference,times)
            block=v[a:b].copy();block[:,0]-=offset;pieces.append(block);events.extend(times-offset)
            audits.append(dict(record=record,side=side,epoch=epoch,status='aligned',source_rows=int(b-a),
                source_start=float(v[a,0]),source_end=float(v[b-1,0]),clock_offset_to_master=offset,
                source_trigger_count=len(times),master=master,**detail))
        mapped[side]=np.concatenate(pieces) if pieces else None
        trigger_times[side]=np.array(events)
    timeline=list(reference);repairs=[]
    if subject=='Sub04_H':
        # Recover a missing FMG pulse only if BOTH insoles corroborate it.
        for t in trigger_times['R']:
            if min(abs(reference-t))>.25 and len(trigger_times['L']) and min(abs(trigger_times['L']-t))<=.10:
                timeline.append(float(t));repairs.append(float(t))
    timeline=np.array(sorted(timeline))
    if np.any(np.diff(timeline)<=.3):raise ValueError('Ambiguous adjacent master triggers')
    return mapped,trigger_times,timeline,master,audits,repairs


def paired_windows(times):
    """Retain unmatched-start context rather than shifting every later pair."""
    i=0;number=1
    while i<len(times):
        if i+1==len(times):raise ValueError('Trailing unpaired trigger requires review')
        start,end=map(float,times[i:i+2])
        if end-start>60:
            # The next pulse begins the next plausible pair. Do not consume it
            # as an end pulse or manufacture a measured end for this window.
            yield number,start,end-DT,'unpaired start: context until just before next start'
            i+=1
        else:
            if end-start<5:raise ValueError('Implausibly short trigger pair')
            yield number,start,end,'paired trigger interval'
            i+=2
        number+=1


def load_trials(subject):
    trials=[];sync=[];window_audit=[]
    for record,streams in catalog(subject).items():
        mapped,events,times,master,audits,repairs=normalize_record(subject,record,streams)
        sync.extend(audits)
        for number,start,end,boundary in paired_windows(times):
            grid=np.arange(0,end-start+.005,DT);arrays=[];quality={};notes=[]
            if boundary.startswith('unpaired'):notes.append(boundary+'; true trial end and protocol membership unverified')
            for side,width in [('F',16),('L',2),('R',2)]:
                st=streams[side];v=mapped[side]
                block=None if v is None else v[(v[:,0]>=start-DT/2)&(v[:,0]<=end+DT/2)]
                if block is None or len(block)<2:
                    arr=np.full((len(grid),width),np.nan)
                else:arr=grid_values(block[:,0]-start,block[:,2:],grid)
                arrays.append(arr)
                anchors={label:bool(len(events[side]) and min(abs(events[side]-t))<=.25)
                         for label,t in [('start',start),('end',end)]}
                quality[side]=dict(coverage_fraction=float(np.mean(np.isfinite(arr))),
                    source=st['path'] if st else '',invalid_rows=st['invalid_rows'] if st else 0,
                    excluded_timestamp_outliers=st['excluded_timestamp_outliers'] if st else 0,
                    boundary_trigger_matches=anchors,aligned_master_start=start,aligned_master_end=end)
                if not all(anchors.values()):notes.append(f'{side}: one or more boundaries inferred from aligned master clock')
            z=np.column_stack(arrays)
            window_audit.append(dict(record=record,trial=number,start_master_seconds=start,end_master_seconds=end,
                duration_seconds=float(grid[-1]),master=master,boundary_method=boundary,
                master_trigger_recovered_from_both_insoles=any(abs(t-start)<.25 or abs(t-end)<.25 for t in repairs)))
            trials.append((record,number,grid,z,quality,boundary,notes))
    return trials,sync,window_audit


def clean_limb_sequence(grid,z):
    """Fallback for pressure noise preceding true FMG-confirmed movement.

    Evaluate each limb independently rather than choosing whichever has more
    pressure peaks. A candidate requires sustained FMG activity near its first
    loading peak and sufficient cycles for the two-regime feature fit.
    """
    score=local_activity(z[:,:16]);active=np.isfinite(score)&(score>1)
    for a,b in runs(~active):
        if a>0 and b<len(active) and b-a<=35:active[a:b]=True
    candidates=[]
    for side,col in [('L',17),('R',19)]:
        peaks,_=load_peaks(z[:,col])
        if len(peaks)<8:continue
        period=float(np.median(np.diff(peaks)));first,last=peaks[0],peaks[-1]
        # The 31-sample FMG activity window has 15-sample temporal support.
        # Permit its threshold crossing just after the pressure extremum.
        bouts=[(a,b) for a,b in runs(active) if b-a>=75 and a-15<=first and b>=first-period]
        if not bouts or grid[-1]-last*DT<.8:continue
        onset=max(0,bouts[0][0]-15)
        if onset<50 or onset>=first:continue
        result=change_window(cycle_features(z,peaks))
        if result is None:continue
        k=result['index'];a,b=peaks[k:k+2]
        if not onset<first<a<b<last:continue
        variability=float(np.median(abs(np.diff(peaks)-period))/period)
        candidates.append((variability,side,peaks,onset,result))
    if not candidates:return None
    _,side,peaks,onset,result=min(candidates,key=lambda c:c[0])
    k=result['index'];a,b=peaks[k:k+2];first,last=peaks[0],peaks[-1]
    bounds=[0,onset,first,a,b,last,len(grid)]
    phases=[dict(phase=label,start_index=int(s),end_index=int(e),start_seconds=round(s*DT,2),
        end_seconds=round(float(grid[-1]) if e==len(grid) else e*DT,2),status='model_assisted_provisional')
        for label,s,e in zip(('QS','GI','SSSW','SLT','SSLW','GT'),bounds[:-1],bounds[1:])]
    info=dict(status='model-assisted provisional six-phase fit',phase_count=6,complete_six_phase=True,
        onset_seconds=round(onset*DT,2),gi_end_seconds=round(first*DT,2),slt_start_seconds=round(a*DT,2),
        slt_end_seconds=round(b*DT,2),gt_start_seconds=round(last*DT,2),reference_limb=side,
        transition_support='FMG-confirmed onset; cleaner-limb fit; unverified',
        alternative_transition_start_seconds=round(peaks[min(result['alternatives'])]*DT,2),
        alternative_transition_end_seconds=round(peaks[max(result['alternatives'])+1]*DT,2),
        flags='Provisional cleaner-limb fallback: first pressure peak requires independently sustained FMG activity; short-to-long order assumed; extrema are not verified heel strikes')
    return phases,info


def analyze(item,trials):
    z,mask,gaps=complete_trial(item,trials)
    phases,info,events=complete_segments(item[2],z)
    if not info['complete_six_phase'] and not item[5].startswith('unpaired'):
        corrected=clean_limb_sequence(item[2],z)
        if corrected is not None:
            phases,update=corrected;info.update(update)
    if info['complete_six_phase'] and 'terminal bilateral peak mismatch' in info['flags']:
        # A shorter pressure chain must not end walking while the other limb
        # still has cadence-consistent cycles with strong independent FMG activity.
        chains=[load_peaks(z[:,c])[0] for c in (17,19)]
        later=max(chains,key=lambda p:p[-1])
        period=float(np.median(np.diff(later)));last=later[-1]
        activity=local_activity(z[:,:16]);half=max(1,int(period/2))
        strong=float(np.percentile(activity[max(0,last-half):min(len(z),last+half)],90))
        if (.8 <= (later[-1]-later[-2])/period <= 1.25 and strong>=2
                and item[2][-1]-last*DT>=.8 and last*DT>float(info['gt_start_seconds'])+period*DT):
            phases[-2].update(end_index=int(last),end_seconds=round(last*DT,2))
            phases[-1].update(start_index=int(last),start_seconds=round(last*DT,2))
            info['gt_start_seconds']=round(last*DT,2)
            info['flags']+='; GT extended to later cadence-consistent pressure peak supported by strong FMG activity'
    if item[5].startswith('unpaired') and item[2][-1]>120:
        phases=[dict(phase='UNKNOWN',start_index=0,end_index=len(z),start_seconds=0.,end_seconds=float(item[2][-1]),status='unpaired_context')]
        info.update(status='partial phase annotation',complete_six_phase=False,phase_count=1,
                    onset_seconds='',gi_end_seconds='',slt_start_seconds='',slt_end_seconds='',gt_start_seconds='',
                    transition_support='unpaired context, not a verified protocol trial')
    info.update(imputed_signal_values=int(mask.sum()),remaining_missing_signal_values=0)
    info['flags']+='; '+'; '.join(item[6])
    return z,mask,gaps,phases,info,events


def generate(subject):
    out=ROOT/'tempgraphs'/subject.lower();out.mkdir(exist_ok=True)
    trials,sync,windows=load_trials(subject)
    records=[];manifest=[];intervals=[];landmarks=[];transforms=[];gap_report=[]
    for item in trials:
        record,number,grid,observed,quality,boundary,notes=item
        z,mask,gaps,phases,info,events=analyze(item,trials)
        records.append(dict(record=record,trial=number,duration_seconds=float(grid[-1]),boundary_method=boundary,**info))
        intervals.extend(dict(record=record,trial=number,**p) for p in phases)
        landmarks.extend(dict(record=record,trial=number,**p) for p in events)
        gap_report.extend(dict(record=record,trial=number,**p) for p in gaps)
        labels=np.full(len(grid),'UNKNOWN',dtype='<U10')
        for p in phases:labels[p['start_index']:p['end_index']]=p['phase']
        for side,cols in [('L',list(range(8))+[16,17]),('R',list(range(8,16))+[18,19])]:
            (out/record).mkdir(exist_ok=True)
            target=out/'segmented_data'/record/f'trial_{number:02d}_{side.lower()}_segmented.csv'
            target.parent.mkdir(parents=True,exist_ok=True)
            png=f'{record}/trial_{number:02d}_{side.lower()}_complete.png'
            message=f'Transition: {info["transition_support"]}; alternative fits {info["alternative_transition_start_seconds"]}-{info["alternative_transition_end_seconds"]} s (not a confidence interval).'
            if not info['complete_six_phase']:message='Partial annotation: insufficient evidence for all six phases; UNKNOWN denotes unmatched-trigger context.'
            qtext=f'{boundary} | source-grid coverage: FMG {quality["F"]["coverage_fraction"]:.0%}, {side} insole {quality[side]["coverage_fraction"]:.0%} | estimated {int(mask[:,cols].sum())}; remaining blanks 0'
            tr=render(out/png,f'{subject} | {record} | Recording interval {number:02d} | {side}',grid,z[:,cols],[],info,qtext,phases,message,mask[:,cols])
            transforms.extend(dict(record=record,trial=number,side=side,**p) for p in tr)
            names=[*[f'FMG_channel_{k+1}' for k in range(8)],'CoP','vGRF']
            with target.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['time_seconds',*names,'provisional_phase','label_status','signal_missing',*[n+'_imputed' for n in names]])
                for t,x,label,im in zip(grid,z[:,cols],labels,mask[:,cols]):
                    w.writerow([f'{t:.2f}',*[f'{v:.17g}' for v in x],label,'model_assisted_not_ground_truth',0,*im.astype(int)])
            manifest.append(dict(record=record,trial=number,side=side,png=png,segmented_csv=str(target.relative_to(out)),
                quality_json=json.dumps(quality),boundary_method=boundary,duration_seconds=float(grid[-1]),**info))
        print(f'{subject} {record}/{number:02d}: {info["status"]}; {int(mask.sum())} estimated values',flush=True)
    summary=dict(subject=subject,graphs=len(manifest),recording_intervals=len(trials),
        complete_six_phase_intervals=sum(r['complete_six_phase'] for r in records),
        partial_phase_intervals=[dict(record=r['record'],trial=r['trial']) for r in records if not r['complete_six_phase']],
        estimated_values=sum(r['imputed_signal_values'] for r in records),remaining_missing_values=0,
        reconstruction_methods=dict(Counter(g['method'] for g in gap_report)),
        warning='Model-completed curves; estimated values are not recovered measurements. Labels are not ground truth.')
    for name,rows in [('complete_trial_manifest',manifest),('phase_boundaries_provisional',records),
        ('phase_intervals_provisional',intervals),('load_landmarks_provisional',landmarks),
        ('display_transforms',transforms),('missing_value_report',gap_report),('synchronization_report',sync),('recording_windows',windows)]:
        write_rows(out/f'{name}.csv',rows)
    review=out/'review_sheets';review.mkdir(exist_ok=True)
    for page in range((len(manifest)+3)//4):
        sheet=Image.new('RGB',(1600,640),'white')
        for j,row in enumerate(manifest[page*4:page*4+4]):
            with Image.open(out/row['png']) as im:
                im.thumbnail((800,320));sheet.paste(im,((j%2)*800,(j//2)*320))
        sheet.save(review/f'page_{page+1:02d}.png')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def verify(subject):
    out=ROOT/'tempgraphs'/subject.lower()
    provenance=json.loads((out/'provenance.json').read_text())
    assert protected_state()==provenance['protected_files']
    with (out/'complete_trial_manifest.csv').open(newline='') as f:manifest=list(csv.DictReader(f))
    index={(r['record'],int(r['trial']),r['side']):r for r in manifest}
    trials,_,_=load_trials(subject);count=0;preserved=0;estimated=0;six=0
    for item in trials:
        record,number,grid,original,*_=item
        z,mask,_,phases,info,_=analyze(item,trials)
        assert np.isfinite(z).all() and np.array_equal(mask,~np.isfinite(original))
        assert np.array_equal(z[~mask],original[~mask]);preserved+=int((~mask).sum());estimated+=int(mask.sum())
        labels=np.full(len(grid),'UNKNOWN',dtype='<U10')
        assert phases[0]['start_index']==0 and phases[-1]['end_index']==len(grid)
        for p in phases:
            assert p['end_index']>p['start_index'];labels[p['start_index']:p['end_index']]=p['phase']
        for a,b in zip(phases[:-1],phases[1:]):assert a['end_index']==b['start_index']
        if info['complete_six_phase']:
            assert [p['phase'] for p in phases]==['QS','GI','SSSW','SLT','SSLW','GT'];six+=1
        for side,cols in [('L',list(range(8))+[16,17]),('R',list(range(8,16))+[18,19])]:
            row=index[(record,number,side)]
            with (out/row['segmented_csv']).open(newline='') as f:samples=list(csv.reader(f))[1:]
            values=np.array([[float(v) for v in r[:11]] for r in samples])
            np.testing.assert_allclose(values[:,0],grid,atol=1e-10)
            np.testing.assert_allclose(values[:,1:],z[:,cols],atol=1e-12,rtol=1e-12)
            assert np.array_equal(np.array([[int(v) for v in r[14:]] for r in samples]),mask[:,cols])
            assert [r[11] for r in samples]==labels.tolist()
            assert all(r[12]=='model_assisted_not_ground_truth' and r[13]=='0' for r in samples)
            with Image.open(out/row['png']) as im:assert im.size==(2200,880);im.verify()
            count+=1
    assert count==len(manifest)==2*len(trials)
    assert protected_state()==provenance['protected_files']
    summary=json.loads((out/'summary.json').read_text())
    assert summary['estimated_values']==estimated and summary['complete_six_phase_intervals']==six
    report=dict(verified_pngs=count,verified_numeric_exports=count,remaining_missing_values=0,
        original_source_grid_values_preserved=preserved,estimated_values=estimated,complete_six_phase_intervals=six,
        raw_and_previous_subject_files_unchanged=True,all_estimates_flagged=True,
        scope='Numeric/file consistency and reproducibility, NOT physiological or ground-truth validation')
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(subject,json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',choices=(*SUBJECTS,'all'),default='all')
    p.add_argument('--audit',action='store_true');p.add_argument('--verify',action='store_true');args=p.parse_args()
    selected=SUBJECTS if args.subject=='all' else (args.subject,)
    if args.verify:
        for subject in selected:verify(subject)
        return
    if args.audit:
        for subject in selected:
            trials,sync,windows=load_trials(subject)
            print(json.dumps(dict(subject=subject,intervals=len(trials),unpaired=[w for w in windows if w['boundary_method'].startswith('unpaired')],
                clock_epochs=[{k:v for k,v in s.items() if k not in ('matched_reference_indices','matched_source_indices')} for s in sync]),indent=2),flush=True)
        return
    before=protected_state();results=[]
    for subject in selected:
        result=generate(subject);assert protected_state()==before
        provenance=dict(protected_files=before,protected_files_unchanged=True,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            view_mode='model_completed',warning='Reconstructed measurements and estimated phase labels are unverified')
        (ROOT/'tempgraphs'/subject.lower()/'provenance.json').write_text(json.dumps(provenance,indent=2))
        results.append(result)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
