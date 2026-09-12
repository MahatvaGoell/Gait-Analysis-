"""Create Subject 06/07 plots only; preserve all existing subjects.

Same provisional six-stage estimator as Subject 08. Acquisition-specific
loading handles single-start files, clock epochs, absent streams, and missing
trigger events. Raw files stay unchanged; bounded grid-gap estimates are flagged.
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
from generate_qs_plots import ROOT, DATASET_ROOT, load_stream, split_source_files
from subject08_review import DT, grid_values, write_rows
from subject08_segmentation import segment_trial
from additional_subject_renderer import render
from missing_value_refinement import refine, segment_refined
from complete_reconstruction import complete_trial, complete_segments

OUTPUTS=[ROOT/'tempgraphs'/n for n in ('sub06_h','sub07_h')]


def snapshot():
    paths=[p for p in (ROOT/'tempgraphs').rglob('*') if p.is_file() and not any(o in p.parents for o in OUTPUTS)]
    paths += [p for p in DATASET_ROOT.rglob('*') if p.is_file()]
    paths += [ROOT/n for n in ('subject08_review.py','subject08_segmentation.py','generate_tempgraphs.py')]
    return {str(p.relative_to(ROOT)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size,
              mtime_ns=p.stat().st_mtime_ns) for p in sorted(paths)}


def read(path,cols):
    if path is None: return None
    st=load_stream(path,cols,'FMG' if cols==18 else 'insole')
    v=st.values; t=v[:,0]; remove=[]
    # Exclude isolated timestamp corruptions only when surrounding packets
    # agree within 0.5 seconds. Do not repair timestamps or remove real resets.
    if len(t)>=5:
        neighbors=np.column_stack([t[:-4],t[1:-3],t[3:-1],t[4:]])
        spread=np.ptp(neighbors,axis=1);middle=t[2:-2]
        deviation=abs(middle-np.median(neighbors,axis=1))
        bad=((spread<.5)&(deviation>1)) | ((deviation>np.maximum(1.,100*spread)) &
                ((middle<neighbors.min(axis=1)-1)|(middle>neighbors.max(axis=1)+1)))
        remove=(np.flatnonzero(bad)+2).tolist()
    keep=np.ones(len(v),bool); keep[remove]=False
    v=v[keep]
    edges=np.flatnonzero((v[1:,1]==1)&(v[:-1,1]==0))+1
    return dict(values=v,edges=edges,path=str(path.relative_to(ROOT)),invalid_rows=st.invalid_rows,
                excluded_timestamp_outliers=len(remove),excluded_source_value_indices=remove)


def offsets_between(master,other):
    mt=master['values'][master['edges'],0]; ot=other['values'][other['edges'],0]
    if not len(ot): raise ValueError('No triggers to establish batch clock offset')
    offset=ot[0]-mt[0]
    for _ in range(4):
        near=np.argmin(abs((mt+offset)[:,None]-ot[None,:]),axis=1)
        ok=abs(ot[near]-mt-offset)<.25
        if ok.mean()<.65: raise ValueError('Batch trigger correspondence insufficient')
        offset=float(np.median(ot[near[ok]]-mt[ok]))
    mapping={int(j):float(ot[near[j]]) for j in np.flatnonzero(ok)}
    if len(set(mapping.values()))!=len(mapping): raise ValueError('Ambiguous trigger mapping')
    return offset,mapping,float(np.max(abs(ot[near[ok]]-mt[ok]-offset)))


def catalog(subject):
    fm,ins=split_source_files(DATASET_ROOT/subject)
    result={}
    for record,path in fm.items():
        result[record]={'F':read(path,18),**{s:read(ins.get((record,s)),4) for s in ('L','R')}}
    return result


def calibrated_offsets(data,subject):
    # Clock reset in rohan_3 requires separate recording epochs. Calibration
    # for missing single-file pulses uses only neighboring compatible records.
    groups={}
    for record,streams in data.items():
        number=int(record.rsplit('_',1)[1])
        group='all' if subject=='Sub06_H' else ('early' if number<=3 else 'middle')
        if subject=='Sub07_H' and (number>=12 or number==3): continue
        f=streams['F']
        if len(f['edges'])!=1:continue
        ft=f['values'][f['edges'][0],0]
        for side in ('L','R'):
            s=streams[side]
            if s is not None and len(s['edges'])==1:
                groups.setdefault((group,side),[]).append(float(s['values'][s['edges'][0],0]-ft))
    return {k:dict(offset=float(np.median(v)),support=len(v),residual_max=float(max(abs(np.array(v)-np.median(v))))) for k,v in groups.items()}


def trial_windows(subject,data):
    calibrations=calibrated_offsets(data,subject)
    for record,streams in data.items():
        f=streams['F']; number=int(record.rsplit('_',1)[1])
        is_batch=subject=='Sub07_H' and number>=12
        if is_batch:
            master_side=max(('F','L','R'),key=lambda s:len(streams[s]['edges']) if streams[s] else -1)
            master=streams[master_side]; mt=master['values'][master['edges'],0]
            if len(mt)%2: raise ValueError(f'{record}: unpaired master trigger sequence')
            mappings={s:offsets_between(master,st) for s,st in streams.items() if st is not None}
            for j in range(len(mt)//2):
                starts={}; ends={}; notes=[]
                for s in ('F','L','R'):
                    offset,mapping,resid=mappings[s]
                    starts[s]=mapping.get(2*j,float(mt[2*j]+offset))
                    ends[s]=mapping.get(2*j+1,float(mt[2*j+1]+offset))
                    missing=[lab for edge,lab in ((2*j,'start'),(2*j+1,'end')) if edge not in mapping]
                    if missing: notes.append(s+' inferred '+','.join(missing)+' trigger from matched batch clock offset')
                yield record,j+1,streams,starts,ends,'paired triggers; reference '+master_side,notes
        else:
            v=f['values']; resets=np.flatnonzero(np.diff(v[:,0]) < -5)+1
            cuts=np.r_[0,resets,len(v)]
            if len(resets) and not (subject=='Sub07_H' and number==3):
                raise ValueError(f'Unexpected sustained clock reset in {record}')
            for episode,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
                e=[int(x) for x in f['edges'] if a<=x<b]
                if len(e)>1:raise ValueError('Unexpected multi-trigger single-file epoch')
                notes=[]; starts={}; ends={}
                group='all' if subject=='Sub06_H' else ('early' if number<=3 else 'middle')
                if e: starts['F']=float(v[e[0],0])
                else:
                    side=next(s for s in ('R','L') if streams[s] is not None and len(streams[s]['edges'])>episode and (group,s) in calibrations)
                    starts['F']=float(streams[side]['values'][streams[side]['edges'][episode],0]-calibrations[(group,side)]['offset'])
                    notes.append('F start inferred from cross-record trigger clock calibration')
                ends['F']=float(v[b-1,0])
                duration=ends['F']-starts['F']
                if duration<=0 or duration>120:raise ValueError('Implausible file-end window')
                for side in ('L','R'):
                    s=streams[side]
                    if s is None:
                        notes.append(side+' insole source missing'); continue
                    if len(s['edges'])>episode: starts[side]=float(s['values'][s['edges'][episode],0])
                    else:
                        starts[side]=starts['F']+calibrations[(group,side)]['offset']
                        notes.append(side+' start inferred from cross-record trigger clock calibration')
                    ends[side]=starts[side]+duration
                notes.append('No end trigger: window ends at observed FMG file/epoch end; ending may be incomplete')
                if len(resets): notes.append('FMG clock reset: plotted as a separate recording epoch')
                yield record,episode+1,streams,starts,ends,'start trigger to FMG file/epoch end',notes


def iter_trials(subject):
    data=catalog(subject)
    for record,trial,streams,starts,ends,boundary,notes in trial_windows(subject,data):
        duration=max(ends[s]-starts[s] for s in starts)
        if not 0<duration<=120:raise ValueError('Invalid trial duration')
        grid=np.arange(0,duration+.005,DT)
        arrays=[]; quality={}
        for s,cols in [('F',16),('L',2),('R',2)]:
            stream=streams[s]
            if stream is None:
                arrays.append(np.full((len(grid),cols),np.nan)); quality[s]={'missing':True,'coverage_fraction':0.}; continue
            v=stream['values']; block=v[(v[:,0]>=starts[s])&(v[:,0]<=ends[s])]
            if len(block)<2:
                arr=np.full((len(grid),cols),np.nan); gaps=0; bad=0
            else:
                t=block[:,0]-starts[s]; arr=grid_values(t,block[:,2:],grid)
                gaps=int(np.sum(np.diff(t)>.10)); bad=int(np.sum(np.diff(t)<=0))
            arrays.append(arr)
            quality[s]=dict(missing=False,coverage_fraction=float(np.mean(np.all(np.isfinite(arr),axis=1))),
                gaps_over_100ms=gaps,repeated_reversed_timestamps=bad,
                excluded_timestamp_outliers=stream['excluded_timestamp_outliers'],invalid_rows=stream['invalid_rows'],
                excluded_source_value_indices=stream['excluded_source_value_indices'],
                source=stream['path'],source_start=starts[s],source_end=ends[s])
        z=np.column_stack(arrays)
        yield record,trial,grid,z,quality,boundary,notes


def evaluate(grid,z,notes,imputed=None):
    if imputed is None: imputed=np.zeros_like(z,dtype=bool)
    missing=any(not np.any(np.isfinite(z[:,k])) for k in range(20))
    no_baseline=any(np.sum(np.isfinite(z[:75,k]) & ~imputed[:75,k])<30 for k in range(20))
    if missing or no_baseline:
        # Get the estimator's normal unresolved schema without feeding NaNs
        # to its local statistics; never draw placeholder signal values.
        phases,info,events=segment_trial(grid,np.ones_like(z))
        info['flags']='Required sensor stream is missing or absent throughout this window' if missing else 'Insufficient observed initial baseline for the six-phase estimator'
        info['excluded_incomplete_cycles']=0
    else:
        phases,info,events=segment_refined(grid,z,imputed)
    info['imputed_signal_values']=int(imputed.sum())
    info['remaining_missing_signal_values']=int((~np.isfinite(z)).sum())
    info['flags']+='; '+'; '.join(notes)
    return phases,info,events


def generate(subject,diagnose=False,complete_model=False):
    out=ROOT/'tempgraphs'/subject.lower()
    records=[]; manifests=[]; intervals=[]; landmarks=[]; transforms=[]; gap_report=[]
    trials=list(iter_trials(subject))
    for item in trials:
        record,trial,grid,observed,quality,boundary,notes=item
        z,imputed,gaps=complete_trial(item,trials) if complete_model else refine(grid,observed)
        gap_report.extend(dict(record=record,trial=trial,**g) for g in gaps)
        if complete_model:
            phases,info,events=complete_segments(grid,z)
            info.update(imputed_signal_values=int(imputed.sum()),remaining_missing_signal_values=0)
            info['flags']+='; '+'; '.join(notes)
        else:phases,info,events=evaluate(grid,z,notes,imputed)
        records.append(dict(record=record,trial=trial,boundary_method=boundary,duration_seconds=float(grid[-1]),**info))
        if diagnose:continue
        print(f'{subject} {record}/{trial:02d}: {info["status"]}; estimated {int(imputed.sum())} values',flush=True)
        intervals.extend(dict(record=record,trial=trial,**p) for p in phases)
        landmarks.extend(dict(record=record,trial=trial,**p) for p in events)
        labels=np.full(len(grid),'UNRESOLVED',dtype='<U10')
        for p in phases:labels[p['start_index']:p['end_index']]=p['phase']
        for side,cols in [('L',list(range(8))+[16,17]),('R',list(range(8,16))+[18,19])]:
            local=z[:,cols]
            png=f'{record}/trial_{trial:02d}_{side.lower()}_complete.png'
            message=(f'Transition: {info["transition_support"]}; alternative fits '
                f'{info["alternative_transition_start_seconds"]}-{info["alternative_transition_end_seconds"]} s (not a confidence interval).') if phases else info['flags']
            if complete_model and not info['complete_six_phase']:
                message='Partial phase evidence: WALK combines unresolved short/long walking states; missing phases are not invented.'
            qtext=f'{boundary} | FMG observed {quality["F"]["coverage_fraction"]:.0%}, {side} insole {quality[side]["coverage_fraction"]:.0%} | estimated {int(imputed[:,cols].sum())} values; still missing {int((~np.isfinite(local)).sum())}'
            if any('inferred' in n for n in notes):qtext+=' | inferred trigger(s); see report'
            tr=render(out/png,f'{subject} | {record} | Recording interval {trial:02d} | {side}',grid,local,[],info,qtext,phases,message,imputed[:,cols])
            transforms.extend(dict(record=record,trial=trial,side=side,**r) for r in tr)
            target=out/'segmented_data'/record/f'trial_{trial:02d}_{side.lower()}_segmented.csv'
            assert target.parent.is_dir(), 'Refinement must use existing folders only'
            with target.open('w',newline='',encoding='utf-8') as f:
                names=[*[f'FMG_channel_{k+1}' for k in range(8)],'CoP','vGRF']
                w=csv.writer(f);w.writerow(['time_seconds',*names,'provisional_phase','label_status','signal_missing',*[n+'_imputed' for n in names]])
                for t,x,label,mask in zip(grid,local,labels,imputed[:,cols]):
                    w.writerow([f'{t:.2f}',*[f'{v:.17g}' if np.isfinite(v) else '' for v in x],label,
                        ('model_assisted_not_ground_truth' if complete_model else ('unresolved' if label=='UNRESOLVED' else 'provisional_assumed_short_to_long')),int(not np.all(np.isfinite(x))),*mask.astype(int)])
            manifests.append(dict(record=record,trial=trial,side=side,png=png,segmented_csv=str(target.relative_to(out)),
                boundary_method=boundary,quality_json=json.dumps(quality),duration_seconds=float(grid[-1]),**info))
    summary=dict(subject=subject,recording_intervals=len(records),graphs=len(manifests),
        imputed_signal_values=sum(r['imputed_signal_values'] for r in records),
        remaining_missing_signal_values=sum(r['remaining_missing_signal_values'] for r in records),
        interpolated_channel_gaps=sum(g['method']=='linear_interpolation' for g in gap_report),
        gap_handling=('MODEL-COMPLETED: cross-sensor ridge versus temporal interpolation; endpoint holds; absent streams use same-subject donor models. NOT recovered measurements.' if complete_model else 'Bounded linear interpolation: endpoint span <=0.50 s; >=20 same-channel held-out blocks, p90 normalized RMSE <=5% and max <=10%; no edge extrapolation'),
        view_mode='model_completed' if complete_model else 'conservative_short_gap',
        complete_six_phase_intervals=sum(r.get('complete_six_phase',r['status']!='unresolved') for r in records),
        partial_phase_intervals=[dict(record=r['record'],trial=r['trial']) for r in records if r['status']=='partial phase annotation'],
        provisional_intervals=sum(r['status']!='unresolved' for r in records),support_counts=dict(Counter(r['transition_support'] for r in records)),
        unresolved_intervals=[dict(record=r['record'],trial=r['trial'],reason=r['flags']) for r in records if r['status']=='unresolved'])
    if not diagnose:
        for name,rows in [('complete_trial_manifest',manifests),('phase_boundaries_provisional',records),
            ('phase_intervals_provisional',intervals),('load_landmarks_provisional',landmarks),('display_transforms',transforms),('missing_value_report',gap_report)]:
            write_rows(out/f'{name}.csv',rows)
        review=out/'review_sheets';assert review.is_dir()
        for page in range((len(manifests)+3)//4):
            sheet=Image.new('RGB',(1600,640),'white')
            for j,row in enumerate(manifests[4*page:4*page+4]):
                with Image.open(out/row['png']) as im:
                    im.thumbnail((800,320));sheet.paste(im,((j%2)*800,(j//2)*320))
            sheet.save(review/f'page_{page+1:02d}.png')
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',choices=('Sub06_H','Sub07_H','both'),default='both')
    p.add_argument('--diagnose',action='store_true')
    p.add_argument('--complete-model',action='store_true',help='Explicit estimated reconstruction, including long gaps and absent streams')
    args=p.parse_args()
    before=snapshot()
    folders_before=sorted(str(p) for p in (ROOT/'tempgraphs').rglob('*') if p.is_dir())
    results=[generate(s,args.diagnose,args.complete_model) for s in (('Sub06_H','Sub07_H') if args.subject=='both' else (args.subject,))]
    after=snapshot(); assert before==after,'Protected source or output changed'
    assert folders_before==sorted(str(p) for p in (ROOT/'tempgraphs').rglob('*') if p.is_dir()), 'Folder structure changed'
    if not args.diagnose:
        for result in results:
            out=ROOT/'tempgraphs'/result['subject'].lower()
            (out/'provenance.json').write_text(json.dumps(dict(protected_files=before,protected_files_unchanged=True,
                generation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                refinement_script_sha256=hashlib.sha256((ROOT/'missing_value_refinement.py').read_bytes()).hexdigest(),
                renderer_script_sha256=hashlib.sha256((ROOT/'additional_subject_renderer.py').read_bytes()).hexdigest(),
                completion_script_sha256=hashlib.sha256((ROOT/'complete_reconstruction.py').read_bytes()).hexdigest(),
                view_mode=result['view_mode'],
                folder_structure_unchanged=True,
                note='Provisional assumed short-to-long labels; not verified ground truth'),indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
