"""Subject 05 only. Reuse Subject 08 analysis without editing its files.

Run with Python -B to avoid creating/updating imported bytecode caches.
All created outputs are confined to this script's directory.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from generate_qs_plots import load_stream, split_source_files, unit_trial_starts
from subject08_review import DT, grid_values, render, write_rows
from subject08_segmentation import segment_trial

SOURCE=ROOT/'dataset'/'Sub05_H'


def fingerprint(paths,base):
    return {str(p.relative_to(base)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
             bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(paths)}


def protected_state():
    paths=[p for p in (ROOT/'tempgraphs').rglob('*') if p.is_file() and HERE not in p.parents]
    paths.extend(p for name in ('Sub05_H','Sub08_H') for p in (ROOT/'dataset'/name).rglob('*') if p.is_file())
    paths.extend(ROOT/name for name in ('subject08_review.py','subject08_segmentation.py','generate_tempgraphs.py'))
    return fingerprint(paths,ROOT)


def align_edges(f,i):
    ft=f.values[f.rising_edges,0]; it=i.values[i.rising_edges,0]
    if not len(ft) or not len(it): raise ValueError('Missing trigger sequence')
    offset=it[0]-ft[0]
    for _ in range(3):
        nearest=np.argmin(np.abs((ft+offset)[:,None]-it[None,:]),axis=1)
        valid=np.abs(it[nearest]-ft-offset)<.25
        if np.mean(valid)<.8: raise ValueError('Trigger sequence cannot be reliably paired')
        offset=float(np.median(it[nearest[valid]]-ft[valid]))
    mapping={int(j):int(i.rising_edges[nearest[j]]) for j in np.flatnonzero(valid)}
    if len(set(mapping.values()))!=len(mapping): raise ValueError('Non-unique trigger correspondence')
    if list(mapping.values())!=sorted(mapping.values()): raise ValueError('Nonmonotonic trigger correspondence')
    residuals=it[nearest[valid]]-ft[valid]-offset
    return mapping,dict(clock_offset_seconds=offset,matched_edges=len(mapping),
        fmg_edges=len(ft),insole_edges=len(it),max_matched_residual_seconds=float(np.max(np.abs(residuals))),
        missing_fmg_edge_numbers=[int(j+1) for j in range(len(ft)) if j not in mapping])


def iter_subject05():
    fm,ins=split_source_files(SOURCE)
    for record,fp in fm.items():
        f=load_stream(fp,18,'FMG')
        streams={s:load_stream(ins[(record,s)],4,'insole',s) for s in ('L','R')}
        aligned={s:align_edges(f,i) for s,i in streams.items()}
        for a,b,trial in unit_trial_starts(f):
            if b is None: raise ValueError('Missing FMG trial-end trigger')
            ft=f.values[a:b+1,0]-f.values[a,0]
            raw={}; quality={}
            for side,i in streams.items():
                mapping,audit=aligned[side]
                first,last=2*(trial-1),2*trial-1
                inferred=[]
                if first in mapping: start=float(i.values[mapping[first],0])
                else:
                    start=float(f.values[a,0]+audit['clock_offset_seconds']); inferred.append('start')
                if last in mapping: end=float(i.values[mapping[last],0])
                else:
                    end=float(f.values[b,0]+audit['clock_offset_seconds']); inferred.append('end')
                if first in mapping and last in mapping:
                    block=i.values[mapping[first]:mapping[last]+1]
                else:
                    block=i.values[(i.values[:,0]>=start)&(i.values[:,0]<=end)]
                if len(block)<2: raise ValueError(f'No observed insole samples: {record}/{trial}/{side}')
                it=block[:,0]-start
                raw[side]=(it,block[:,2:4])
                quality[side]=dict(full_fmg_seconds=float(ft[-1]),full_insole_seconds=float(it[-1]),
                    trigger_end_error_seconds=float(end-start-ft[-1]),
                    trigger_boundary_status='observed' if not inferred else 'inferred '+','.join(inferred)+' from matched clock offset',
                    fmg_gaps=int(np.sum(np.diff(ft)>.10)),insole_gaps=int(np.sum(np.diff(it)>.10)),
                    nonmonotonic_or_duplicate_timestamps=int(np.sum(np.diff(ft)<=0)+np.sum(np.diff(it)<=0)),
                    invalid_fmg_rows_in_source=f.invalid_rows,invalid_insole_rows_in_source=i.invalid_rows,
                    **audit)
            grid=np.arange(0,max(ft[-1],*(t[-1] for t,_ in raw.values()))+.005,DT)
            z=np.column_stack([grid_values(ft,f.values[a:b+1,2:18],grid),
                               *(grid_values(*raw[s],grid) for s in ('L','R'))])
            yield record,trial,grid,z,quality


def evaluate(grid,z,quality):
    phases,info,landmarks=segment_trial(grid,z)
    for side,q in quality.items():
        if q['trigger_boundary_status']!='observed': info['flags']+=f'; {side} insole {q["trigger_boundary_status"]}'
        if q['fmg_gaps'] or q['insole_gaps']: info['flags']+=f'; {side} timestamp gaps retained'
    return phases,info,landmarks


def generate(diagnose=False):
    before=protected_state()
    records=[]; plots=[]; intervals=[]; landmarks=[]; transforms=[]; sync=[]
    for record,trial,grid,z,quality in iter_subject05():
        phases,info,events=evaluate(grid,z,quality)
        records.append(dict(record=record,trial=trial,**info))
        if diagnose: continue
        intervals.extend(dict(record=record,trial=trial,**p) for p in phases)
        landmarks.extend(dict(record=record,trial=trial,**p) for p in events)
        labels=np.full(len(grid),'UNRESOLVED',dtype='<U10')
        for p in phases: labels[p['start_index']:p['end_index']]=p['phase']
        for side,cols in [('L',list(range(8))+[16,17]),('R',list(range(8,16))+[18,19])]:
            local=z[:,cols]; q=quality[side]
            png=f'{record}/trial_{trial:02d}_{side.lower()}_complete.png'
            path=HERE/png
            note=(f'Transition: {info["transition_support"]}. Alternative fits: '
                  f'{info["alternative_transition_start_seconds"]}-{info["alternative_transition_end_seconds"]} s; not a confidence interval.') if phases else info['flags']
            quality_text=f'Triggers: {q["trigger_boundary_status"]} | end difference {q["trigger_end_error_seconds"]:+.3f} s | gaps: FMG {q["fmg_gaps"]}, insole {q["insole_gaps"]} | repeated/reversed timestamps {q["nonmonotonic_or_duplicate_timestamps"]}'
            tr=render(path,f'Sub05_H | {record} | Trigger-pair trial {trial:02d} | {side}',grid,local,[],info,quality_text,phases,note)
            transforms.extend(dict(record=record,trial=trial,side=side,**r) for r in tr)
            csvpath=HERE/'segmented_data'/record/f'trial_{trial:02d}_{side.lower()}_segmented.csv'
            csvpath.parent.mkdir(parents=True,exist_ok=True)
            with csvpath.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f); w.writerow(['time_seconds',*[f'FMG_channel_{k+1}' for k in range(8)],'CoP','vGRF','provisional_phase','label_status','signal_missing'])
                for t,x,label in zip(grid,local,labels):
                    w.writerow([f'{t:.2f}',*[f'{v:.17g}' if np.isfinite(v) else '' for v in x],label,
                        'unresolved' if label=='UNRESOLVED' else 'provisional_assumed_short_to_long',int(not np.all(np.isfinite(x)))])
            plots.append(dict(record=record,trial=trial,side=side,png=png,segmented_csv=str(csvpath.relative_to(HERE)),**q,**info))
            sync.append(dict(record=record,trial=trial,side=side,**q))
    summary=dict(recorded_trials=len(records),graphs=len(plots),provisional_trials=sum(r['status']!='unresolved' for r in records),
        support_counts=dict(Counter(r['transition_support'] for r in records)),
        unresolved_trials=[dict(record=r['record'],trial=r['trial'],reason=r['flags']) for r in records if r['status']=='unresolved'])
    if not diagnose:
        for name,rows in [('phase_boundaries_provisional',records),('phase_intervals_provisional',intervals),
                          ('load_landmarks_provisional',landmarks),('complete_trial_manifest',plots),
                          ('display_transforms',transforms),('synchronization_report',sync)]:
            write_rows(HERE/f'{name}.csv',rows)
        review=HERE/'review_sheets'; review.mkdir(exist_ok=True)
        for page in range((len(plots)+3)//4):
            sheet=Image.new('RGB',(1600,640),'white')
            for j,row in enumerate(plots[4*page:4*page+4]):
                with Image.open(HERE/row['png']) as im:
                    im.thumbnail((800,320)); sheet.paste(im,((j%2)*800,(j//2)*320))
            sheet.save(review/f'page_{page+1:02d}.png')
    after=protected_state()
    assert before==after, 'A protected file changed: review before reporting completion'
    summary['subject08_and_existing_tempgraphs_unchanged']=True
    summary['raw_subject05_unchanged']=True
    if not diagnose:
        provenance=dict(summary=summary,protected_files_before=before,protected_files_after=after,
            generation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            method='Unmodified Subject 08 provisional segmentation and renderer; Subject 05 timestamp-based trigger matching',
            warning='Short-to-long sequence is assumed, not verified ground truth')
        (HERE/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--diagnose',action='store_true')
    generate(p.parse_args().diagnose)
