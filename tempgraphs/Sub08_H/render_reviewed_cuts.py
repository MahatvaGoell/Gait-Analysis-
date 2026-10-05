"""Render Sub08_H from explicitly reviewed per-recording cuts, without refitting.

The CSV contains individual visual-review decisions. These are provisional
signal annotations, not video-validated foot-contact or step-length labels.
"""
import csv
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tempgraphs')]
import resegment_sub08_five_steps as source
from reference_layout import render
from refine_valley_samples import refine
from generate_qs_plots import font

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    snapshot = HERE / 'pre_individual_review_manifest.csv'
    if not snapshot.exists():
        shutil.copy2(source.MANIFEST, snapshot)
    rows = read(snapshot)
    decisions = {(r['record'], r['trial_side']): r for r in read(HERE / 'reviewed_cuts.csv')}
    reviewed = []
    valley_audit = []
    for row in rows:
        path = ROOT / row['source_csv']
        parts = path.name.split('_')
        decision = decisions.get((path.parent.name, parts[1] + parts[2]))
        if decision is None:
            continue
        t, values, missing = source.read_trial(path)
        cuts = [float(decision[k]) for k in ('gi_start','sssw_start','slt_start','sslw_start','gt_start')]
        valleys = list(map(float, row['vgrf_valleys_seconds'].split(';')))
        if decision['replace_preparatory_with_terminal']:
            valleys = valleys[1:] + [float(decision['replace_preparatory_with_terminal'])]
        gi, first, slt, sslw, gt = cuts
        assert 0 < gi < first < slt < sslw < gt < t[-1], path
        events = refine(t, values[:,9], missing[:,9], valleys,
                        [(first,slt),(slt,sslw),(sslw,gt)],path.parent.name,parts[1]+parts[2])
        valleys = [event['seconds'] for event in events]
        for event in events:
            valley_audit.append(dict(source_csv=row['source_csv'], **event))
        counts = [sum(a < v < b for v in valleys) for a,b in ((first,slt),(slt,sslw),(sslw,gt))]
        assert counts == [5,1,5], (path, counts)
        # Show available recorded GT only; do not fabricate an ending plateau.
        end = min(float(t[-1]), gt + 3)
        boundaries = [float(t[0]), *cuts, end]
        phases = [dict(phase=name, start_seconds=a, end_seconds=b)
                  for name,a,b in zip(('QS','GI','SSSW','SLT','SSLW','GT'),boundaries,boundaries[1:])]
        candidates, meta = source.candidate_valleys(t, values[:,9])
        extra = [float(t[i]) for i in candidates if first < t[i] < end
                 and min(abs(float(t[i])-v) for v in valleys) > .30]
        # Small within-cycle candidates can be ripples. Retain them as an
        # explicit audit concern rather than claiming that selected count
        # proves all physical steps have been found.
        concern = ('Additional unclassified minima: ' + ', '.join(f'{v:.2f}s' for v in extra)) if extra else ''
        row.update(gi_start_seconds=f'{gi:.2f}', sssw_start_seconds=f'{first:.2f}',
                   sssw_end_seconds=f'{slt:.2f}', slt_end_seconds=f'{sslw:.2f}',
                   gt_start_seconds=f'{gt:.2f}', gt_end_seconds=f'{end:.2f}',
                   vgrf_valleys_seconds=';'.join(f'{v:.2f}' for v in valleys),
                   segmentation_method='individual reviewed cuts; valleys refined on unsmoothed input in local windows (15 samples; two reviewed exceptions)',
                   review_issue=concern)
        stop = np.searchsorted(t,end,side='right')
        target = ROOT / row['png']
        render(target, source.graph_title(path), t[:stop],values[:stop],[],
               dict(onset_seconds=gi,view_mode='model_completed',valleys=valleys,valley_events=events),
               concern or 'Valley markers: local input minima; flat-bottom ties use middle minimum sample. Counts: 5 / 1 / 5.',phases,
               'Individual reviewed cuts | S1-S5: short steps | T: transition | L1-L5: long steps',missing[:stop])
        reviewed.append(row)
    write(source.MANIFEST, rows)
    write(HERE / 'valley_sample_audit.csv', valley_audit)
    source.update_global_manifest(rows)
    combined = ROOT/'tempgraphs'/'all_subject_five_step_segmentation_manifest.csv'
    if combined.exists():
        all_rows=read(combined)
        updates={r['source_csv']:r for r in reviewed}
        for r in all_rows:
            if r['source_csv'] in updates:
                u=updates[r['source_csv']]
                r.update(u)
                r['first_landing_seconds']=u['vgrf_valleys_seconds'].split(';')[0]
                r['gt_duration_seconds']=f"{float(u['gt_end_seconds'])-float(u['gt_start_seconds']):.2f}"
        write(combined,all_rows)
    source.write_review_sheets(rows)
    print('Individually annotated:',len(reviewed))
    print('5/1/5 selected-event checks: all passed')
    print('Remaining extra-minimum concerns:',sum(bool(r['review_issue']) for r in reviewed))
    print('Insufficient recordings:',len(rows)-len(reviewed))
    print('Refined valley markers:',len(valley_audit))
    print('Markers moved:',sum(r['shift_samples'] != 0 for r in valley_audit))
    print('Local-edge markers requiring review:',sum('search edge' in r['review_flag'] for r in valley_audit))

if __name__=='__main__':
    main()
