"""Source-preserving, trial-local graph refinement for Sub01_A / Sub02_A.

Amputee recordings are evaluated independently; no healthy-subject event
times, amplitudes or imposed bilateral symmetry are used. CoP valley/peak
events remain candidates, not validated foot contacts or clinical labels.
The 5-1-5 phase names express the requested protocol ordering only.
"""
from pathlib import Path
import json
import csv
import shutil
import tempfile
import time
import numpy as np

import refine_subject03_04 as engine

SUBJECTS = ("Sub01_A", "Sub02_A")
old = engine.old
for subject in SUBJECTS:
    root = Path(__file__).resolve().parent / "tempgraphs" / subject.lower()
    old.SUBJECTS[subject] = (root / "segmented_data", root)
engine.SUBJECTS = SUBJECTS

clean_signals = engine.clean_signals
cadence = engine.cadence
load_train = engine.load_train
find_stances = engine.find_stances
_base_segment = engine.segment
_base_draw = engine.draw
REVIEW_FILE = Path(__file__).with_name('amputee_edge_reviews.json')
# Reviewed sustained departures in the edge panels. Early isolated standing
# shifts/pulses must not set GI simply because they precede a walking train.
# These indices apply only with the corresponding source-hash-locked review.
REVIEWED_ONSETS = {
    ('Sub01_A', 'jitu_1', 8, 'L'): 253,
    ('Sub02_A', 'vipin_1', 1, 'R'): 340,
    ('Sub02_A', 'vipin_1', 4, 'L'): 270,
    ('Sub02_A', 'vipin_21', 3, 'R'): 170,
    ('Sub02_A', 'vipin_21', 6, 'R'): 170,
    ('Sub02_A', 'vipin_21', 7, 'L'): 170,
    ('Sub02_A', 'vipin_21', 7, 'R'): 170,
}


def terminal_cop_recovery(z, stances, stride):
    """Use an observed terminal CoP floor recovery, never a predicted index."""
    v, p, _ = stances[-1]
    s = old.smooth(engine.median_filter(z[:, 8], size=3, mode='nearest'), 3)
    a = max(p + 4, int(v + .60 * stride))
    b = min(len(z) - 12, int(v + 1.55 * stride))
    if b <= a:
        return None
    low = a + int(np.argmin(s[a:b]))
    climb = float(s[p] - s[v])
    end = low
    while end + 1 < b and s[end + 1] <= s[low] + .06 * climb:
        end += 1
    if end + 1 >= b or s[min(b - 1, end + 6)] - s[low] < .025 * climb:
        return None
    later = s[min(len(s) - 1, end + int(.4 * stride)):]
    if len(later) < int(.25 * stride) or np.ptp(later) > .60 * climb:
        return None
    return int(end), 'terminal_cop_floor_recovery_candidate'


def segment(subject, record, trial, left, right, fl, fr):
    result = _base_segment(subject, record, trial, left, right, fl, fr)
    reviews = json.loads(REVIEW_FILE.read_text(encoding='utf-8'))['reviews']
    for side in ('L', 'R'):
        review = next((r for r in reviews if
                       (r['subject'], r['record'], r['trial'], r['side']) ==
                       (subject, record, trial, side)), None)
        if review is None:
            continue
        seg = result[side]
        if len(seg['stances']) < 4:
            continue  # a quiet/invalid input must never acquire reviewed steps
        path = old.SUBJECTS[subject][0] / record / f'trial_{trial:02d}_{side.lower()}_segmented.csv'
        if engine.sha256(path) != review['source_sha256']:
            raise ValueError(f'Source changed; trial-local review must be repeated: {path}')
        ss = list(seg['stances'])
        ss += [(v, p, 'individually_reviewed_cop_edge') for v, p in review['add_pairs']]
        ss.sort()
        assert len({v for v, _, _ in ss}) == len(ss)
        seg['stances'] = ss
        seg['individual_review'] = review['evidence']
        first = min(ss[0][0], result['L']['stances'][0][0], result['R']['stances'][0][0])
        rise, supported = engine.local_onset(result['L']['display_values'], result['R']['display_values'], first, seg['stride_samples'])
        rise = REVIEWED_ONSETS.get((subject, record, trial, side), rise)
        seg['reviewed_onset_sample'] = rise
        seg['review_notes'] = ['Trial-local edge correction; signal-derived candidates, not validated contacts']
        if not supported:
            seg['review_notes'].append('GI onset uncertain: independent foot-lift evidence needed')
        end = (ss[11][0], 'individually_reviewed_cop_edge') if len(ss) > 11 else None
        if len(ss) == 11:
            end = terminal_cop_recovery(seg['display_values'], ss, seg['stride_samples'])
            if end is None:
                end = engine.final_contact(seg['display_values'], ss, seg['stride_samples'])
                if end is not None:
                    seg['review_notes'].append('Final cut uses loading support; CoP terminal recovery is indistinct')
        lay = old.layout(ss, end, rise, len(left))
        if lay:
            assert not old.check(lay)
            seg.update(layout=lay, reason='', extra_steps=[], GT_start_sample=lay[0][5])
        else:
            seg.update(layout=None, reason='Terminal contact remains uncertain; earlier supported cuts retained', extra_steps=[])
            seg['partial_rise'] = min(rise, ss[0][0] - old.MIN_GI)
    return result


def partial_intervals(seg, n):
    ss = seg['stances']; rise = seg['partial_rise']
    bounds = [0, rise, ss[0][0], ss[5][0], ss[6][0], n]
    return list(zip(('QS', 'GI', 'SSSW', 'SLT', 'SSLW_CANDIDATE'), bounds[:-1], bounds[1:]))


def save_png_atomic(im, path):
    """Complete the PNG before replacing a Windows-viewer-accessed output."""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.png', delete=False) as f:
        pending = Path(f.name)
    try:
        im.save(pending, format='PNG')
        for attempt in range(4):
            try:
                pending.replace(path)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(.25)
    finally:
        pending.unlink(missing_ok=True)


def draw(path, title, times, seg, flags):
    if seg['layout'] or 'partial_rise' not in seg:
        return _base_draw(path, title, times, seg, flags)
    # Keep the complete measured timeline and all traces. An uncertain final
    # contact must not erase the earlier supported phase boundaries.
    from PIL import Image, ImageDraw
    import math
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.png', delete=False) as f:
        base_path = Path(f.name)
    try:
        _base_draw(base_path, title, times, seg, flags)
        with Image.open(base_path) as source:
            im = source.copy()
    finally:
        base_path.unlink(missing_ok=True)
    d = ImageDraw.Draw(im); left, right, top, bottom = 138, 2015, 190, 620
    xmax = max(2200, math.ceil(len(times) / 200) * 200)
    xp = lambda x: left + x / xmax * (right - left)
    d.rectangle((left, 78, right, 185), fill='white')
    for name, a, b in partial_intervals(seg, len(times)):
        text = 'SSLW* (end unconfirmed; GT boundary unknown)' if name == 'SSLW_CANDIDATE' else old.PHASE_TITLES[name] + '*'
        centre = (xp(a) + xp(b - 1)) / 2
        box = d.textbbox((0, 0), text, font=old.font(19, True))
        yy = 86 if name in ('GI', 'SLT') else 151
        d.text((centre - box[2] / 2, yy), text, font=old.font(19, True), fill='#955000' if name=='SSLW_CANDIDATE' else '#111111')
        if a:
            x = xp(a)
            for y in range(top + 6, bottom, 15):
                d.line((x, y, x, min(y + 8, bottom)), fill='#333333', width=2)
            d.polygon([(x, top - 7), (x - 5, top + 4), (x + 5, top + 4)], fill='#333333')
    for k, (v, p, _) in enumerate(seg['stances'][:11]):
        phase, num = old.STEP_PLAN[k]
        text = old.step_name(phase, num) + ('?' if k == 10 else '')
        d.text((xp(p), 215), text, font=old.font(13), fill='#666666')
    d.rectangle((left, 704, 2199, 790), fill='white')
    d.text((left, 704), 'PARTIAL: QS/GI/SSSW/SLT cuts retained. Last SSLW interval has no confirmed closing contact; no GT cut invented.', font=old.font(16, True), fill='#955000')
    d.text((left, 734), '* 5 short intervals / 1 transition / 4 closed long intervals + one unclosed climb. L5? is not counted as a complete step.', font=old.font(16), fill='#66502d')
    d.text((left, 762), 'SSLW_CANDIDATE includes the unassigned tail, not a verified SSLW end. Source measurements unchanged; not training labels.', font=old.font(16), fill='#555555')
    save_png_atomic(im, path)


def augment_partial_outputs(subject, summary):
    source, root = old.SUBJECTS[subject]
    read = lambda p: list(csv.DictReader(p.open(encoding='utf-8', newline='')))
    cuts = read(root/'phase_cuts.csv'); intervals = read(root/'phase_intervals.csv')
    stances = read(root/'stance_segments.csv')
    audit = json.loads((root/'preprocessing_audit.json').read_text(encoding='utf-8'))
    review_map = {(r['record'], r['trial'], r['side']): r for r in json.loads(REVIEW_FILE.read_text(encoding='utf-8'))['reviews'] if r['subject']==subject}
    partial = 0
    for row in cuts:
        record, trial, side = row['record'], int(row['trial']), row['side']
        review = review_map.get((record, trial, side))
        if review:
            row['individual_review_evidence'] = review['evidence']
        if row['status'] != 'unresolved':
            continue
        lp = source/record/f'trial_{trial:02d}_l_segmented.csv'
        rp = lp.with_name(lp.name.replace('_l_', '_r_'))
        t, l, fl = old.parse_csv(lp); tr, r, fr = old.parse_csv(rp)
        seg = segment(subject, record, trial, l, r, fl, fr)[side]
        if 'partial_rise' not in seg:
            continue
        partial += 1; flags = fl if side == 'L' else fr
        path = lp if side == 'L' else rp
        raw = read(path); key = dict(record=record, trial=trial, side=side)
        row.update(status='partial_cop_cut', review_required=1, GT_start_sample='',
                   closed_short_intervals=5, transition_intervals=1, closed_long_intervals=4,
                   unclosed_long_climbs=1, review_notes=seg['reason'])
        for name, a, b in partial_intervals(seg, len(t)):
            row[name+'_start_sample'] = a; row[name+'_start_s'] = a/100
            intervals.append({**key, 'phase':name, 'start_sample':a, 'end_sample_exclusive':b,
                              'start_seconds':a/100, 'end_seconds_exclusive':b/100,
                              'boundary_status':'recording_end_not_phase_end' if name=='SSLW_CANDIDATE' else 'candidate'})
        for k, (v, p, kind) in enumerate(seg['stances'][:11]):
            phase, num = old.STEP_PLAN[k]; name = old.step_name(phase, num)
            nxt = seg['stances'][k+1][0] if k < 10 else ''
            filename = f'trial_{trial:02d}_{side.lower()}_{phase}_{name}.csv'
            target = root/'stance_segments'/record/filename
            target.parent.mkdir(parents=True, exist_ok=True)
            chunk = []
            for i in range(v, p+1):
                chunk.append({'time_seconds':raw[i]['time_seconds'], 'sample':i, 'time_from_valley_seconds':f'{(i-v)/100:.2f}',
                    **{col:raw[i][col] for col in old.VALUE_COLUMNS}, 'FMG_estimated':int(flags[i,:8].any()),
                    'CoP_estimated':int(flags[i,8]), 'vGRF_estimated':int(flags[i,9]),
                    'spike_repaired_for_detection':int(seg['repaired'][i].any())})
            old.write_dicts(target, chunk); f = flags[v:p+1]
            stances.append({**key, 'phase':phase, 'step':name, 'cop_valley_sample':v, 'cop_peak_sample':p,
                'next_valley_sample':nxt, 'cop_valley_s':v/100, 'cop_peak_s':p/100, 'next_valley_s':nxt/100 if nxt else '',
                'samples':p-v+1, 'valley_kind':kind, 'cop_estimated_samples':int(f[:,8].sum()),
                'fmg_estimated_samples':int(f[:,:8].any(1).sum()), 'vgrf_estimated_samples':int(f[:,9].sum()),
                'fully_measured':int(not f.any()), 'values':'original_export_not_smoothed',
                'interval_status':'unclosed' if k==10 else 'candidate_complete',
                'file':f'stance_segments/{record}/{filename}'})
    for row in audit['graphs']:
        match = next(c for c in cuts if (c['record'],int(c['trial']),c['side'])==(row['record'],row['trial'],row['side']))
        row['status'] = match['status']
        row['individual_review_evidence'] = match.get('individual_review_evidence','')
    for name, rows in [('phase_cuts.csv',cuts), ('phase_intervals.csv',intervals), ('stance_segments.csv',stances)]:
        old.write_dicts(root/name, rows)
    (root/'preprocessing_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    summary.update(partially_segmented_graphs=partial, graphs_without_any_segmentation=sum(r['status']=='unresolved' for r in cuts),
                   saved_stance_candidates=len(stances), individually_corrected_graphs=len(review_map))
    (root/'cop_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (root/'README_cop_segmentation.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## Individual edge review correction\n\n{len(review_map)} trial/limb edge reviews are stored in amputee_edge_reviews.json, locked to source CSV SHA256. '
                f'{partial} graphs retain supported early phase cuts but have an unconfirmed final contact. '
                'Their SSLW_CANDIDATE range includes the unassigned tail, NOT a verified long-walking interval. '
                'Their last saved valley-to-peak climb has interval_status=unclosed and a blank next_valley_sample. '
                'No GT boundary or fifth complete long interval is manufactured. These cases are not training labels.\n')
    return summary


def refresh_review_sheets(subject, backup):
    """Refresh existing overview pages; do not leave stale unsegmented views."""
    from PIL import Image, ImageDraw
    _, root = old.SUBJECTS[subject]
    graphs = sorted(root.rglob('*_complete.png'))
    count = 0
    for folder in ('review_sheets', 'five_step_review_sheets'):
        targets = sorted((root/folder).glob('page_*.png'))
        if not targets:
            continue
        for target, indices in zip(targets, np.array_split(np.arange(len(graphs)), len(targets))):
            saved = backup/folder/target.name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
            sheet = Image.new('RGB', (2200, 80+800*len(indices)), 'white')
            ImageDraw.Draw(sheet).text((138,15),f'{subject} | Current CoP candidate segmentation | {folder} | {target.stem}',font=old.font(24,True),fill='black')
            for j, k in enumerate(indices):
                with Image.open(graphs[k]) as im:
                    sheet.paste(im, (0,80+j*800))
            # A viewer may momentarily hold an existing PNG open on Windows.
            # Write to a separate file, then replace atomically; never truncate
            # an existing overview before a complete replacement is ready.
            save_png_atomic(sheet, target)
            count += 1
    return count


def process(subject, backup_root):
    if subject not in SUBJECTS:
        raise ValueError("Only Sub01_A and Sub02_A are in this run's scope")
    source, root = old.SUBJECTS[subject]
    if not source.is_dir() or not (root / "recording_windows.csv").is_file():
        raise FileNotFoundError(f"Missing source exports or trial windows for {subject}")
    previous_segment, previous_draw = engine.segment, engine.draw
    engine.segment, engine.draw = segment, draw
    try:
        backup = Path(backup_root).resolve()/subject
        # The shared writer backs up all other current derived files itself.
        audit_path = root/'preprocessing_audit.json'
        audit_bytes = audit_path.read_bytes() if audit_path.exists() else None
        summary = engine.process(subject, backup_root)
        if audit_bytes is not None:
            (backup/'preprocessing_audit.json').write_bytes(audit_bytes)
        summary = augment_partial_outputs(subject, summary)
        summary['existing_overview_sheets_refreshed'] = refresh_review_sheets(subject, backup)
        (root/'cop_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        with (root/'README_cop_segmentation.md').open('a',encoding='utf-8') as f:
            f.write('\nBoth existing review_sheets and five_step_review_sheets have been refreshed from the current complete graphs; their previous pages are backed up.\n')
        print(json.dumps(summary))
        return summary
    finally:
        engine.segment, engine.draw = previous_segment, previous_draw


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", nargs="+", choices=SUBJECTS)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args()
    for subject in args.subjects:
        process(subject, args.backup_root)
