"""Verify continuous model-completed views against unchanged source grids."""
import csv
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
from generate_additional_subjects import ROOT, iter_trials, snapshot
from complete_reconstruction import complete_trial, complete_segments


def verify(subject,expected):
    out=ROOT/'tempgraphs'/subject.lower()
    provenance=json.loads((out/'provenance.json').read_text())
    assert provenance['view_mode']=='model_completed'
    assert snapshot()==provenance['protected_files']
    with (out/'complete_trial_manifest.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==expected
    lookup={(r['record'],int(r['trial']),r['side']):r for r in rows}
    trials=list(iter_trials(subject));count=0;preserved=0;estimated=0;six=0;partial=[]
    for trial in trials:
        record,number,grid,observed,*_=trial
        z,mask,reports=complete_trial(trial,trials)
        phases,info,_=complete_segments(grid,z)
        assert np.isfinite(z).all()
        assert np.array_equal(mask,~np.isfinite(observed))
        assert np.array_equal(z[~mask],observed[~mask])
        preserved+=int((~mask).sum());estimated+=int(mask.sum())
        labels=np.full(len(grid),'UNKNOWN',dtype='<U10')
        assert phases[0]['start_index']==0 and phases[-1]['end_index']==len(grid)
        for p in phases:
            assert p['start_index']<p['end_index']
            labels[p['start_index']:p['end_index']]=p['phase']
        for a,b in zip(phases[:-1],phases[1:]):assert a['end_index']==b['start_index']
        if info['complete_six_phase']:
            assert [p['phase'] for p in phases]==['QS','GI','SSSW','SLT','SSLW','GT'];six+=1
        else:partial.append(f'{record}/{number:02d}')
        for side,cols in [('L',list(range(8))+[16,17]),('R',list(range(8,16))+[18,19])]:
            r=lookup[(record,number,side)]
            with (out/r['segmented_csv']).open(newline='') as f:
                samples=list(csv.reader(f))[1:]
            values=np.array([[float(v) for v in row[:11]] for row in samples])
            np.testing.assert_allclose(values[:,0],grid,atol=1e-10)
            np.testing.assert_allclose(values[:,1:],z[:,cols],rtol=1e-12,atol=1e-12)
            assert np.array_equal(np.array([[int(v) for v in row[14:]] for row in samples]),mask[:,cols])
            assert [row[11] for row in samples]==list(labels)
            assert all(row[12]=='model_assisted_not_ground_truth' and row[13]=='0' for row in samples)
            assert r['status']==info['status']
            with Image.open(out/r['png']) as im:
                assert im.size==(2200,880);im.verify()
            count+=1
    assert count==expected and snapshot()==provenance['protected_files']
    summary=json.loads((out/'summary.json').read_text())
    assert summary['remaining_missing_signal_values']==0
    assert summary['imputed_signal_values']==estimated
    assert summary['complete_six_phase_intervals']==six
    report=dict(subject=subject,verified_pngs=count,verified_numeric_exports=count,
        remaining_missing_values=0,source_grid_values_preserved=preserved,estimated_values=estimated,
        complete_model_assisted_six_phase_intervals=six,partial_phase_intervals=partial,
        raw_data_and_other_subjects_unchanged=True,all_estimates_flagged=True,
        scope='Numerical reproducibility/coverage only. No validation of actual missing measurements or physiological phase labels.')
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    verify('Sub06_H',80)
    verify('Sub07_H',86)
