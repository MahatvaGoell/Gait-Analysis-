"""Verify every Subject 06/07 numeric export, PNG and protected source hash."""
import csv
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
from generate_additional_subjects import ROOT, snapshot, iter_trials, evaluate
from missing_value_refinement import refine, blocked_validation, gap_aware_change


def unit_checks():
    grid=np.arange(1000)*.01
    z=np.tile(np.arange(1000,dtype=float)[:,None],(1,20))
    original=z.copy();original[200:210]=np.nan;original[300:350]=np.nan
    original[:3]=np.nan;original[-3:]=np.nan;original[:,19]=np.nan
    fixed,mask,gaps=refine(grid,original)
    np.testing.assert_allclose(fixed[200:210,:19],z[200:210,:19])
    assert mask[200:210,:19].all() and mask.sum()==190
    assert np.isnan(fixed[300:350]).all() and np.isnan(fixed[:3]).all()
    assert np.isnan(fixed[-3:]).all() and np.isnan(fixed[:,19]).all()
    assert not blocked_validation(np.arange(10,dtype=float),10,2)['accepted']
    rng=np.random.default_rng(617)
    assert not blocked_validation(rng.normal(size=1000),10,.1)['accepted']
    features=[np.full(41,float(i>=6)) for i in range(12)]
    features[2]=None
    result=gap_aware_change(features)
    assert result is not None and result['index'] in (5,6)
    assert gap_aware_change([None]*12) is None
    print('Gap-handling unit checks passed',flush=True)


def verify(subject,expected):
    out=ROOT/'tempgraphs'/subject.lower()
    provenance=json.loads((out/'provenance.json').read_text(encoding='utf-8'))
    assert snapshot()==provenance['protected_files'],'Protected files changed'
    with (out/'complete_trial_manifest.csv').open(newline='',encoding='utf-8') as f:manifest=list(csv.DictReader(f))
    assert len(manifest)==expected and len({r['png'] for r in manifest})==expected
    indexed={(r['record'],int(r['trial']),r['side']):r for r in manifest}
    with (out/'missing_value_report.csv').open(newline='',encoding='utf-8') as f: reported_gaps=list(csv.DictReader(f))
    gap_index={(r['record'],int(r['trial']),int(r['channel_index']),float(r['start_seconds'])):r for r in reported_gaps}
    count=0; unresolved=[]; durations=[]; imputed_count=0; preserved_count=0
    for record,trial,grid,observed,quality,boundary,notes in iter_trials(subject):
        z,imputed,gaps=refine(grid,observed)
        valid=np.isfinite(observed)
        assert np.array_equal(z[valid],observed[valid])
        assert np.array_equal(imputed,~valid & np.isfinite(z))
        imputed_count+=int(imputed.sum());preserved_count+=int(valid.sum())
        for gap in gaps:
            logged=gap_index[(record,trial,int(gap['channel_index']),gap['start_seconds'])]
            assert logged['method']==gap['method'] and int(logged['missing_samples'])==gap['missing_samples']
            if gap['method']=='linear_interpolation':
                assert gap['endpoint_span_seconds']<=.50+1e-9
                assert gap['holdout_blocks']>=20 and gap['holdout_p90_normalized_rmse']<=.05
                assert gap['holdout_max_normalized_rmse']<=.10
        phases,info,_=evaluate(grid,z,notes,imputed); durations.append(float(grid[-1]))
        labels=np.full(len(grid),'UNRESOLVED',dtype='<U10')
        if phases:
            assert [p['phase'] for p in phases]==['QS','GI','SSSW','SLT','SSLW','GT']
            assert phases[0]['start_index']==0 and phases[-1]['end_index']==len(grid)
            for a,b in zip(phases[:-1],phases[1:]): assert a['end_index']==b['start_index']
        else:unresolved.append(f'{record}/{trial:02d}')
        for p in phases:
            assert p['start_index']<p['end_index'];labels[p['start_index']:p['end_index']]=p['phase']
        for side,cols in [('L',list(range(8))+[16,17]),('R',list(range(8,16))+[18,19])]:
            row=indexed[(record,trial,side)]
            with (out/row['segmented_csv']).open(newline='',encoding='utf-8') as f: samples=list(csv.reader(f))[1:]
            values=np.array([[float(v) if v else np.nan for v in r[:11]] for r in samples])
            np.testing.assert_allclose(values[:,0],grid,atol=1e-10)
            np.testing.assert_allclose(values[:,1:],z[:,cols],rtol=1e-12,atol=1e-12,equal_nan=True)
            assert [r[11] for r in samples]==labels.tolist()
            assert [int(r[13]) for r in samples]==(~np.all(np.isfinite(z[:,cols]),axis=1)).astype(int).tolist()
            assert np.array_equal(np.array([[int(v) for v in r[14:]] for r in samples]),imputed[:,cols])
            assert row['status']==info['status']
            with Image.open(out/row['png']) as im:assert im.size==(2200,880);im.verify()
            count+=1
    assert count==expected and snapshot()==provenance['protected_files']
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert summary['graphs']==count and summary['imputed_signal_values']==imputed_count
    assert summary['provisional_intervals']==len(durations)-len(unresolved)
    report=dict(subject=subject,verified_pngs=count,verified_numeric_exports=count,
                unresolved_intervals=unresolved,duration_range_seconds=[min(durations),max(durations)],
                earlier_subject_files_and_all_raw_data_unchanged=True,
                original_finite_grid_values_preserved=preserved_count,
                verified_imputed_values=imputed_count,all_imputed_values_flagged=True,
                validation_scope='Numeric/file consistency, not physiological validation of estimated labels')
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    unit_checks()
    verify('Sub06_H',80)
    verify('Sub07_H',86)
