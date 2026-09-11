"""Verify numeric exports and preserve Subject 08; not physiological validation."""
import csv
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
from generate_subject05 import HERE, iter_subject05, evaluate, protected_state


def verify():
    provenance=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
    assert protected_state()==provenance['protected_files_before']==provenance['protected_files_after']
    count=0; provisional=0; inferred=[]; durations=[]
    with (HERE/'complete_trial_manifest.csv').open(newline='',encoding='utf-8') as f:
        manifest=list(csv.DictReader(f))
    assert len(manifest)==80 and len({r['png'] for r in manifest})==80
    for record,trial,grid,z,quality in iter_subject05():
        phases,info,_=evaluate(grid,z,quality)
        provisional+=bool(phases); durations.append(float(grid[-1]))
        labels=np.full(len(grid),'UNRESOLVED',dtype='<U10')
        if phases:
            assert [p['phase'] for p in phases]==['QS','GI','SSSW','SLT','SSLW','GT']
            assert phases[0]['start_index']==0 and phases[-1]['end_index']==len(grid)
        for p in phases:
            assert p['start_index']<p['end_index']
            labels[p['start_index']:p['end_index']]=p['phase']
        for side,cols in [('L',list(range(8))+[16,17]),('R',list(range(8,16))+[18,19])]:
            row=next(r for r in manifest if r['record']==record and int(r['trial'])==trial and r['side']==side)
            with (HERE/row['segmented_csv']).open(newline='',encoding='utf-8') as f:
                samples=list(csv.reader(f))[1:]
            values=np.array([[float(v) if v else np.nan for v in r[:11]] for r in samples])
            np.testing.assert_allclose(values[:,0],grid,atol=1e-10)
            np.testing.assert_allclose(values[:,1:],z[:,cols],rtol=1e-12,atol=1e-12,equal_nan=True)
            assert [r[11] for r in samples]==labels.tolist()
            assert [int(r[13]) for r in samples]==(~np.all(np.isfinite(z[:,cols]),axis=1)).astype(int).tolist()
            with Image.open(HERE/row['png']) as im:
                assert im.size==(2200,880); im.verify()
            if quality[side]['trigger_boundary_status']!='observed':
                inferred.append(f'{record}/trial_{trial:02d}/{side}')
            count+=1
    assert count==80 and inferred==['sub_21/trial_16/L']
    assert protected_state()==provenance['protected_files_before']
    result=dict(verified_pngs=count,verified_numeric_exports=count,provisional_trials=provisional,
                inferred_end_trigger_intervals=inferred,min_duration_seconds=min(durations),max_duration_seconds=max(durations),
                subject08_and_existing_tempgraphs_unchanged=True,raw_subject05_unchanged=True,
                scope='Numerical consistency and file preservation; not validation of gait-event labels')
    (HERE/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': verify()
