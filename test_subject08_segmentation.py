"""Consistency checks, not clinical validation of the provisional labels."""
import unittest
import csv
import hashlib
import json
import numpy as np
from PIL import Image
from subject08_segmentation import OUT, DATASET_ROOT, grid_values, load_peaks, segment_trial, iter_trials


class SegmentationTests(unittest.TestCase):
    def test_timestamp_gap_is_not_filled(self):
        t=np.array([0,.01,.02,.30,.31])
        grid=np.arange(0,.35,.01)
        z=grid_values(t,np.column_stack([t,t*2]),grid)
        self.assertTrue(np.isnan(z[3:30]).all())
        self.assertTrue(np.isnan(z[32:]).all())
        np.testing.assert_allclose(z[:3,0],grid[:3])

    def test_no_movement_does_not_create_six_stages(self):
        grid=np.arange(0,22,.01)
        z=np.ones((len(grid),20))*100
        phases,info,landmarks=segment_trial(grid,z)
        self.assertEqual(phases,[])
        self.assertEqual(info['status'],'unresolved')
        self.assertEqual(landmarks,[])

    def test_pressure_peak_sequence_tracks_delayed_signal(self):
        t=np.arange(0,24,.01)
        def trace(delay):
            v=np.ones(len(t))*1000
            for center in np.arange(4+delay,16+delay,1.2):
                v+=600*np.exp(-((t-center)/.18)**2)
            return v
        p0,_=load_peaks(trace(0)); p3,_=load_peaks(trace(3))
        self.assertEqual(len(p0),len(p3))
        np.testing.assert_allclose(np.array(p3)-p0,300,atol=1)

    def test_each_real_trial_has_ordered_intervals_or_explicit_abstention(self):
        count=0
        for record,trial,grid,z,_ in iter_trials():
            count+=1
            phases,info,_=segment_trial(grid,z)
            if not phases:
                self.assertEqual(info['status'],'unresolved')
                continue
            self.assertEqual([p['phase'] for p in phases],['QS','GI','SSSW','SLT','SSLW','GT'])
            self.assertEqual(phases[0]['start_index'],0)
            self.assertEqual(phases[-1]['end_index'],len(grid))
            for p in phases: self.assertLess(p['start_index'],p['end_index'])
            for a,b in zip(phases[:-1],phases[1:]): self.assertEqual(a['end_index'],b['start_index'])
        self.assertEqual(count,41)

    def test_exported_samples_and_labels_match_recomputed_sources(self):
        for record,trial,grid,z,_ in iter_trials():
            phases,_,_=segment_trial(grid,z)
            expected=np.full(len(grid),'UNRESOLVED',dtype='<U10')
            for p in phases: expected[p['start_index']:p['end_index']]=p['phase']
            for side,columns in [('l',list(range(8))+[16,17]),('r',list(range(8,16))+[18,19])]:
                path=OUT/'segmented_data'/f'Sub08_H/{record}/trial_{trial:02d}_{side}_segmented.csv'
                with path.open(newline='',encoding='utf-8') as f: rows=list(csv.reader(f))[1:]
                actual=np.array([[float(v) if v else np.nan for v in row[:11]] for row in rows])
                np.testing.assert_allclose(actual[:,0],grid,atol=1e-10)
                np.testing.assert_allclose(actual[:,1:],z[:,columns],rtol=1e-12,atol=1e-12,equal_nan=True)
                self.assertEqual([row[11] for row in rows],expected.tolist())
                with Image.open(OUT/f'Sub08_H/{record}/trial_{trial:02d}_{side}_complete.png') as im:
                    self.assertEqual(im.size,(2200,880)); im.verify()

    def test_raw_hashes_are_unchanged(self):
        provenance=json.loads((OUT/'segmentation_provenance.json').read_text())
        for relative,expected in provenance['raw_sha256'].items():
            self.assertEqual(hashlib.sha256((DATASET_ROOT/relative).read_bytes()).hexdigest(),expected)


if __name__=='__main__': unittest.main()
