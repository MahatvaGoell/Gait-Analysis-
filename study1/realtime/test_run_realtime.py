import unittest

import numpy as np

from study1.realtime.run_realtime import build, debounce, delays, false_switches
from study1.realtime.run_ablations import FULL_BAND, channel_sets, feature_columns


class RealtimeTest(unittest.TestCase):
    def test_switch_is_causal_and_short_excursions_are_ignored(self):
        pred = np.array(["standing", "GI", "GI", "standing", "GI", "GI", "GI"])
        times = np.arange(len(pred)) * 2
        np.testing.assert_array_equal(debounce(pred, times), ["standing"] * 6 + ["GI"])
        self.assertEqual(false_switches(pred, np.array(["standing"] * 7), times), 1)
        self.assertEqual(false_switches(np.array(["standing"] * 7), pred, times), -3)

    def test_gap_breaks_confirmation_and_does_not_count_as_a_switch(self):
        pred = np.array(["standing", "GI", "GI", "GI"])
        times = np.array([0, 2, 4, 10])
        np.testing.assert_array_equal(debounce(pred, times), ["standing", "standing", "standing", "GI"])
        self.assertEqual(false_switches(pred, np.array(["standing"] * 4), times), 0)
        info = {
            "transition_class": "short_to_long",
            "events": {"GI": 2, "transition": 20, "GT": 40},
            "event_ends": {"GI": 12, "transition": 40, "GT": 50},
        }
        self.assertTrue(np.isnan(delays(pred, times, info)["GI"]))

    def test_delay_uses_confirmation_time_and_excludes_end_boundary(self):
        pred = np.array(["GI", "GI", "GI"])
        times = np.array([2, 4, 6])
        info = {
            "transition_class": "short_to_long",
            "events": {"GI": 1, "transition": 20, "GT": 40},
            "event_ends": {"GI": 7, "transition": 40, "GT": 50},
        }
        self.assertEqual(delays(pred, times, info)["GI"], 0.05)
        info["event_ends"]["GI"] = 6
        self.assertTrue(np.isnan(delays(pred, times, info)["GI"]))

    def test_build_excludes_calibration_and_imputation_without_using_baseline(self):
        n = 400
        fmg = np.ones((n, 8))
        estimated = np.zeros_like(fmg, dtype=bool)
        estimated[30, 0] = True
        graphs = [
            dict(subject="S", record=f"walk_{i}", trial=1, fmg=fmg, estimated=estimated,
                 rise=100, valleys=[112 + 12 * k for k in range(12)], rest=260, n=n,
                 baseline=np.full(8, np.nan))
            for i in range(1, 5)
        ]
        dirs = {("S", f"walk_{i}", 1): "short_to_long" for i in range(1, 4)}
        dirs[("S", "walk_4", 1)] = "unclear"
        (scored,) = build(graphs, dirs, 20)
        self.assertEqual(scored["graph"], 2)
        self.assertFalse(((30 <= scored["t"]) & (scored["t"] < 50)).any())
        np.testing.assert_array_equal(scored["X"][:, :8], np.ones_like(scored["X"][:, :8]))

    def test_matched_windows_use_same_endpoints_without_looking_back_further_for_features(self):
        n = 400
        fmg = np.repeat(np.arange(n)[:, None], 8, axis=1).astype(float)
        estimated = np.zeros_like(fmg, dtype=bool)
        estimated[43, 7] = True
        graphs = [
            dict(subject="S", record=f"walk_{i}", trial=1, fmg=fmg, estimated=estimated,
                 rise=100, valleys=[112 + 12 * k for k in range(12)], rest=260, n=n)
            for i in range(1, 4)
        ]
        dirs = {("S", f"walk_{i}", 1): "short_to_long" for i in range(1, 4)}
        (short,) = build(graphs, dirs, 10, match_length=40)
        (long,) = build(graphs, dirs, 40, match_length=40)
        np.testing.assert_array_equal(short["t"], long["t"])
        np.testing.assert_array_equal(short["y"], long["y"])
        self.assertFalse(((43 <= short["t"]) & (short["t"] < 83)).any())
        np.testing.assert_allclose(short["X"][:, 0], short["t"] - 4.5)
        np.testing.assert_allclose(long["X"][:, 0], long["t"] - 19.5)
        self.assertEqual(short["trial_key"], ("S", "walk_3", 1))

    def test_channel_masks_select_all_six_features_for_the_same_sensors(self):
        groups = channel_sets()
        self.assertEqual(len(groups), 9)
        for region in ("anterior", "medial", "posterior", "lateral"):
            alone, without = set(groups[f"{region}_only"]), set(groups[f"without_{region}"])
            self.assertFalse(alone & without)
            self.assertEqual(alone | without, set(FULL_BAND))
        np.testing.assert_array_equal(feature_columns(groups["lateral_only"]), [6, 14, 22, 30, 38, 46])


if __name__ == "__main__":
    unittest.main()
