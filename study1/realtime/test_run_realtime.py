import unittest

import numpy as np

from study1.realtime.run_realtime import build, debounce, delays, false_switches


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


if __name__ == "__main__":
    unittest.main()
