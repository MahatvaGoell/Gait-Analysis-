"""Refine reviewed valley events locally on the original 100 Hz input grid.

Cycle identity and individually reviewed phase boundaries are preserved.
This corrects timing precision; it does not establish a physical heel strike.
"""
import numpy as np


def refine(times, values, imputed, selected, bounds, record='', trial_side=''):
    events = []
    labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'T', 'L1', 'L2', 'L3', 'L4', 'L5']
    for number, (old, label) in enumerate(zip(selected, labels)):
        a, b = bounds[0] if number < 5 else bounds[1] if number == 5 else bounds[2]
        old_index = int(np.argmin(abs(times-old)))
        # 15 samples is a local correction, never a search for another step.
        # Two individually inspected S4 troughs lie just beyond 15 samples.
        radius = {('sir_21','09r','S4'):17, ('sir_21','14r','S4'):18}.get(
            (record,trial_side,label),15)
        start, end = max(0, old_index-radius), min(len(times), old_index+radius+1)
        candidates = np.arange(start, end)
        candidates = candidates[(times[candidates] > a) & (times[candidates] < b)
                                & np.isfinite(values[candidates])]
        if not len(candidates):
            raise ValueError(f'{label}: no finite samples inside the reviewed phase')
        minimum = float(np.min(values[candidates]))
        # Equal-valued flat bottoms have no unique minimum. Use the middle
        # observed minimum sample in the contiguous run closest to the old event.
        equal = np.isclose(values[candidates], minimum, atol=1e-7, rtol=0)
        minima = candidates[equal]
        groups = np.split(minima, np.flatnonzero(np.diff(minima)>1)+1)
        chosen = min(groups, key=lambda g: (np.min(abs(g-old_index)), -len(g)))
        index = int(chosen[(len(chosen)-1)//2])
        flags = []
        if index in (int(candidates[0]), int(candidates[-1])):
            flags.append('minimum at local search edge')
        if imputed[index]:
            flags.append('estimated input at marker')
        events.append(dict(label=label, old_seconds=float(old), seconds=float(times[index]),
                           sample_index=index, shift_samples=index-old_index,
                           value=float(values[index]), equal_minimum_samples=len(chosen),
                           search_start_sample=int(candidates[0]), search_end_sample=int(candidates[-1]),
                           search_radius_samples=radius,
                           review_flag='; '.join(flags)))
        assert abs(index-old_index) <= radius
        assert values[index] <= np.min(values[candidates]) + 1e-7
    assert all(x['seconds'] < y['seconds'] for x,y in zip(events,events[1:]))
    return events
