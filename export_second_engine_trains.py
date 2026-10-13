"""Export the stance trains found by the second (noise-aware) engine.

`refine_sub08_h_valleys.py` cross-checks its own CoP stances against the trains
found by `refine_subject03_04.py` and its wrappers, and falls back to them where
its own detector does not find eleven stances. This script only records what that
engine finds (candidate valley, candidate peak, kind) in
`second_engine_trains.json`; it writes no graphs, tables or labels.

The wrappers patch the shared engine when imported, so every group runs in its
own Python process.

Run:  python export_second_engine_trains.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "second_engine_trains.json"
GROUPS = ("0102", "0304", "05", "amputee")


def export_group(group: str) -> dict:
    import refine_subject03_04 as engine

    locks = json.loads((ROOT / "amputee_review_signal_hashes.json").read_text(encoding="utf-8"))["files"]

    def sha256(path):
        """Whole-file hash, except for individually reviewed amputee files.

        `amputee_edge_reviews.json` locks each review to the hash of the whole CSV as it was on a
        Windows checkout. That lock also breaks when only the phase label columns or the line ends
        change. For those files the signals are checked instead (every column except the two label
        columns); if they are unchanged the review still applies and the locked hash is returned.
        """
        relative = Path(path).resolve().relative_to(ROOT).as_posix()
        data = Path(path).read_bytes()
        if relative in locks:
            rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
            keep = [i for i, name in enumerate(rows[0]) if name not in ("provisional_phase", "label_status")]
            signals = hashlib.sha256("\n".join(",".join(row[i] for i in keep) for row in rows).encode()).hexdigest()
            return locks[relative]["reviewed_file_sha256"] if signals == locks[relative]["signal_sha256"] else "signals-changed-" + signals
        return hashlib.sha256(data).hexdigest()

    engine.sha256 = sha256   # hashlib.file_digest needs Python 3.11
    old = engine.old
    if group == "0304":
        segment, subjects = engine.segment, ("Sub03_H", "Sub04_H")
    elif group == "0102":
        import refine_subject01_02 as wrapper
        segment, subjects = wrapper.segment, wrapper.SUBJECTS
    elif group == "amputee":
        import refine_subject01_02_amputee as wrapper
        segment, subjects = wrapper.segment, wrapper.SUBJECTS
    else:
        # Sub05_H has no recording_windows.csv: every interval is an ordinary paired trial.
        engine.SUBJECTS = subjects = ("Sub05_H",)
        real_open = Path.open

        def open_with_windows(self, *args, **kwargs):
            if self.name == "recording_windows.csv" and not self.exists():
                rows = ["record,trial,boundary_method"] + [f"{r},{t},paired" for r in ("sub_1", "sub_21") for t in range(1, 100)]
                return io.StringIO("\n".join(rows) + "\n")
            return real_open(self, *args, **kwargs)

        engine.Path.open = open_with_windows
        segment = engine.segment
    trains = {}
    for subject in subjects:
        source, _ = old.SUBJECTS[subject]
        for left_path in sorted(source.rglob("trial_*_l_segmented.csv")):
            right_path = left_path.with_name(left_path.name.replace("_l_", "_r_"))
            _, left, flags_left = old.parse_csv(left_path)
            _, right, flags_right = old.parse_csv(right_path)
            record, trial = left_path.parent.name, int(left_path.name.split("_")[1])
            key = f"{subject}|{record}|{trial}"
            try:
                result = segment(subject, record, trial, left, right, flags_left, flags_right)
            except Exception as error:   # keep going: the main script falls back to its own detector
                trains[key] = {"error": repr(error)}
                continue
            entry = {"window": result.get("source_window_method", ""), "rise": int(result["rise"])}
            for side in ("L", "R"):
                seg = result[side]
                lay = seg["layout"]
                entry[side] = {
                    "stances": [[int(v), int(p), kind] for v, p, kind in seg["stances"]],
                    "resolved": bool(lay),
                    "end_cut": int(lay[0][5]) if lay else None,
                    "end_kind": lay[2] if lay else "",
                    "reviewed_onset": int(seg["reviewed_onset_sample"]) if "reviewed_onset_sample" in seg else None,
                    "individual_review": seg.get("individual_review", ""),
                }
            trains[key] = entry
    return trains


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--group":
        json.dump(export_group(sys.argv[2]), sys.stdout)
        return
    trains = {}
    for group in GROUPS:
        done = subprocess.run([sys.executable, "-B", __file__, "--group", group], capture_output=True, text=True, check=True)
        trains.update(json.loads(done.stdout))
    OUTPUT.write_text(json.dumps(trains, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    failed = sum("error" in entry for entry in trains.values())
    print(f"{len(trains)} recordings exported to {OUTPUT.name}; {failed} could not be run by the second engine")


if __name__ == "__main__":
    main()
