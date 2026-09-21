"""Create the empty per-phase output hierarchy for every detected subject."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "Individual Outputs"
PHASES = ("GI", "SSSW", "SLT", "SSLW", "GT")


def detected_subjects() -> list[str]:
    """Return the current dataset subject folders, preserving their names."""
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET}")
    return sorted(folder.name for folder in DATASET.iterdir() if folder.is_dir())


def create_hierarchy(subjects: list[str]) -> None:
    """Create missing folders only; existing content is left unchanged."""
    for phase in PHASES:
        (OUTPUT / phase / "Avg" / "Avg of Trials").mkdir(parents=True, exist_ok=True)
        (OUTPUT / phase / "Avg" / "Final Avg").mkdir(parents=True, exist_ok=True)
        for subject in subjects:
            (OUTPUT / phase / "Per Trial" / subject).mkdir(parents=True, exist_ok=True)


def print_tree(path: Path) -> None:
    """Print a compact directory-only tree for verification."""
    print(f"{path.name}/")

    def walk(folder: Path, prefix: str = "") -> None:
        directories = sorted((item for item in folder.iterdir() if item.is_dir()), key=lambda item: item.name.lower())
        for index, directory in enumerate(directories):
            final = index == len(directories) - 1
            connector = "`-- " if final else "|-- "
            print(f"{prefix}{connector}{directory.name}/")
            walk(directory, prefix + ("    " if final else "|   "))

    walk(path)


if __name__ == "__main__":
    subjects = detected_subjects()
    create_hierarchy(subjects)
    print(f"Detected subjects: {', '.join(subjects)}")
    print_tree(OUTPUT)
