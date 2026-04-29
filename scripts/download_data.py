"""Download the Kaggle Sports Classification dataset via kagglehub.

Usage:
    pip install kagglehub
    python scripts/download_data.py [--link_dir ./sports_dataset]

By default this prints the path returned by ``kagglehub`` and creates a
symlink (or junction on Windows) at ``./sports_dataset`` so the rest of
the pipeline (``scripts/train.py``) can find the data with its default
``--data_dir`` argument.

The dataset is roughly 1.4 GB; subsequent calls reuse the local cache.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import kagglehub


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--link_dir",
        default="./sports_dataset",
        help=(
            "Where to point a symlink/junction at the downloaded folder. "
            "Use --no_link to skip."
        ),
    )
    p.add_argument(
        "--no_link",
        action="store_true",
        help="Don't create the convenience symlink/junction.",
    )
    return p.parse_args()


def make_link(target: Path, link: Path) -> None:
    """Create a symlink (POSIX) or junction (Windows) at ``link`` -> ``target``."""
    link = link.resolve() if link.exists() else link
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_dir():
            print(f"[skip] {link} already exists.")
            return
        raise SystemExit(f"{link} exists and is not a directory/symlink.")

    if os.name == "nt":
        # Windows: use a directory junction (no admin required).
        import subprocess
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
        )
        print(f"[ok] junction {link} -> {target}")
    else:
        link.symlink_to(target, target_is_directory=True)
        print(f"[ok] symlink {link} -> {target}")


def main() -> None:
    args = parse_args()

    print("Downloading gpiosenka/sports-classification via kagglehub ...")
    path = Path(kagglehub.dataset_download("gpiosenka/sports-classification"))
    print(f"Path to dataset files: {path}")

    expected = ["train", "valid", "test"]
    missing = [d for d in expected if not (path / d).exists()]
    if missing:
        print(
            f"[warn] expected sub-folders not found at top level: {missing}",
            file=sys.stderr,
        )
        # Some Kaggle versions nest the data inside an extra folder; auto-detect.
        for child in path.iterdir():
            if child.is_dir() and (child / "train").exists():
                path = child
                print(f"[info] using nested folder: {path}")
                break

    if not args.no_link:
        make_link(path, Path(args.link_dir))


if __name__ == "__main__":
    main()
