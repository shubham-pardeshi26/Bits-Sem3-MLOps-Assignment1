"""UCI Heart Disease from all four hospitals: verified download, loading and cleaning.

python -m cardiorisk.data [--force]
"""

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from cardiorisk import config

# Digests of the extracted files. Every downstream number depends on these bytes, so a changed
# upstream archive (or a hand-edited file) must stop the pipeline instead of silently shifting it.
RAW_SHA256 = {
    "processed.cleveland.data": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    "processed.hungarian.data": "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796",
    "processed.switzerland.data": "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314",  # noqa: E501
    "processed.va.data": "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f",
    "heart-disease.names": "ec7ed5aa4ed8321097808f47b836301c2a0c36f3bb09c40b8b43b622363e6b61",
}


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _unverified(raw_dir: Path) -> list[str]:
    """Pinned files that are missing from raw_dir or whose content differs from RAW_SHA256."""
    return [
        name
        for name, digest in RAW_SHA256.items()
        if not (raw_dir / name).is_file() or _sha256((raw_dir / name).read_bytes()) != digest
    ]


def download(
    raw_dir: Path = config.RAW_DIR, force: bool = False, url: str = config.UCI_ZIP_URL
) -> None:
    """Extract the four site files and the codebook from the UCI archive into raw_dir.

    No network access when every file is already present and verified. Nothing is written unless
    every file in the archive matches its pinned digest.
    """
    needed = list(RAW_SHA256) if force else _unverified(raw_dir)
    if not needed:
        return
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Could not download {url} ({exc}). data/raw can be populated manually: download the "
            f"archive elsewhere and extract {', '.join(needed)} into {raw_dir}."
        ) from exc
    members = {Path(name).name: name for name in archive.namelist()}
    blobs = {name: archive.read(members[name]) for name in RAW_SHA256 if name in members}
    bad = [name for name, digest in RAW_SHA256.items() if _sha256(blobs.get(name, b"")) != digest]
    if bad:
        raise ValueError(
            f"Checksum mismatch for {', '.join(bad)} in {url}: the published data differs from the "
            f"pinned version. Review the change before updating RAW_SHA256 in {__file__}."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in blobs.items():
        (raw_dir / name).write_bytes(blob)


def load_raw(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """The four site files stacked (no header in the files; '?' marks a missing value)."""
    frames = [
        pd.read_csv(
            raw_dir / filename, header=None, names=config.RAW_COLUMNS, na_values="?"
        ).assign(site=site)
        for site, filename in config.SITE_FILES.items()
    ]
    return pd.concat(frames, ignore_index=True)[["site", *config.RAW_COLUMNS]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules below; returns columns site, RAW_COLUMNS, target."""
    df = df[["site", *config.RAW_COLUMNS]]
    df[config.RAW_COLUMNS] = df[config.RAW_COLUMNS].astype(float)
    # Zero blood pressure or cholesterol is impossible in a living patient: those sites wrote 0 for
    # "not measured" (chol at every Switzerland row and 49 VA rows, trestbps once at VA).
    for col in ("trestbps", "chol"):
        df[col] = df[col].mask(df[col] == 0)
    # Kept on purpose: negative oldpeak (12 rows, mostly Switzerland) is ST elevation, a real
    # finding, and extreme values such as chol 603 or thalach 60 are clinically plausible.
    # Exact duplicate records (one Hungarian, one VA pair) could land on both sides of a split.
    df = df.drop_duplicates(ignore_index=True)
    # Hungary only coded num 0/1, the other sites graded 1-4; presence vs absence is comparable.
    return df.assign(**{config.TARGET: (df["num"] > 0).astype(int)})


def load_processed(path: Path = config.PROCESSED_CSV) -> pd.DataFrame:
    """The cleaned table written by main(), for EDA and training."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found: run 'make data' (python -m cardiorisk.data).")
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download, verify and clean the UCI data.")
    parser.add_argument("--force", action="store_true", help="download even if data/raw verifies")
    args = parser.parse_args(argv)

    try:
        download(force=args.force)
    except (RuntimeError, ValueError) as exc:  # expected failures: a message, not a traceback
        raise SystemExit(f"error: {exc}") from None
    raw = load_raw()
    df = clean(raw)
    config.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_CSV, index=False)

    by_site = df.groupby("site")[config.TARGET].agg(rows="size", prevalence="mean")
    missing = df[config.RAW_COLUMNS].isna().groupby(df["site"]).mean().T
    missing["all"] = df[config.RAW_COLUMNS].isna().mean()
    print(f"{config.PROCESSED_CSV}: {len(df)} rows, {len(raw) - len(df)} exact duplicates dropped")
    print(by_site.assign(prevalence=by_site.prevalence.round(3)).to_string())
    print("\nMissing share (%) by site:")
    print((100 * missing[missing["all"] > 0]).round().astype(int).to_string())


if __name__ == "__main__":
    main()
