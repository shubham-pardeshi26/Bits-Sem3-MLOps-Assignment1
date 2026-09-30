"""Loading, cleaning and the verified download of the UCI site files, all offline and synthetic."""

import hashlib
import zipfile

import numpy as np
import pytest
from conftest import make_patients

from cardiorisk import config, data, eda

# Raw UCI format per site: no header, '?' = missing, 0 = unmeasured chol/trestbps, num 0-4.
ROWS = {
    "cleveland": [
        "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0",
        "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2",
    ],
    "hungarian": [
        "28,1,2,130,132,0,2,185,0,0,?,?,?,0",
        "29,1,2,140,?,0,0,170,0,0,?,?,?,1",
        "29,1,2,140,?,0,0,170,0,0,?,?,?,1",  # exact duplicate of the row above
    ],
    "switzerland": [
        "32,1,1,95,0,?,0,127,0,.7,1,?,?,1",
        "35,1,4,?,0,?,0,130,1,-1.1,?,?,7,3",
    ],
    "va": [
        "63,1,4,140,260,0,1,112,1,3,2,?,?,4",
        "55,1,3,0,0,0,0,155,0,1.5,2,?,?,0",
    ],
}
FILES = {config.SITE_FILES[site]: ("\n".join(rows) + "\n").encode() for site, rows in ROWS.items()}
FILES["heart-disease.names"] = b"UCI Heart Disease codebook\n"


@pytest.fixture
def pinned(monkeypatch) -> dict[str, bytes]:
    """The synthetic files, pinned in RAW_SHA256 as if they were the published ones."""
    digests = {name: hashlib.sha256(blob).hexdigest() for name, blob in FILES.items()}
    monkeypatch.setattr(data, "RAW_SHA256", digests)
    return FILES


def write(directory, files: dict[str, bytes]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (directory / name).write_bytes(blob)


def archive(path, files: dict[str, bytes]) -> str:
    """A zip like the UCI one (members inside a folder, to exercise basename matching)."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, blob in files.items():
            zf.writestr(f"heart+disease/{name}", blob)
    return path.as_uri()


def test_load_raw_stacks_the_sites_and_reads_question_marks_as_missing(tmp_path):
    write(tmp_path, FILES)
    df = data.load_raw(tmp_path)

    assert list(df.columns) == ["site", *config.RAW_COLUMNS]
    assert df["site"].value_counts().to_dict() == {s: len(r) for s, r in ROWS.items()}
    hungarian = df[df["site"] == "hungarian"]
    assert hungarian[["slope", "ca", "thal"]].isna().all().all()
    assert df.loc[df["site"] == "cleveland", "ca"].notna().all()


def test_clean_applies_every_rule(tmp_path):
    write(tmp_path, FILES)
    raw = data.load_raw(tmp_path)
    df = data.clean(raw)

    assert list(df.columns) == ["site", *config.RAW_COLUMNS, config.TARGET]
    assert len(df) == len(raw) - 1  # the exact duplicate
    assert set(df.dtypes[config.RAW_COLUMNS]) == {np.dtype(float)}
    # Zeros used as "not measured" become missing; real measurements are untouched.
    assert df.loc[df["site"] == "switzerland", "chol"].isna().all()
    va_zero = df[(df["site"] == "va") & (df["age"] == 55)]
    assert va_zero[["trestbps", "chol"]].isna().all(axis=None)
    assert df.loc[df["site"] == "cleveland", "chol"].tolist() == [233.0, 286.0]
    assert -1.1 in df["oldpeak"].tolist()  # ST elevation is a finding, not an error
    # Any severity 1-4 is disease.
    assert df.groupby("num")[config.TARGET].unique().map(list).to_dict() == {
        0.0: [0],
        1.0: [1],
        2.0: [1],
        3.0: [1],
        4.0: [1],
    }


def test_download_skips_the_network_when_files_verify(tmp_path, pinned):
    write(tmp_path, pinned)
    data.download(tmp_path, url=(tmp_path / "unreachable.zip").as_uri())  # would raise if fetched


def test_download_extracts_a_verified_archive(tmp_path, pinned):
    raw_dir = tmp_path / "raw"
    data.download(raw_dir, url=archive(tmp_path / "uci.zip", pinned))
    assert {path.name: path.read_bytes() for path in raw_dir.iterdir()} == pinned


def test_download_rejects_a_checksum_mismatch_and_writes_nothing(tmp_path, pinned):
    tampered = pinned | {"processed.va.data": pinned["processed.va.data"] + b"1,1,1\n"}
    raw_dir = tmp_path / "raw"
    with pytest.raises(ValueError, match=r"Checksum mismatch for processed\.va\.data"):
        data.download(raw_dir, url=archive(tmp_path / "uci.zip", tampered))
    assert not raw_dir.exists()


def test_download_failure_names_the_url_and_the_manual_route(tmp_path, pinned):
    url = (tmp_path / "offline.zip").as_uri()
    with pytest.raises(RuntimeError, match="offline.zip") as error:
        data.download(tmp_path / "raw", url=url)
    assert "populated manually" in str(error.value)


def test_every_eda_figure_renders_from_a_cleaned_table(tmp_path):
    df = make_patients().assign(site=lambda d: np.resize(list(config.SITE_FILES), len(d)))
    for name, plot in eda.FIGURES.items():
        path = tmp_path / f"eda_{name}.png"
        plot(df).savefig(path, **eda.SAVE)
        assert path.stat().st_size > 10_000, name
