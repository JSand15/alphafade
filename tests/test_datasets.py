"""Tests for the Ken French loaders.

The fixtures in tests/data/ are the real French files cut down to a few rows of each table
(preamble, header, first rows, blank line, annual rows, copyright), so the parser runs against
the real format without touching the network.
"""

from __future__ import annotations

import io
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphafade import AlphaFadeError, DataDroppedWarning, FrequencyError, InputError, datasets
from alphafade.datasets import DownloadError, load_ff3, load_momentum

DATA = Path(__file__).parent / "data"
FF3_MONTHLY_ZIP = DATA / "ff3_monthly.zip"
FF3_WEEKLY_CSV = DATA / "ff3_weekly.csv"
FF3_DAILY_CSV = DATA / "ff3_daily.csv"
MOM_MONTHLY_CSV = DATA / "mom_monthly.csv"
MOM_DAILY_ZIP = DATA / "mom_daily.zip"
COLUMNS = ["Mkt-RF", "SMB", "HML", "RF"]


def _csv_text(path: Path) -> str:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            return zf.read(zf.namelist()[0]).decode()
    return path.read_bytes().decode()


def _write(tmp_path: Path, text: str, name: str = "file.csv") -> Path:
    out = tmp_path / name
    out.write_bytes(text.encode())
    return out


class _Downloads:
    """Stand-in for datasets._download that serves fixture bytes and counts calls."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.payload


# --- parsing the real format ---------------------------------------------------------------


def test_ff3_monthly_from_zip() -> None:
    ff3 = load_ff3("M", path=FF3_MONTHLY_ZIP)
    assert list(ff3.columns) == COLUMNS
    assert (ff3.dtypes == "float64").all()
    assert isinstance(ff3.index, pd.DatetimeIndex)
    assert ff3.index.name == "date"
    assert ff3.index.is_monotonic_increasing
    # Month-end dates: 192607 -> 1926-07-31.
    assert ff3.index[0] == pd.Timestamp("1926-07-31")
    assert ff3.index.is_month_end.all()
    # Percent -> decimals: the file says 2.89, -2.42, -2.75, 0.22.
    np.testing.assert_allclose(ff3.iloc[0].to_numpy(), [0.0289, -0.0242, -0.0275, 0.0022])
    np.testing.assert_allclose(ff3.loc["1926-12-31", "RF"], 0.0028)


def test_ff3_monthly_excludes_annual_table() -> None:
    ff3 = load_ff3("M", path=FF3_MONTHLY_ZIP)
    # The fixture has 6 monthly rows (1926-07..12) and then annual rows for 1927 and 1928.
    assert len(ff3) == 6
    assert ff3.index[-1] == pd.Timestamp("1926-12-31")
    assert not (ff3.index.year >= 1927).any()
    # The annual 1927 Mkt-RF (29.44%) must not have leaked in anywhere.
    assert not np.isclose(ff3.to_numpy(), 0.2944).any()


def test_ff3_weekly_keeps_exact_dates() -> None:
    ff3 = load_ff3("W", path=FF3_WEEKLY_CSV)
    assert list(ff3.columns) == COLUMNS
    expected = pd.to_datetime(
        ["1926-07-02", "1926-07-10", "1926-07-17", "1926-07-24", "1926-07-31"]
    )
    # Weekly dates are the week's last trading day (1920s weeks often ended on Saturday).
    assert list(ff3.index) == list(expected)
    assert ff3.index.name == "date"
    np.testing.assert_allclose(ff3["Mkt-RF"].iloc[0], 0.0158)


def test_ff3_daily() -> None:
    ff3 = load_ff3("D", path=FF3_DAILY_CSV)
    assert len(ff3) == 5
    assert ff3.index[0] == pd.Timestamp("1926-07-01")
    assert ff3.index[2] == pd.Timestamp("1926-07-06")  # skips the holiday weekend
    np.testing.assert_allclose(ff3["HML"].to_numpy()[:2], [-0.0028, -0.0003])


def test_momentum_monthly_from_csv() -> None:
    umd = load_momentum("M", path=MOM_MONTHLY_CSV)
    assert isinstance(umd, pd.Series)
    assert umd.name == "UMD"
    assert umd.dtype == "float64"
    assert umd.index.name == "date"
    assert umd.index[0] == pd.Timestamp("1927-01-31")
    assert umd.index.is_month_end.all()
    # Annual rows (24.52, 26.43) are excluded.
    assert len(umd) == 6
    np.testing.assert_allclose(umd.to_numpy()[:3], [0.0057, -0.0151, 0.0352])


def test_momentum_daily_from_zip() -> None:
    umd = load_momentum("daily", path=MOM_DAILY_ZIP)
    assert umd.name == "UMD"
    assert len(umd) == 5
    assert umd.index[0] == pd.Timestamp("1926-11-03")
    np.testing.assert_allclose(umd.iloc[0], 0.0055)


def test_path_accepts_str() -> None:
    a = load_ff3("M", path=str(FF3_MONTHLY_ZIP))
    b = load_ff3("M", path=FF3_MONTHLY_ZIP)
    pd.testing.assert_frame_equal(a, b)


def test_rows_are_sorted_and_whitespace_stripped(tmp_path: Path) -> None:
    text = (
        "Some preamble, with a comma\r\n"
        "\r\n"
        "  ,  Mkt-RF ,SMB,  HML,RF  \r\n"
        " 192608 ,  2.64,  -1.44,   4.13,   0.25\r\n"
        "192607,   2.89,  -2.42,  -2.75,   0.22  \r\n"
        "\r\n"
    )
    ff3 = load_ff3("M", path=_write(tmp_path, text))
    assert list(ff3.index) == [pd.Timestamp("1926-07-31"), pd.Timestamp("1926-08-31")]
    np.testing.assert_allclose(ff3["Mkt-RF"].to_numpy(), [0.0289, 0.0264])


def test_excel_style_blank_row_ends_table(tmp_path: Path) -> None:
    text = _csv_text(FF3_WEEKLY_CSV).replace("\r\n\r\nCopyright", "\r\n,,,,\r\nCopyright")
    extra = text.replace(",,,,\r\n", ",,,,\r\n19990101, 1, 1, 1, 1\r\n")
    ff3 = load_ff3("W", path=_write(tmp_path, extra))
    assert len(ff3) == 5


def test_header_match_ignores_case_and_extra_columns(tmp_path: Path) -> None:
    text = "preamble\r\n\r\n,Other,MOM\r\n192701, 9.9, 0.57\r\n\r\n"
    umd = load_momentum("M", path=_write(tmp_path, text))
    np.testing.assert_allclose(umd.to_numpy(), [0.0057])


def test_non_utf8_preamble_is_tolerated(tmp_path: Path) -> None:
    raw = b"Caf\xe9 preamble\r\n,Mom\r\n192701, 0.57\r\n\r\n"
    path = tmp_path / "latin1.csv"
    path.write_bytes(raw)
    np.testing.assert_allclose(load_momentum("M", path=path).to_numpy(), [0.0057])


# --- missing-data codes --------------------------------------------------------------------


def test_missing_codes_become_nan_with_warning(tmp_path: Path) -> None:
    text = _csv_text(FF3_MONTHLY_ZIP)
    text = text.replace("192608,   2.64,", "192608, -99.99,")
    text = text.replace("192610,  -3.27,  -0.18,", "192610,  -3.27,  -999,")
    with pytest.warns(DataDroppedWarning, match="2 missing-data code"):
        ff3 = load_ff3("M", path=_write(tmp_path, text))
    assert np.isnan(ff3.loc["1926-08-31", "Mkt-RF"])
    assert np.isnan(ff3.loc["1926-10-31", "SMB"])
    assert int(ff3.isna().sum().sum()) == 2
    np.testing.assert_allclose(ff3.loc["1926-07-31", "Mkt-RF"], 0.0289)


def test_missing_code_in_momentum(tmp_path: Path) -> None:
    text = _csv_text(MOM_MONTHLY_CSV).replace("192702,  -1.51", "192702, -99.99")
    with pytest.warns(DataDroppedWarning, match="1 missing-data code"):
        umd = load_momentum("M", path=_write(tmp_path, text))
    assert np.isnan(umd.loc["1927-02-28"])
    assert umd.notna().sum() == 5


def test_no_warning_without_missing_codes() -> None:
    # The preamble mentions "-99.99 or -999"; that text must not count as missing data.
    # (The autouse fixture turns any alphafade warning into an error.)
    load_momentum("M", path=MOM_MONTHLY_CSV)


# --- malformed files -----------------------------------------------------------------------

HEADER = "preamble\r\n\r\n,Mkt-RF,SMB,HML,RF\r\n"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("just some text\r\nwith no table\r\n", "header row"),
        (",Mkt-RF,SMB,HML\r\n192607, 1, 2, 3\r\n", "header row"),  # RF column missing
        (HEADER + "\r\nCopyright\r\n", "no data rows"),
        (HEADER + "Annual Factors\r\n", "should start with a monthly date"),
        (HEADER + "192607, 1, 2, x, 4\r\n", "numeric values"),
        (HEADER + "192607, 1, 2\r\n", "numeric values"),
        (HEADER + "192613, 1, 2, 3, 4\r\n", "invalid date"),
        (HEADER + "192607, 1, 2, 3, 4\r\n192607, 1, 2, 3, 4\r\n", "duplicate dates"),
        (HEADER + "19260701, 1, 2, 3, 4\r\n", "look like YYYYMMDD"),
    ],
)
def test_malformed_csv_raises(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(InputError, match=match):
        load_ff3("M", path=_write(tmp_path, text))


def test_monthly_file_loaded_as_daily_raises() -> None:
    with pytest.raises(InputError, match="look like YYYYMM data"):
        load_ff3("D", path=FF3_MONTHLY_ZIP)


def test_fake_zip_raises(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="not a valid zip"):
        load_ff3("M", path=_write(tmp_path, "not a zip", name="fake.zip"))


@pytest.mark.parametrize("members", [[], ["a.csv", "b.csv"]])
def test_zip_must_hold_one_csv(tmp_path: Path, members: list[str]) -> None:
    path = tmp_path / "multi.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "hi")
        for name in members:
            zf.writestr(name, _csv_text(FF3_MONTHLY_ZIP))
    with pytest.raises(InputError, match=r"exactly one \.csv"):
        load_ff3("M", path=path)


def test_damaged_zip_raises(tmp_path: Path) -> None:
    raw = bytearray(FF3_MONTHLY_ZIP.read_bytes())
    # Corrupt the compressed data but keep the directory, so it still looks like a zip.
    raw[40:80] = b"\x00" * 40
    path = tmp_path / "damaged.zip"
    path.write_bytes(bytes(raw))
    with pytest.raises(InputError, match="damaged zip"):
        load_ff3("M", path=path)


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="Couldn't read"):
        load_ff3("M", path=tmp_path / "nope.csv")


# --- frequency -----------------------------------------------------------------------------


@pytest.mark.parametrize("freq", ["Q", "A", "annual"])
def test_ff3_rejects_quarterly_and_annual(freq: str) -> None:
    with pytest.raises(FrequencyError, match="resample"):
        load_ff3(freq, path=FF3_MONTHLY_ZIP)


def test_unknown_freq_raises() -> None:
    with pytest.raises(FrequencyError, match="Unknown freq"):
        load_ff3("hourly", path=FF3_MONTHLY_ZIP)


def test_momentum_rejects_weekly() -> None:
    with pytest.raises(FrequencyError, match="doesn't publish a weekly momentum"):
        load_momentum("W", path=MOM_DAILY_ZIP)


def test_momentum_rejects_quarterly() -> None:
    with pytest.raises(FrequencyError, match="load_momentum supports"):
        load_momentum("Q", path=MOM_DAILY_ZIP)


# --- cache ---------------------------------------------------------------------------------


def test_cache_is_reused_until_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Downloads(FF3_MONTHLY_ZIP.read_bytes())
    monkeypatch.setattr(datasets, "_download", fake)

    first = load_ff3("M", cache_dir=tmp_path)
    second = load_ff3("M", cache_dir=tmp_path)
    assert fake.urls == [
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_CSV.zip"
    ]
    pd.testing.assert_frame_equal(first, second)
    cached = tmp_path / "F-F_Research_Data_Factors_CSV.zip"
    assert cached.read_bytes() == FF3_MONTHLY_ZIP.read_bytes()
    # Atomic write leaves no temp files behind.
    assert [p.name for p in tmp_path.iterdir()] == [cached.name]

    load_ff3("M", cache_dir=tmp_path, refresh=True)
    assert len(fake.urls) == 2


def test_momentum_downloads_the_right_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Downloads(MOM_DAILY_ZIP.read_bytes())
    monkeypatch.setattr(datasets, "_download", fake)
    umd = load_momentum("D", cache_dir=tmp_path)
    assert fake.urls[0].endswith("/F-F_Momentum_Factor_daily_CSV.zip")
    assert len(umd) == 5
    assert (tmp_path / "F-F_Momentum_Factor_daily_CSV.zip").is_file()


def test_env_var_sets_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_dir = tmp_path / "from_env" / "nested"
    monkeypatch.setenv("ALPHAFADE_CACHE", str(env_dir))
    fake = _Downloads(FF3_MONTHLY_ZIP.read_bytes())
    monkeypatch.setattr(datasets, "_download", fake)

    load_ff3("M")
    load_ff3("M")
    assert (env_dir / "F-F_Research_Data_Factors_CSV.zip").is_file()
    assert len(fake.urls) == 1


def test_explicit_cache_dir_beats_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHAFADE_CACHE", str(tmp_path / "env"))
    monkeypatch.setattr(datasets, "_download", _Downloads(FF3_MONTHLY_ZIP.read_bytes()))
    load_ff3("M", cache_dir=tmp_path / "explicit")
    assert (tmp_path / "explicit" / "F-F_Research_Data_Factors_CSV.zip").is_file()
    assert not (tmp_path / "env").exists()


def test_default_cache_dir_is_home_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHAFADE_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    monkeypatch.setattr(datasets, "_download", _Downloads(FF3_MONTHLY_ZIP.read_bytes()))
    load_ff3("M")
    assert (tmp_path / ".cache" / "alphafade" / "F-F_Research_Data_Factors_CSV.zip").is_file()


def test_path_never_uses_network_or_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str) -> bytes:
        raise AssertionError("network used")

    monkeypatch.setattr(datasets, "_download", boom)
    monkeypatch.setenv("ALPHAFADE_CACHE", str(tmp_path / "cache"))
    load_ff3("M", path=FF3_MONTHLY_ZIP, refresh=True)
    assert not (tmp_path / "cache").exists()


def test_damaged_cache_suggests_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "F-F_Research_Data_Factors_CSV.zip").write_bytes(b"garbage")
    monkeypatch.setattr(datasets, "_download", _Downloads(FF3_MONTHLY_ZIP.read_bytes()))
    with pytest.raises(InputError, match="refresh=True"):
        load_ff3("M", cache_dir=tmp_path)
    # refresh=True replaces the damaged copy.
    assert len(load_ff3("M", cache_dir=tmp_path, refresh=True)) == 6


def test_unwritable_cache_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder")
    monkeypatch.setattr(datasets, "_download", _Downloads(FF3_MONTHLY_ZIP.read_bytes()))
    with pytest.raises(DownloadError, match="ALPHAFADE_CACHE"):
        load_ff3("M", cache_dir=blocker / "sub")


def test_failed_rename_cleans_up_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets, "_download", _Downloads(FF3_MONTHLY_ZIP.read_bytes()))

    def failing_replace(src: str, dst: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(DownloadError, match="Permission denied"):
        load_ff3("M", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- downloading ---------------------------------------------------------------------------


def test_download_failure_is_helpful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*args: object, **kwargs: object) -> object:
        raise urllib.error.URLError("nodename nor servname provided")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", offline)
    with pytest.raises(DownloadError, match="path=") as info:
        load_ff3("W", cache_dir=tmp_path)
    message = str(info.value)
    assert "F-F_Research_Data_Factors_weekly_CSV.zip" in message
    assert "nodename nor servname provided" in message
    assert isinstance(info.value, AlphaFadeError)
    assert list(tmp_path.iterdir()) == []


def test_download_sends_request_with_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_urlopen(self: object, request: urllib.request.Request, timeout: float) -> io.BytesIO:
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["agent"] = request.get_header("User-agent")
        return io.BytesIO(MOM_DAILY_ZIP.read_bytes())

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_urlopen)
    umd = load_momentum("D", cache_dir=tmp_path)
    assert len(umd) == 5
    assert seen["url"] == datasets.BASE_URL + "F-F_Momentum_Factor_daily_CSV.zip"
    assert seen["timeout"] == 30
    assert "alphafade" in str(seen["agent"])


def test_non_zip_download_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets, "_download", _Downloads(b"<html>Service unavailable</html>"))
    with pytest.raises(DownloadError, match="isn't a zip file"):
        load_ff3("M", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- real download (deselected by default; run with `pytest -m network`) -------------------


@pytest.mark.network
def test_real_monthly_ff3_download(tmp_path: Path) -> None:
    ff3 = load_ff3("M", cache_dir=tmp_path)
    assert list(ff3.columns) == COLUMNS
    assert len(ff3) > 1000
    assert ff3.index[0] == pd.Timestamp("1926-07-31")
    assert ff3.index.is_month_end.all()
    assert ff3.index.is_monotonic_increasing
    # Decimals, not percents: monthly market excess returns stay well inside +/-100%.
    assert ff3["Mkt-RF"].abs().max() < 1.0
    assert (tmp_path / "F-F_Research_Data_Factors_CSV.zip").is_file()


def test_oversized_download_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets, "_MAX_DOWNLOAD_BYTES", 100)
    monkeypatch.setattr(
        urllib.request.OpenerDirector,
        "open",
        lambda self, request, timeout: io.BytesIO(b"x" * 1000),
    )
    with pytest.raises(DownloadError, match="refusing it"):
        load_ff3("M", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_zip_bomb_member_is_not_decompressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets, "_MAX_CSV_BYTES", 1000)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bomb.csv", "0" * 100_000)  # compresses to almost nothing
    bomb = tmp_path / "bomb.zip"
    bomb.write_bytes(buf.getvalue())
    with pytest.raises(InputError, match="refusing to decompress"):
        load_ff3("M", path=bomb)


# --- hardening: forged zip sizes, redirects, slow servers ----------------------------------


def _zip_with_forged_size(real_bytes: int, claimed: int = 1024) -> bytes:
    """A zip whose CSV really expands to ``real_bytes`` but whose headers claim ``claimed``."""
    import struct
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("x.csv", b"0" * real_bytes)
    raw = bytearray(buf.getvalue())
    raw[22:26] = struct.pack("<I", claimed)  # local header: uncompressed size
    central = raw.rfind(b"PK\x01\x02")
    raw[central + 24 : central + 28] = struct.pack("<I", claimed)
    return bytes(raw)


@pytest.mark.parametrize("claimed", [1024, 300 * 2**20], ids=["forged-size", "honest-size"])
def test_zip_bomb_is_stopped_early(claimed: int) -> None:
    import tracemalloc

    bomb = _zip_with_forged_size(300 * 2**20, claimed=claimed)
    tracemalloc.start()
    try:
        with pytest.raises(InputError, match=r"refusing|damaged"):
            datasets._extract_text(bomb, "bomb.zip", expect_zip=True)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 250 * 2**20  # stopped near the 100 MB cap, not after all 300 MB


def _serve(handler: type) -> tuple[object, int]:
    import http.server
    import threading

    server = http.server.HTTPServer(("127.0.0.1", 0), handler)  # type: ignore[arg-type]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, int(server.server_address[1])


def test_redirect_to_plain_http_is_refused() -> None:
    import http.server

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/evil.zip")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server, port = _serve(Redirect)
    try:
        with pytest.raises(DownloadError, match="non-https redirect"):
            datasets._download(f"http://127.0.0.1:{port}/file.zip")
    finally:
        server.shutdown()  # type: ignore[attr-defined]


def test_slow_download_hits_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter(range(0, 10_000, 60))  # every read "takes" 60 s
    monkeypatch.setattr(datasets.time, "monotonic", lambda: float(next(clock)))

    class Drip:
        def __enter__(self) -> Drip:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self, n: int) -> bytes:
            return b"x"

    monkeypatch.setattr(
        urllib.request.OpenerDirector, "open", lambda self, request, timeout: Drip()
    )
    with pytest.raises(DownloadError, match="no complete download"):
        datasets._download("https://example.invalid/file.zip")
