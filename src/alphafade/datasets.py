"""Optional loaders for Kenneth French's data library (Fama-French factors and momentum).

Nothing here touches the internet when alphafade is imported. A file is downloaded only when
you call :func:`load_ff3` or :func:`load_momentum`, and it's then kept in a local cache
folder so later calls work offline. The cache folder is ``$ALPHAFADE_CACHE`` if that
environment variable is set, otherwise ``~/.cache/alphafade/``. You can also skip the
network entirely by downloading the file yourself and passing ``path=``.

French publishes returns in **percent**. These loaders divide by 100, so every value comes
back as a decimal (0.0289 means +2.89%).
"""

from __future__ import annotations

import contextlib
import http.client
import io
import os
import tempfile
import time
import warnings
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

from ._errors import DataDroppedWarning, DownloadError, FrequencyError, InputError
from ._validate import parse_freq

__all__ = ["DownloadError", "load_ff3", "load_momentum"]

BASE_URL: Final = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

_FF3_FILES: Final[dict[str, str]] = {
    "M": "F-F_Research_Data_Factors_CSV.zip",
    "W": "F-F_Research_Data_Factors_weekly_CSV.zip",
    "D": "F-F_Research_Data_Factors_daily_CSV.zip",
}
_MOM_FILES: Final[dict[str, str]] = {
    "M": "F-F_Momentum_Factor_CSV.zip",
    "D": "F-F_Momentum_Factor_daily_CSV.zip",
}
FF3_COLUMNS: Final = ("Mkt-RF", "SMB", "HML", "RF")
_MOM_COLUMN: Final = "Mom"

_TIMEOUT_SECONDS: Final = 30.0
_DEADLINE_SECONDS: Final = 120.0  # total budget for one download, however slowly it drips
# French's largest file is a few MB. These caps stop a bad download or a tampered cache
# file (e.g. a zip bomb) from exhausting memory.
_MAX_DOWNLOAD_BYTES: Final = 20 * 1024 * 1024
_MAX_CSV_BYTES: Final = 100 * 1024 * 1024
_USER_AGENT: Final = "alphafade (+https://github.com/JSand15/alphafade)"
# French marks missing observations with these codes (in percent, before dividing by 100).
_MISSING_CODES: Final = (-99.99, -999.0)
_DATE_DIGITS: Final[dict[str, int]] = {"M": 6, "W": 8, "D": 8}
_FREQ_NAMES: Final[dict[str, str]] = {"M": "monthly", "W": "weekly", "D": "daily"}

_Loadable = Literal["M", "W", "D"]


def load_ff3(
    freq: str = "M",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Load the Fama-French three factors plus the risk-free rate, as decimals.

    The three factors are the market excess return (Mkt-RF: stock market minus T-bills),
    size (SMB: small minus big stocks) and value (HML: high minus low book-to-market).
    RF is the one-month T-bill return. Only the main table is read; the annual table that
    French appends to the monthly file is ignored.

    Parameters
    ----------
    freq : {"M", "W", "D"}, default "M"
        Monthly, weekly, or daily. Quarterly and annual aren't offered: resample the monthly
        data instead, e.g. ``(1 + ff3).resample("QE").prod() - 1``.
    cache_dir : str or path-like, optional
        Folder for the downloaded zip. Defaults to ``$ALPHAFADE_CACHE`` if set, otherwise
        ``~/.cache/alphafade/``.
    refresh : bool, default False
        Download again even if a cached copy exists (French updates the files monthly).
    path : str or path-like, optional
        A local ``.zip`` or ``.csv`` file in French's format. When given, no network or
        cache is used at all, and ``cache_dir``/``refresh`` are ignored.

    Returns
    -------
    DataFrame
        Columns ``["Mkt-RF", "SMB", "HML", "RF"]`` (float64, decimals), indexed by a sorted
        DatetimeIndex named ``"date"``. Monthly rows are dated at month-end; weekly and daily
        rows keep French's exact date (for weekly data that's the week's last trading day,
        usually a Friday).

    Raises
    ------
    FrequencyError
        If ``freq`` isn't monthly, weekly, or daily.
    DownloadError
        If the download fails or the cache can't be written. The message explains how to
        load the file offline with ``path=``.
    InputError
        If the file isn't in French's format.

    Warns
    -----
    DataDroppedWarning
        If the file contains French's missing-data codes (-99.99 or -999). Those values
        become NaN.

    Examples
    --------
    >>> from alphafade.datasets import load_ff3
    >>> ff3 = load_ff3("M")  # doctest: +SKIP
    >>> list(ff3.columns)  # doctest: +SKIP
    ['Mkt-RF', 'SMB', 'HML', 'RF']
    >>> ff3.loc["1926-07-31", "Mkt-RF"]  # doctest: +SKIP
    0.0289
    >>> weekly = load_ff3("W", path="F-F_Research_Data_Factors_weekly_CSV.zip")  # doctest: +SKIP
    """
    code = _loadable_freq(freq, ("M", "W", "D"), "load_ff3")
    data = _load(_FF3_FILES[code], FF3_COLUMNS, code, cache_dir, refresh, path)
    return data


def load_momentum(
    freq: str = "M",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    path: str | os.PathLike[str] | None = None,
) -> pd.Series[float]:
    """Load French's momentum factor (UMD, "up minus down"), as decimals.

    UMD is the return of recent winners minus recent losers (prior 12-month return,
    skipping the latest month). French labels the column "Mom"; alphafade names the
    Series "UMD".

    Parameters
    ----------
    freq : {"M", "D"}, default "M"
        Monthly or daily. French doesn't publish a weekly momentum file; load daily data and
        compound it, e.g. ``(1 + umd).resample("W-FRI").prod() - 1``.
    cache_dir : str or path-like, optional
        Folder for the downloaded zip. Defaults to ``$ALPHAFADE_CACHE`` if set, otherwise
        ``~/.cache/alphafade/``.
    refresh : bool, default False
        Download again even if a cached copy exists.
    path : str or path-like, optional
        A local ``.zip`` or ``.csv`` file in French's format. When given, no network or
        cache is used at all.

    Returns
    -------
    Series
        Float64 decimals named ``"UMD"``, indexed by a sorted DatetimeIndex named
        ``"date"`` (month-end dates for monthly data).

    Raises
    ------
    FrequencyError
        If ``freq`` isn't monthly or daily.
    DownloadError
        If the download fails or the cache can't be written.
    InputError
        If the file isn't in French's format.

    Warns
    -----
    DataDroppedWarning
        If the file contains French's missing-data codes (-99.99 or -999). Those values
        become NaN.

    Examples
    --------
    >>> from alphafade.datasets import load_momentum
    >>> umd = load_momentum("M")  # doctest: +SKIP
    >>> umd.name  # doctest: +SKIP
    'UMD'
    >>> umd.index[0]  # doctest: +SKIP
    Timestamp('1927-01-31 00:00:00')
    """
    code = _loadable_freq(freq, ("M", "D"), "load_momentum")
    frame = _load(_MOM_FILES[code], (_MOM_COLUMN,), code, cache_dir, refresh, path)
    umd = frame[_MOM_COLUMN].rename("UMD")
    return umd


# ---------------------------------------------------------------------------------------
# Frequency handling


def _loadable_freq(freq: str, allowed: Sequence[_Loadable], func: str) -> _Loadable:
    code = parse_freq(freq)
    if code == "W" and "W" not in allowed:
        raise FrequencyError(
            "Ken French doesn't publish a weekly momentum factor, so load_momentum supports "
            "only 'M' (monthly) and 'D' (daily). For weekly returns, load daily data and "
            'compound it: `(1 + load_momentum("D")).resample("W-FRI").prod() - 1`.'
        )
    if code not in allowed:
        names = ", ".join(f"{c!r} ({_FREQ_NAMES[c]})" for c in allowed)
        raise FrequencyError(
            f"{func} supports freq {names}, got {freq!r}. French's files only include the "
            "annual table as an extra, which alphafade doesn't load. For quarterly or annual "
            'returns, load monthly data and compound it, e.g. `(1 + df).resample("QE")'
            ".prod() - 1`."
        )
    return code  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------
# Fetching: local path, cache, or download


def _load(
    filename: str,
    columns: Sequence[str],
    freq: _Loadable,
    cache_dir: str | os.PathLike[str] | None,
    refresh: bool,
    path: str | os.PathLike[str] | None,
) -> pd.DataFrame:
    if path is not None:
        local = Path(path).expanduser()
        try:
            raw = local.read_bytes()
        except OSError as exc:
            raise InputError(f"Couldn't read {str(local)!r}: {exc.strerror or exc}.") from None
        text = _extract_text(raw, str(local), expect_zip=local.suffix.lower() == ".zip")
        return _parse_french_csv(text, columns, freq, str(local))

    cached = _cache_root(cache_dir) / filename
    if refresh or not cached.is_file():
        url = BASE_URL + filename
        raw = _download(url)
        if not zipfile.is_zipfile(io.BytesIO(raw)):
            raise DownloadError(
                f"The download from {url} isn't a zip file (the server may have returned an "
                f"error page). Try again later, or download it in a browser and load it "
                f"offline with path='/path/to/{filename}'."
            )
        _write_atomic(cached, raw)
    else:
        raw = cached.read_bytes()
    try:
        text = _extract_text(raw, str(cached), expect_zip=True)
        return _parse_french_csv(text, columns, freq, str(cached))
    except InputError as exc:
        raise InputError(
            f"{exc} The cached copy may be damaged: pass refresh=True to download it again."
        ) from None


def _cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env = os.environ.get("ALPHAFADE_CACHE", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "alphafade"


def _download(url: str) -> bytes:
    """Fetch ``url`` and return its bytes (the only function that uses the network)."""
    # Imported here so `import alphafade` stays fast and never loads networking code.
    import urllib.error
    import urllib.request

    class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
        """Follow redirects only to https URLs (never downgrade to plain http)."""

        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            if not newurl.lower().startswith("https://"):
                raise urllib.error.HTTPError(
                    req.full_url, code, f"refusing non-https redirect to {newurl}", headers, fp
                )
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_HttpsOnlyRedirects)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    deadline = time.monotonic() + _DEADLINE_SECONDS
    try:
        with opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
            pieces: list[bytes] = []
            got = 0
            # Read in chunks so one slow-dripping server can't hold us past the deadline.
            while chunk := response.read(1 << 16):
                got += len(chunk)
                if got > _MAX_DOWNLOAD_BYTES:
                    raise DownloadError(
                        f"The download from {url} is larger than "
                        f"{_MAX_DOWNLOAD_BYTES // 2**20} MB, far bigger than any Ken French "
                        "file; refusing it. The server may be returning unexpected content."
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no complete download within {_DEADLINE_SECONDS} s")
                pieces.append(chunk)
            data = b"".join(pieces)
    except (OSError, http.client.HTTPException) as exc:
        reason = getattr(exc, "reason", None) or exc
        filename = url.rsplit("/", 1)[-1]
        raise DownloadError(
            f"Couldn't download {url} ({reason}). Check your internet connection, or "
            f"download the file in a browser and load it offline with "
            f"path='/path/to/{filename}'."
        ) from exc
    return data


def _write_atomic(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` via a temp file and a rename, so no one sees a partial file."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dest)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except OSError as exc:
        raise DownloadError(
            f"Couldn't save the download to the cache folder {str(dest.parent)!r} "
            f"({exc.strerror or exc}). Pass cache_dir= or set the ALPHAFADE_CACHE environment "
            "variable to a writable folder."
        ) from exc


# ---------------------------------------------------------------------------------------
# Parsing French's CSV format


def _extract_text(raw: bytes, source: str, *, expect_zip: bool) -> str:
    """Return the CSV text from a zip archive or from raw CSV bytes."""
    if zipfile.is_zipfile(io.BytesIO(raw)):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                if len(members) != 1:
                    raise InputError(
                        f"{source} should contain exactly one .csv file, found "
                        f"{len(members)} ({archive.namelist()})."
                    )
                # Never trust the size in the zip header (a forged one is how zip bombs work):
                # decompress in chunks and stop as soon as the real output passes the cap.
                chunks: list[bytes] = []
                total = 0
                with archive.open(members[0]) as member:
                    while chunk := member.read(1 << 20):
                        total += len(chunk)
                        if total > _MAX_CSV_BYTES:
                            raise InputError(
                                f"{source} contains a CSV over {_MAX_CSV_BYTES // 2**20} MB "
                                "uncompressed; refusing to decompress it. This isn't a Ken "
                                "French file."
                            )
                        chunks.append(chunk)
                raw = b"".join(chunks)
        except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as exc:
            raise InputError(f"{source} is a damaged zip file: {exc}.") from None
    elif expect_zip:
        raise InputError(f"{source} is not a valid zip file.")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _parse_french_csv(
    text: str, columns: Sequence[str], freq: _Loadable, source: str
) -> pd.DataFrame:
    """Parse the first table in a French CSV: preamble, header, rows, then a blank line.

    Everything after the first blank line that follows the data (the annual table and the
    copyright notice) is ignored.
    """
    lines = text.splitlines()
    wanted = [c.lower() for c in columns]

    header_at: int | None = None
    positions: list[int] = []
    for i, line in enumerate(lines):
        fields = [f.strip().lower() for f in line.split(",")]
        if len(fields) > 1 and set(wanted) <= set(fields[1:]):
            header_at = i
            positions = [fields.index(c, 1) for c in wanted]
            break
    if header_at is None:
        raise InputError(
            f"Couldn't find a header row with columns {list(columns)} in {source}. Is this "
            "the right Ken French file, saved as CSV?"
        )

    digits = _DATE_DIGITS[freq]
    dates: list[str] = []
    rows: list[list[float]] = []
    for lineno, line in enumerate(lines[header_at + 1 :], start=header_at + 2):
        if not line.replace(",", "").strip():
            break  # end of the first table
        fields = [f.strip() for f in line.split(",")]
        stamp = fields[0]
        if not (stamp.isdigit() and len(stamp) == digits):
            kind = {6: "YYYYMM", 8: "YYYYMMDD"}.get(len(stamp)) if stamp.isdigit() else None
            hint = (
                f" Dates like {stamp!r} look like {kind} data: pass the matching freq= "
                "(or the matching file)."
                if kind
                else ""
            )
            raise InputError(
                f"Line {lineno} of {source} should start with a {_FREQ_NAMES[freq]} date "
                f"({'YYYYMM' if digits == 6 else 'YYYYMMDD'}), got {line.strip()!r}.{hint}"
            )
        try:
            rows.append([float(fields[p]) for p in positions])
        except (IndexError, ValueError):
            raise InputError(
                f"Line {lineno} of {source} doesn't have numeric values for "
                f"{list(columns)}: {line.strip()!r}."
            ) from None
        dates.append(stamp)

    if not rows:
        raise InputError(f"{source} has a header row but no data rows under it.")

    fmt = "%Y%m" if digits == 6 else "%Y%m%d"
    try:
        index = pd.DatetimeIndex(pd.to_datetime(dates, format=fmt), name="date")
    except ValueError as exc:
        raise InputError(f"{source} contains an invalid date: {exc}") from None
    if freq == "M":
        index = pd.DatetimeIndex(index + pd.offsets.MonthEnd(0), name="date")
    index = index.as_unit("ns")
    if index.has_duplicates:
        dupes = [str(d.date()) for d in index[index.duplicated()].unique()[:3]]
        raise InputError(f"{source} has duplicate dates (e.g. {dupes}).")

    values = np.asarray(rows, dtype="float64")
    missing = np.zeros(values.shape, dtype=bool)
    for code in _MISSING_CODES:
        missing |= np.isclose(values, code, rtol=0.0, atol=1e-9)
    n_missing = int(missing.sum())
    if n_missing:
        values[missing] = np.nan
        warnings.warn(
            f"{source} contains {n_missing} missing-data code(s) (-99.99 or -999); "
            "they were converted to NaN.",
            DataDroppedWarning,
            stacklevel=4,
        )

    frame = pd.DataFrame(values / 100.0, index=index, columns=list(columns), dtype="float64")
    return frame.sort_index()
