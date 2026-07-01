"""
Prepare eBird observation parquet for MAV species listed in an Excel workbook.

IMPORTANT: eBird API 2.0 cannot bulk-download multi-year history. The historic
endpoint is ONE calendar day per request (see API docs:
https://documenter.getpostman.com/view/664302/S1ENwy59). A 1990–2025 run across
AR/LA/MS would be ~39,000 API calls and is not supported here.

Use instead:
  1. eBird Basic Dataset **Custom Download** (web form) per species/region/date range
     https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data
  2. Place extracts under /mnt/f/biodiversity/ebird (ebd_<pkg>/ebd_<pkg>.txt)
  3. Run this script in **convert** mode (default) → hive parquet on /mnt/f/biodiversity/ebirdpolars

This script:
  - Reads species from "Common Name" / "Scientific name" in mavBiodiversityToolSpeciesList.xlsx
  - Resolves eBird species codes (small API use: taxonomy cache)
  - **convert** mode: runs convert_ebird_downloads.py on local TSV files
  - **manifest** mode: writes a species-code manifest + EBD download instructions
  - **gaps** mode: list species missing EBD TSV and/or parquet (uses taxonomy cache)
  - **pipeline** mode: full status through param CSVs (EBD → parquet → geeDataFromPoints)
  - **api-historic** mode: opt-in daily API loop (slow; not recommended)

Requires:
  conda activate rapids-25.10
  export EBIRD_API_KEY='your-key'

Usage:
  cd dataprep
  export EBIRD_API_KEY='...'
  python -u download_ebird_api_mav.py                    # convert local EBD TSVs
  python -u download_ebird_api_mav.py --mode manifest   # species list + EBD help
  python -u download_ebird_api_mav.py --mode gaps       # missing downloads / parquet
  python -u download_ebird_api_mav.py --mode pipeline   # through param CSV stage
  python -u download_ebird_api_mav.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
from shapely.geometry import Point

from ebird_polars_io import COMPLETE_MARKER, PARTITION_PREFIX, log

from gap_species import (
    GAP_TEST_SPECIES,
    gap_test_species_for_gee,
    gap_test_species_need_ebd,
    has_parquet_partition,
    report_pipeline_gaps,
)

from paths import DEFAULT_EBD_ROOT, DEFAULT_EBIRD_PARQUET, DEFAULT_PARAM_CSV

DATAPREP = Path(__file__).resolve().parent
DEFAULT_EXCEL = DATAPREP / "mavBiodiversityToolSpeciesList.xlsx"
DEFAULT_AOI = DATAPREP / "mav_counties_4326.parquet"
DEFAULT_AOI_GEOJSON = DATAPREP / "mav_counties_4326.geojson"
DEFAULT_OUTPUT = DEFAULT_EBIRD_PARQUET
CONVERT_SCRIPT = DATAPREP / "convert_ebird_downloads.py"

EBIRD_API_BASE = "https://api.ebird.org/v2"
MAV_STATE_REGIONS = ("US-AR", "US-LA", "US-MS")

SPECIES_COLUMN_NAMES = frozenset(
    {
        "common name",
        "common_name",
        "scientific name",
        "scientific_name",
        "scientificname",
    }
)

TAXONOMY_CACHE_NAME = "_ebird_taxonomy_species.json"
MANIFEST_NAME = "mav_species_ebird_codes.json"
CHECKPOINT_NAME = "_api_download_checkpoint.json"


def _insecure_ssl_default() -> bool:
    return os.environ.get("EBIRD_INSECURE_SSL", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def _api_headers(api_key: str) -> dict[str, str]:
    return {
        "X-eBirdApiToken": api_key,
        "Accept": "application/json",
    }


def _ssl_context(verify_ssl: bool) -> ssl.SSLContext | None:
    if verify_ssl:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def api_get(
    path: str,
    api_key: str,
    params: dict | None = None,
    *,
    verify_ssl: bool = True,
) -> object:
    url = f"{EBIRD_API_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=_api_headers(api_key))
    try:
        with urllib.request.urlopen(
            req, timeout=120, context=_ssl_context(verify_ssl)
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"eBird API {exc.code} for {url}: {body}") from exc


def _normalize_col(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def read_species_from_excel(excel_path: Path) -> tuple[list[str], list[str]]:
    scientific: set[str] = set()
    common: set[str] = set()
    xl = pd.ExcelFile(excel_path)
    found_cols: list[str] = []

    for sheet in xl.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        col_map = {_normalize_col(c): c for c in df.columns}
        for norm, orig in col_map.items():
            if norm not in SPECIES_COLUMN_NAMES:
                continue
            found_cols.append(f"{sheet}!{orig}")
            values = df[orig].dropna().astype(str).str.strip()
            values = values[values != ""]
            if "scientific" in norm:
                scientific.update(values.tolist())
            else:
                common.update(values.tolist())

    if not found_cols:
        sheet_cols = {
            sheet: list(pd.read_excel(excel_path, sheet_name=sheet).columns)
            for sheet in xl.sheet_names
        }
        raise ValueError(
            "No 'Common Name' or 'Scientific name' columns found.\n"
            f"Sheets/columns seen: {sheet_cols}"
        )

    log(f"Species columns: {', '.join(found_cols)}")
    log(f"  scientific: {len(scientific)}, common: {len(common)}")
    return sorted(scientific), sorted(common)


def load_taxonomy_cache(
    output_dir: Path, api_key: str, *, verify_ssl: bool
) -> list[dict]:
    cache = output_dir / TAXONOMY_CACHE_NAME
    if cache.is_file():
        with cache.open(encoding="utf-8") as f:
            return json.load(f)
    log("Fetching eBird taxonomy (one-time API call) ...")
    taxa = api_get(
        "/ref/taxonomy/ebird",
        api_key,
        {"cat": "species", "fmt": "json", "locale": "en"},
        verify_ssl=verify_ssl,
    )
    if not isinstance(taxa, list):
        raise RuntimeError("Unexpected taxonomy response")
    output_dir.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as f:
        json.dump(taxa, f)
    log(f"Cached {len(taxa)} species")
    return taxa


def resolve_species_codes(
    scientific: list[str],
    common: list[str],
    taxa: list[dict],
) -> dict[str, dict]:
    by_sci = {t.get("sciName", "").strip().lower(): t for t in taxa}
    by_com = {t.get("comName", "").strip().lower(): t for t in taxa}
    resolved: dict[str, dict] = {}

    for name in scientific:
        key = name.strip().lower().replace(" ", "_")
        t = by_sci.get(name.strip().lower())
        if t is None:
            log(f"WARNING: no eBird match for scientific name '{name}'")
            continue
        resolved[key] = t

    for name in common:
        t = by_com.get(name.strip().lower())
        if t is None:
            log(f"WARNING: no eBird match for common name '{name}'")
            continue
        key = t.get("sciName", "").strip().lower().replace(" ", "_")
        if key not in resolved:
            resolved[key] = t

    return resolved


def write_manifest(
    output_dir: Path,
    resolved: dict[str, dict],
    *,
    start_year: int,
    end_year: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_NAME
    payload = {
        "start_year": start_year,
        "end_year": end_year,
        "aoi_parquet": str(DEFAULT_AOI),
        "ebd_root": str(DEFAULT_EBD_ROOT),
        "states_for_custom_download": list(MAV_STATE_REGIONS),
        "species": [
            {
                "scientific_name_key": key,
                "sciName": t.get("sciName"),
                "comName": t.get("comName"),
                "speciesCode": t.get("speciesCode"),
            }
            for key, t in sorted(resolved.items())
        ],
        "ebd_custom_download_help": (
            "https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data"
        ),
        "instructions": (
            "Request one EBD Custom Download per species: region=Mississippi/Louisiana/Arkansas "
            f"(or MAV counties), dates={start_year}-{end_year}. Save as ebd_<pkg>/ebd_<pkg>.txt "
            f"under {DEFAULT_EBD_ROOT}, then run: python -u download_ebird_api_mav.py --mode convert"
        ),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"Wrote manifest: {path}")
    return path


def species_keys_from_excel(excel_path: Path) -> list[str]:
    """Scientific-name keys only (matches geeDataFromPoints species_list)."""
    scientific, _common = read_species_from_excel(excel_path)
    keys: list[str] = []
    seen: set[str] = set()
    for name in scientific:
        key = name.strip().lower().replace(" ", "_")
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return sorted(keys)


def ebd_folders_for_code(ebd_root: Path, species_code: str) -> list[str]:
    """EBD folders whose name contains the eBird species code (e.g. rewbla)."""
    code = species_code.strip().lower()
    if not code or not ebd_root.is_dir():
        return []
    return sorted(
        d.name
        for d in ebd_root.iterdir()
        if d.is_dir() and d.name.startswith("ebd_") and code in d.name.lower()
    )


def report_gaps(
    *,
    excel_path: Path,
    ebd_root: Path,
    output_dir: Path,
    resolved: dict[str, dict],
    species_keys: list[str],
) -> int:
    """Print which MAV list species still need EBD download and/or convert."""
    need_ebd: list[tuple[str, str, str]] = []
    need_convert: list[tuple[str, str, str, list[str]]] = []
    gbif_only: list[str] = []

    log(f"EBD root: {ebd_root}")
    log(f"Parquet output: {output_dir}")
    log("")

    for key in species_keys:
        if key not in resolved:
            gbif_only.append(key)
            continue
        t = resolved[key]
        code = str(t.get("speciesCode") or "")
        com = str(t.get("comName") or "")
        folders = ebd_folders_for_code(ebd_root, code)
        has_pq = has_parquet_partition(output_dir, key)
        if has_pq:
            ebd_note = folders[0] if folders else "parquet only (legacy?)"
            log(f"OK  {key}  ({code})  parquet=yes  ebd={ebd_note}")
        elif folders:
            need_convert.append((key, code, com, folders))
        else:
            need_ebd.append((key, code, com))

    if need_ebd:
        log("")
        log(f"NEED EBD CUSTOM DOWNLOAD ({len(need_ebd)}) — save TSV under {ebd_root}:")
        log(
            "  https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data"
        )
        for key, code, com in need_ebd:
            log(f"  {key}")
            log(f"    {com}  →  eBird species code: {code}")
            log(f"    expect folder like ebd_US_{code}_relApr-2026/ebd_US_{code}_relApr-2026.txt")

    if need_convert:
        log("")
        log(f"NEED CONVERT ({len(need_convert)}) — TSV on disk, no parquet yet:")
        for key, code, com, folders in need_convert:
            log(f"  {key}  ({code})  folders: {', '.join(folders)}")

    if gbif_only:
        log("")
        log(f"NOT IN EBIRD TAXONOMY ({len(gbif_only)}) — use GBIF CSV, not EBD:")
        for key in gbif_only:
            log(f"  {key}")

    log("")
    log(
        f"Summary: {len(species_keys)} in list, "
        f"{len(need_ebd)} need download, {len(need_convert)} need convert, "
        f"{len(gbif_only)} GBIF-only"
    )
    return 0


def run_convert(ebd_root: Path, output_dir: Path, extra_args: list[str]) -> int:
    if not CONVERT_SCRIPT.is_file():
        log(f"ERROR: missing {CONVERT_SCRIPT}")
        return 1
    env = os.environ.copy()
    env["EBIRD_INPUT_ROOT"] = str(ebd_root)
    env["EBIRD_OUTPUT"] = str(output_dir)
    cmd = [sys.executable, "-u", str(CONVERT_SCRIPT), *extra_args]
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=DATAPREP, env=env)
    return result.returncode


# --- Optional slow API path (not default) ---


def iter_dates(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def fetch_historic_day(
    region: str,
    day: date,
    api_key: str,
    target_codes: set[str],
    *,
    verify_ssl: bool,
) -> list[dict]:
    path = f"/data/obs/{region}/historic/{day.year}/{day.month:02d}/{day.day:02d}"
    rows = api_get(
        path,
        api_key,
        {"maxResults": 10000, "detail": "full", "includeProvisional": "true"},
        verify_ssl=verify_ssl,
    )
    if not isinstance(rows, list):
        return []
    return [r for r in rows if r.get("speciesCode") in target_codes]


def ensure_aoi_parquet(aoi_path: Path) -> gpd.GeoDataFrame:
    if aoi_path.is_file():
        gdf = gpd.read_parquet(aoi_path)
    elif DEFAULT_AOI_GEOJSON.is_file():
        gdf = gpd.read_file(DEFAULT_AOI_GEOJSON).to_crs("EPSG:4326")
        aoi_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(aoi_path)
    else:
        raise FileNotFoundError(f"AOI not found: {aoi_path}")
    return gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf


def api_historic_download(
    *,
    resolved: dict[str, dict],
    aoi_path: Path,
    output_dir: Path,
    api_key: str,
    start_year: int,
    end_year: int,
    sleep_s: float,
    verify_ssl: bool,
) -> int:
    """One API call per day per state — NOT recommended for multi-year ranges."""
    code_to_key = {
        t["speciesCode"]: k for k, t in resolved.items() if t.get("speciesCode")
    }
    target_codes = set(code_to_key)
    aoi_gdf = ensure_aoi_parquet(aoi_path)
    aoi = aoi_gdf.union_all() if len(aoi_gdf) > 1 else aoi_gdf.geometry.iloc[0]

    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    dates = iter_dates(start, end)
    total = len(dates) * len(MAV_STATE_REGIONS)
    log(f"API historic mode: {total} calls ({len(dates)} days x {len(MAV_STATE_REGIONS)} states)")

    checkpoint = output_dir / CHECKPOINT_NAME
    done: set[str] = set()
    if checkpoint.is_file():
        with checkpoint.open(encoding="utf-8") as f:
            done = set(json.load(f))

    buffers: dict[str, list[dict]] = {k: [] for k in resolved}
    calls = 0

    for day in dates:
        for region in MAV_STATE_REGIONS:
            ck = f"{region}:{day.isoformat()}"
            calls += 1
            if ck in done:
                continue
            log(f"[{calls}/{total}] {region} {day.isoformat()}")
            rows = fetch_historic_day(
                region, day, api_key, target_codes, verify_ssl=verify_ssl
            )
            for row in rows:
                lat, lng = row.get("lat"), row.get("lng")
                if lat is None or lng is None:
                    continue
                if not aoi.contains(Point(lng, lat)):
                    continue
                key = code_to_key.get(row.get("speciesCode"))
                if key:
                    buffers[key].append(row)
            done.add(ck)
            if calls % 50 == 0:
                with checkpoint.open("w", encoding="utf-8") as f:
                    json.dump(sorted(done), f)
            if sleep_s > 0:
                time.sleep(sleep_s)

    for key, buf in buffers.items():
        if not buf:
            continue
        df = pl.DataFrame(buf)
        if "sciName" in df.columns:
            df = df.with_columns(
                pl.col("sciName")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .alias("scientific_name")
            )
        part = output_dir / f"{PARTITION_PREFIX}{key}"
        part.mkdir(parents=True, exist_ok=True)
        out = part / "api_observations.parquet"
        if out.is_file():
            df = pl.concat([pl.read_parquet(out), df], how="diagonal_relaxed").unique()
        df.write_parquet(out, compression="snappy")
        (part / COMPLETE_MARKER).touch()

    with checkpoint.open("w", encoding="utf-8") as f:
        json.dump(sorted(done), f)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MAV eBird species → parquet (EBD convert default; API historic opt-in)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Recommended workflow for 1990–2025:\n"
            "  1. --mode manifest  → get species codes + EBD download instructions\n"
            "  2. --mode gaps      → see missing EBD / parquet (e.g. rewbla not rwbl)\n"
            "  3. Download TSVs from eBird Custom Download into /mnt/f/biodiversity/ebird\n"
            "  4. --mode convert   → fast local conversion to /mnt/f/biodiversity/ebirdpolars\n"
            "  5. --mode pipeline  → see which species still need geeDataFromPoints\n"
        ),
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ebd-root", type=Path, default=DEFAULT_EBD_ROOT)
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--mode",
        choices=("convert", "manifest", "gaps", "pipeline", "api-historic"),
        default="convert",
        help=(
            "convert=local EBD TSVs (default); manifest=species JSON; gaps=missing EBD/parquet; "
            "pipeline=through param CSVs; api-historic=slow"
        ),
    )
    parser.add_argument(
        "--species",
        action="append",
        default=None,
        help="Scientific name key(s) for pipeline mode (repeatable).",
    )
    parser.add_argument(
        "--param-dir",
        type=Path,
        default=DEFAULT_PARAM_CSV,
        help="param_csvs directory for pipeline mode",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--i-know-api-is-slow",
        action="store_true",
        help="Required with --mode api-historic",
    )
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument(
        "--insecure",
        action=argparse.BooleanOptionalAction,
        default=_insecure_ssl_default(),
    )
    args, extra = parser.parse_known_args(argv)

    if not args.excel.is_file():
        log(f"ERROR: Excel not found: {args.excel}")
        return 1

    api_key = os.environ.get("EBIRD_API_KEY", "").strip()
    tax_cache = args.output / TAXONOMY_CACHE_NAME
    need_taxonomy = args.mode in ("manifest", "gaps", "pipeline", "api-historic") or args.dry_run
    if need_taxonomy and not tax_cache.is_file() and not api_key:
        log("ERROR: set EBIRD_API_KEY (or run manifest once to cache taxonomy)")
        return 1

    verify_ssl = not args.insecure
    if not verify_ssl:
        log("WARNING: SSL certificate verification disabled")

    scientific, common = read_species_from_excel(args.excel)
    species_keys = species_keys_from_excel(args.excel)
    resolved: dict[str, dict] = {}
    if need_taxonomy:
        if tax_cache.is_file() and not api_key:
            log(f"Using cached taxonomy: {tax_cache}")
            with tax_cache.open(encoding="utf-8") as f:
                taxa = json.load(f)
        else:
            taxa = load_taxonomy_cache(args.output, api_key, verify_ssl=verify_ssl)
        resolved = resolve_species_codes(scientific, common, taxa)
        if not resolved:
            log("ERROR: no species resolved to eBird codes")
            return 1
        log(f"Resolved {len(resolved)} species")

    if args.dry_run:
        for key, t in sorted(resolved.items()):
            log(f"  {key}  {t.get('comName')}  code={t.get('speciesCode')}")
        log(f"Mode would be: {args.mode}")
        return 0

    if args.mode == "manifest":
        write_manifest(
            args.output, resolved, start_year=args.start_year, end_year=args.end_year
        )
        log(
            "Next: request EBD Custom Downloads (one TSV per species), "
            f"save under {args.ebd_root}, then run --mode convert"
        )
        return 0

    if args.mode == "gaps":
        return report_gaps(
            excel_path=args.excel,
            ebd_root=args.ebd_root,
            output_dir=args.output,
            resolved=resolved,
            species_keys=species_keys,
        )

    if args.mode == "pipeline":
        keys = list(GAP_TEST_SPECIES) if args.species is None else [
            s.strip().lower().replace(" ", "_") for s in args.species
        ]
        report_pipeline_gaps(
            keys,
            resolved=resolved,
            ebd_root=args.ebd_root,
            ebird_parquet=args.output,
            param_dir=args.param_dir,
        )
        need_gee = gap_test_species_for_gee(
            ebird_parquet=args.output, param_dir=args.param_dir
        )
        need_ebd = gap_test_species_need_ebd(ebird_parquet=args.output)
        if need_gee:
            log("")
            log("Next: run geeDataFromPoints with only_species =")
            log(f"  {need_gee}")
        if need_ebd:
            log("")
            log("Next: EBD Custom Download for:")
            log(f"  {need_ebd}")
        return 0

    if args.mode == "convert":
        log(
            "Using local EBD tab files (not day-by-day API). "
            f"Years {args.start_year}-{args.end_year} must be covered by your downloads."
        )
        convert_args = list(extra)
        if args.dry_run:
            convert_args.append("--dry-run")
        return run_convert(args.ebd_root, args.output, convert_args)

    if args.mode == "api-historic":
        if not args.i_know_api_is_slow:
            log(
                "ERROR: api-historic uses ONE API call per day per state (~39k for 1990-2025). "
                "Use --mode convert with EBD Custom Download files instead, or pass --i-know-api-is-slow"
            )
            return 1
        return api_historic_download(
            resolved=resolved,
            aoi_path=args.aoi,
            output_dir=args.output,
            api_key=api_key,
            start_year=args.start_year,
            end_year=args.end_year,
            sleep_s=args.sleep,
            verify_ssl=verify_ssl,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
