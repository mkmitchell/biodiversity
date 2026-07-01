#!/usr/bin/env python3
"""Batch MaxEnt inference without papermill — parallel per species."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dataprep"))
sys.path.insert(0, str(ROOT))

from species_manifest import load_species_manifest  # noqa: E402

from maxent_model import (  # noqa: E402
    INFERENCE_SEASONS,
    audit_inference_outputs,
    inference_jobs,
    inference_season_complete,
    run_inference_species,
    species_output_dir,
)

DEFAULT_DATA_ROOT = Path("/mnt/f/biodiversity")


def _parse_seasons(raw: str) -> list[str]:
    if not raw.strip():
        return list(INFERENCE_SEASONS)
    seasons = [s.strip().lower() for s in raw.split(",") if s.strip()]
    unknown = set(seasons) - set(INFERENCE_SEASONS)
    if unknown:
        raise ValueError(f"Unknown seasons: {sorted(unknown)}; expected subset of {INFERENCE_SEASONS}")
    return seasons


def _species_list_from_args(species_arg: str, data_root: Path) -> list[str]:
    if species_arg.strip():
        return [s.strip().lower().replace(" ", "_") for s in species_arg.split(",") if s.strip()]
    manifest = load_species_manifest()
    return [s.replace(" ", "_").lower() for s in manifest["scientific_name"].tolist()]


def run_one_species(
    sp: str,
    data_root: str,
    seasons: list[str],
    *,
    force_stack: bool,
    verbose: bool,
) -> tuple:
    print(f"Running {sp} ({len(seasons)} season(s))", flush=True)
    try:
        results = run_inference_species(
            sp, data_root, seasons, force_stack=force_stack, verbose=verbose
        )
        spdir = species_output_dir(data_root, sp)
        for row in results:
            if not inference_season_complete(spdir, sp, row["season"]):
                raise RuntimeError(f"prediction tifs missing for {row['season']}")
        return (sp, "ok", results)
    except Exception as exc:
        print(f"FAILED {sp}: {exc}")
        traceback.print_exc()
        return (sp, "fail", str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MaxEnt inference in parallel (per species).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mode", choices=("all", "missing_only"), default="missing_only")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel species workers")
    parser.add_argument("--species", type=str, default="", help="Comma-separated species override")
    parser.add_argument("--seasons", type=str, default="", help="Comma-separated seasons (default: all 4)")
    parser.add_argument("--force-stack", action="store_true", help="Rebuild inference_stack_*.tif caches")
    parser.add_argument("--verbose", action="store_true", help="Print band mapping and GDAL/predict details")
    args = parser.parse_args(argv)

    data_root = str(args.data_root)
    seasons = _parse_seasons(args.seasons)
    species_list = _species_list_from_args(args.species, args.data_root)

    if args.species.strip():
        job_list = [(sp, season) for sp in species_list for season in seasons]
    else:
        job_list = inference_jobs(
            species_list,
            data_root,
            missing_only=(args.mode == "missing_only"),
            seasons=seasons,
        )

    if not job_list:
        print("Nothing to run.")
        return 0

    by_species: dict[str, list[str]] = {}
    for sp, season in job_list:
        by_species.setdefault(sp, [])
        if season not in by_species[sp]:
            by_species[sp].append(season)

    species_order = sorted(by_species.keys())
    print(f"Queued {len(job_list)} season job(s) across {len(species_order)} species")

    if args.jobs <= 1:
        completed = [
            run_one_species(
                sp, data_root, by_species[sp], force_stack=args.force_stack, verbose=args.verbose
            )
            for sp in species_order
        ]
    else:
        completed = Parallel(n_jobs=args.jobs, verbose=1)(
            delayed(run_one_species)(
                sp, data_root, by_species[sp], force_stack=args.force_stack, verbose=args.verbose
            )
            for sp in species_order
        )

    ok = [row for row in completed if row[1] == "ok"]
    failed = [row for row in completed if row[1] == "fail"]
    print(f"\nFinished: {len(ok)} species ok, {len(failed)} species failed")
    for sp, _, err in failed:
        print(f"  FAIL {sp}: {err}")

    audit = audit_inference_outputs(species_list, data_root, seasons=seasons)
    print(
        f"After run — complete: {len(audit['complete'])}/{audit['modelable_species']} modelable "
        f"({len(audit['complete'])}/{len(species_list)} manifest)"
    )
    print(f"Prediction tifs: {audit['present_tifs']}/{audit['expected_tifs']}")
    if audit["partial"]:
        print("Still partial:")
        for row in audit["partial"]:
            print(f"  {row['species']}: missing {row['missing_seasons']}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
