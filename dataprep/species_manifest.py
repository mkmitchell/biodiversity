"""Load MAV species list from Excel with group → spgroup mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATAPREP = Path(__file__).resolve().parent
DEFAULT_EXCEL = DATAPREP / "mavBiodiversityToolSpeciesList.xlsx"

EXCEL_GROUP_TO_SPGROUP: dict[str, str] = {
    "birds": "avian",
    "amphibians": "herp",
    "reptiles": "herp",
    "mammals": "mammal",
}


def normalize_scientific_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def excel_group_to_spgroup(excel_group: str) -> str:
    key = excel_group.strip().lower()
    try:
        return EXCEL_GROUP_TO_SPGROUP[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Excel group {excel_group!r}; expected one of "
            f"{sorted(EXCEL_GROUP_TO_SPGROUP)}"
        ) from exc


def load_species_manifest(path: str | Path = DEFAULT_EXCEL) -> pd.DataFrame:
    """Return manifest with scientific_name, excel_group, and spgroup columns."""
    df = pd.read_excel(path)
    if "Scientific name" not in df.columns or "Group" not in df.columns:
        raise ValueError(
            "Excel must contain 'Scientific name' and 'Group' columns; "
            f"found {list(df.columns)}"
        )

    manifest = pd.DataFrame(
        {
            "scientific_name": df["Scientific name"].astype(str).map(normalize_scientific_name),
            "excel_group": df["Group"].astype(str).str.strip().str.lower(),
        }
    )
    manifest["spgroup"] = manifest["excel_group"].map(excel_group_to_spgroup)
    return manifest.drop_duplicates(subset=["scientific_name"], keep="first").reset_index(drop=True)


def get_species_entry(manifest: pd.DataFrame, sp: str) -> dict[str, str]:
    """Look up excel_group and spgroup for a normalized scientific name."""
    key = normalize_scientific_name(sp)
    row = manifest.loc[manifest["scientific_name"] == key]
    if row.empty:
        raise KeyError(f"Species {sp!r} not found in manifest")
    record = row.iloc[0]
    return {
        "scientific_name": record["scientific_name"],
        "excel_group": record["excel_group"],
        "spgroup": record["spgroup"],
    }
