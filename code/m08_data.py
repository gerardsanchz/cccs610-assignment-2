"""Data acquisition for session 8: NRCan fuel-consumption ratings and the Quebec City property roll.

Both sources are untrusted downloads: every field is parsed with an explicit type and checked
against a plausible range, and a failure raises instead of being skipped.
"""

import datetime
import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

CARS_PACKAGE = "https://open.canada.ca/data/api/action/package_show?id=98f1a129-f628-4ce4-b24d-6f16bf24dd64"
CARS_URL = (
    "https://open.canada.ca/data/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64/resource/"
    "9df1b18d-d036-4783-a61c-99f1f75b3ac5/download/my2026-fuel-consumption-ratings.csv"
)
CARS_YEAR = 2026
CARS_LICENCE = {
    "title": "Open Government Licence - Canada",
    "url": "https://open.canada.ca/en/open-government-licence-canada",
}
ROLL_URL = "https://donneesouvertes.affmunqc.net/role/RL23027_2026.xml"
ROLL_INDEX_URL = "https://donneesouvertes.affmunqc.net/role/indexRole2026.csv"
ROLL_LICENCE = {
    "title": "Attribution (CC-BY 4.0)",
    "url": "https://www.donneesquebec.ca/licence/#cc-by",
}

# Generic codes (RL0101Ex) that the open-data guide does not decode; these are read off the street
# names they accompany. Any code not listed stays as the raw code.
GENERIC = {
    "RU": "rue",
    "AV": "avenue",
    "BO": "boulevard",
    "CH": "chemin",
    "PL": "place",
    "CT": "cote",
    "RO": "route",
    "RA": "rang",
    "IM": "impasse",
    "TE": "terrasse",
    "MO": "montee",
    "AL": "allee",
    "RL": "ruelle",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_dir():
    base = os.environ.get("CONDOR_SCRATCH")
    root = Path(base) if base else Path.home() / ".cache" / "cccs610-m08"
    d = root / "m08-roll"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url, dest, timeout=120):
    """Stream a URL to dest; raise on any HTTP error."""
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.writelines(r.iter_content(1 << 20))


def ensure_cars(data_dir):
    """Return (path, provenance-or-None). provenance is set only when the file was fetched now."""
    path = Path(data_dir) / f"m08-fuel-consumption-ratings-{CARS_YEAR}.csv"
    if path.exists():
        return path, None
    download(CARS_URL, path)
    return path, {
        "access_date": datetime.datetime.now(tz=datetime.UTC).date().isoformat()
    }


def load_cars(path):
    """Read the ratings strictly as UTF-8 (the MY2026 file is plain ASCII) and validate it."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = [
        "Model year",
        "Make",
        "Model",
        "Vehicle class",
        "Engine size (L)",
        "Cylinders",
        "Transmission",
        "Fuel type",
        "City (L/100 km)",
        "Highway (L/100 km)",
        "Combined (L/100 km)",
        "Combined (mpg)",
        "CO2 emissions (g/km)",
        "CO2 rating",
        "Smog rating",
    ]
    if list(df.columns) != expected:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError("missing values in the ratings file")
    if not (df["Model year"] == CARS_YEAR).all():
        raise ValueError("rows from another model year")
    for col, lo, hi in [
        ("Engine size (L)", 0.5, 10),
        ("Combined (L/100 km)", 2, 40),
        ("CO2 emissions (g/km)", 0, 700),
        ("CO2 rating", 1, 10),
        ("Smog rating", 1, 10),
    ]:
        if not df[col].between(lo, hi).all():
            raise ValueError(f"{col} outside {lo}..{hi}")
    df = df.reset_index(drop=True)
    df["label"] = (
        df["Make"].str.strip() + " " + df["Model"].str.strip() + f" ({CARS_YEAR})"
    )
    df["row"] = df.index + 1
    return df


def _street_label(a):
    num = a.findtext("RL0101Ax")
    letter = a.findtext("RL0101Bx") or ""
    gen = a.findtext("RL0101Ex")
    name = a.findtext("RL0101Gx")
    kind = GENERIC.get(gen, gen.upper() if gen else "")
    return f"{num}{letter} {kind} {name}".replace("  ", " ").strip(), num, name


def _f(el, tag):
    t = el.findtext(tag)
    return float(t) if t not in (None, "") else float("nan")


def extract_roll(xml_path):
    """Stream-parse the roll and keep detached single-dwelling residential units (one dwelling, land owner).

    Returns (DataFrame, funnel dict of counts at each filter step).
    """
    funnel = {
        "units_in_roll": 0,
        "predominant_use_1000": 0,
        "one_dwelling_detached": 0,
        "land_owner_no_rooms_no_commercial": 0,
    }
    rows = []
    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag != "RLUEx":
            continue
        funnel["units_in_roll"] += 1
        if el.findtext("RL0105A") == "1000":
            funnel["predominant_use_1000"] += 1
            if el.findtext("RL0311A") == "1" and el.findtext("RL0309A") == "1":
                funnel["one_dwelling_detached"] += 1
                if (
                    el.findtext("RL0201/RL0201U") == "1"
                    and not el.findtext("RL0312A")
                    and not el.findtext("RL0313A")
                ):
                    funnel["land_owner_no_rooms_no_commercial"] += 1
                    a = el.find("RL0101/RL0101x")
                    if (
                        a is not None
                        and a.findtext("RL0101Ax")
                        and a.findtext("RL0101Gx")
                    ):
                        label, _num, name = _street_label(a)
                        rows.append({
                            "dossier": int(el.findtext("RL0106A")),
                            "address": label,
                            "street": name,
                            "lot_m2": _f(el, "RL0302A"),
                            "floor_m2": _f(el, "RL0308A"),
                            "storeys": _f(el, "RL0306A"),
                            "year_built": _f(el, "RL0307A"),
                            "year_kind": el.findtext("RL0307B") or "",
                            "land_value": _f(el, "RL0402A"),
                            "building_value": _f(el, "RL0403A"),
                            "value": _f(el, "RL0404A"),
                        })
        el.clear()
    return pd.DataFrame(rows), funnel


def clean_roll(df):
    """Drop rows that are incomplete or implausible for a house purchase; return (df, funnel)."""
    f = {"extracted": len(df)}
    df = df.dropna(
        subset=[
            "lot_m2",
            "floor_m2",
            "year_built",
            "value",
            "storeys",
            "land_value",
            "building_value",
        ]
    )
    f["complete_fields"] = len(df)
    df = df[
        (df["floor_m2"] >= 40)
        & (df["floor_m2"] <= 1000)
        & (df["value"] >= 50_000)
        & (df["lot_m2"] >= 100)
        & (df["lot_m2"] <= 20_000)
        & (df["year_built"] >= 1700)
        & (df["year_built"] <= 2026)
    ]
    f["plausible_ranges"] = len(df)
    df = df.drop_duplicates("dossier").sort_values("dossier").reset_index(drop=True)
    f["unique_dossier"] = len(df)
    for c in ("lot_m2", "floor_m2"):
        df[c] = df[c].round(1)
    for c in ("storeys", "year_built", "land_value", "building_value", "value"):
        df[c] = df[c].astype(int)
    return df, f


def ensure_houses(data_dir):
    """Return (derived DataFrame, provenance-or-None, funnel-or-None).

    The derived csv.gz is the committed artefact; the 270 MB XML is only downloaded when it is missing.
    """
    out = Path(data_dir) / "m08-quebec-houses.csv.gz"
    if out.exists():
        return pd.read_csv(out), None, None
    xml = cache_dir() / "RL23027_2026.xml"
    if not xml.exists():
        download(ROLL_URL, xml, timeout=600)
    raw, f1 = extract_roll(xml)
    df, f2 = clean_roll(raw)
    df.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    prov = {
        "access_date": datetime.datetime
        .fromtimestamp(xml.stat().st_mtime, tz=datetime.UTC)
        .date()
        .isoformat(),
        "raw_xml_bytes": xml.stat().st_size,
        "raw_xml_sha256": sha256_of(xml),
    }
    return df, prov, {**f1, **f2}
