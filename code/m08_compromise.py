"""Session 8, multiobjective compromising: real cars and real houses, Pareto set, and five ways to pick.

Run: python -I m08_compromise.py
Cars: NRCan fuel consumption ratings (no price, no mileage in that file).
Houses: Quebec City property assessment roll (assessed value, not an asking price).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import m08_data as md
import m08_pipeline as mpl

D = HERE.parent
DATA, FIG, RES = D / "data", D / "figures", D / "results"
for p in (DATA, FIG, RES):
    p.mkdir(exist_ok=True)

SEED = 8
SOURCE_JSON = RES / "m08-source.json"


def previous_source():
    return json.loads(SOURCE_JSON.read_text()) if SOURCE_JSON.exists() else {}


def cars_run(source):
    path, prov = md.ensure_cars(DATA)
    old = previous_source().get("cars", {})
    access = prov["access_date"] if prov else old.get("access_date")
    if access is None:
        raise RuntimeError(
            "cars file exists but its access date was never recorded in m08-source.json"
        )
    df = md.load_cars(path)
    source["cars"] = {
        "dataset": "Fuel consumption ratings, Natural Resources Canada",
        "resource": "2026 Fuel Consumption Ratings (2026-10-06): newest model year in the dataset",
        "dataset_page_api": md.CARS_PACKAGE,
        "download_url": md.CARS_URL,
        "licence": md.CARS_LICENCE,
        "access_date": access,
        "local_copy": f"data/{path.name}",
        "bytes": path.stat().st_size,
        "rows": len(df),
        "columns": [c for c in df.columns if c not in ("label", "row")],
        "sha256": md.sha256_of(path),
        "encoding": "UTF-8 (plain ASCII); the MY2024 file in the same dataset needed a Latin-1 or cp1252 decoder",
        "scope_note": "internal-combustion ratings only (fuel types Z, X, D, E); battery-electric and plug-in hybrid vehicles are separate files in the dataset and are left out",
        "no_price_no_mileage": "This data has no price and no mileage, so the class criteria Price and Milage cannot be used.",
    }
    crit = [
        ("Combined (L/100 km)", "min"),
        ("Engine size (L)", "max"),
        ("Smog rating", "max"),
    ]
    spec = {
        "name": "cars",
        "df": df,
        "id_col": "row",
        "criteria": crit,
        "corr_cols": [
            "Combined (L/100 km)",
            "CO2 emissions (g/km)",
            "CO2 rating",
            "Engine size (L)",
            "Cylinders",
            "Smog rating",
        ],
        "weights": {
            "Combined (L/100 km)": 0.5,
            "Engine size (L)": 0.3,
            "Smog rating": 0.2,
        },
        "eps": {
            "primary": "Combined (L/100 km)",
            "bounds": {"Engine size (L)": (">=", 3.0), "Smog rating": (">=", 5)},
            "sweep_col": "Engine size (L)",
            "sweep_op": ">=",
            "sweep_values": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
        },
        "lex": {
            "order": ["Combined (L/100 km)", "Engine size (L)", "Smog rating"],
            "tol": 0.1,
        },
        "goal": {"Combined (L/100 km)": 8.0, "Engine size (L)": 3.0, "Smog rating": 6},
        "k": 6,
        "seed": SEED,
        "full_naive": True,
        "plot": {
            "x": "Combined (L/100 km)",
            "y": "Engine size (L)",
            "color": "Smog rating",
            "size": None,
            "xlabel": "Combined fuel consumption (L/100 km), minimise",
            "ylabel": "Engine size (L), maximise",
            "clabel": "Smog rating (1-7, higher is cleaner), maximise",
        },
        # Text and layout of the two figures in the report (US spelling; see m08_pipeline.run).
        "print": {
            "title": "MY2026 cars",
            "xlabel": "Combined fuel consumption (L/100 km), minimize",
            "ylabel": "Engine size (L), maximize",
            "clabel": "Smog rating (1-7, higher is cleaner), maximize",
            "x_tick_step": None,
            "pareto_ax_h": 3.0,
            "weights_ax_h": 3.1,
        },
        "fig_pareto": FIG / "m08-cars-pareto.png",
        "fig_weights": FIG / "m08-cars-weights.png",
        "fig_diversity": FIG / "m08-cars-diversity.png",
    }
    out = mpl.run(spec)
    out["criteria_rationale"] = {
        "Combined (L/100 km)": "minimise: fuel cost and the proxy for CO2, the stand-in for the class Price and FuelEfficiency",
        "Engine size (L)": "maximise: proxy for power, towing and cabin refinement (no power rating exists in the file); it conflicts with fuel",
        "Smog rating": "maximise: higher is cleaner for local air pollutants (NOx, particulates); only partly aligned with fuel",
        "excluded": "CO2 g/km and CO2 rating (correlation with combined fuel about 0.99 in magnitude, they would add no conflict); highway and city fuel (components of combined); cylinders (collinear with engine size)",
    }
    return out


def houses_run(source):
    df, prov, funnel = md.ensure_houses(DATA)
    old = previous_source().get("houses", {})
    if prov is None:
        if "access_date" not in old:
            raise RuntimeError(
                "houses file exists but its provenance was never recorded in m08-source.json"
            )
        prov, funnel = old, old["funnel"]
    path = DATA / "m08-quebec-houses.csv.gz"
    source["houses"] = {
        "dataset": "Rôle d'évaluation foncière du Québec, municipality 23027 (Ville de Québec), 2026 roll",
        "download_url": md.ROLL_URL,
        "index_url": md.ROLL_INDEX_URL,
        "licence": md.ROLL_LICENCE,
        "access_date": prov["access_date"],
        "raw_xml_bytes": prov["raw_xml_bytes"],
        "raw_xml_sha256": prov["raw_xml_sha256"],
        "raw_xml_committed": False,
        "derived_file": f"data/{path.name}",
        "derived_bytes": path.stat().st_size,
        "derived_sha256": md.sha256_of(path),
        "derived_rows": len(df),
        "derived_columns": list(df.columns),
        "funnel": funnel,
        "filter": "RL0105A = 1000 (predominant use: dwelling), RL0311A = 1 (one dwelling), RL0309A = 1 (detached), RL0201U = 1 (land owner, so no condominium units), no rooms for rent (RL0312A) and no commercial premises (RL0313A); then complete fields and plausible ranges",
        "not_an_asking_price": "value is RL0404A, the assessed value of land plus building at the roll's reference date, not an asking or sale price",
        "label_note": "address = civic number + generic + street name from RL0101; the link particle (de, du, des) is omitted and generic codes are expanded by inference (RU rue, AV avenue, BO boulevard, ...); unknown codes stay raw",
    }
    seg = df[
        (df["value"] <= 750_000)
        & (df["lot_m2"] <= 4000)
        & (df["year_built"] >= 1900)
        & (df["floor_m2"].between(60, 400))
    ].copy()
    seg["label"] = seg["address"] + " (#" + seg["dossier"].astype(str) + ")"
    crit = [
        ("value", "min"),
        ("floor_m2", "max"),
        ("year_built", "max"),
        ("lot_m2", "max"),
    ]
    spec = {
        "name": "houses",
        "df": seg,
        "id_col": "dossier",
        "criteria": crit,
        "corr_cols": ["value", "floor_m2", "year_built", "lot_m2"],
        "weights": {"value": 0.4, "floor_m2": 0.3, "year_built": 0.2, "lot_m2": 0.1},
        "eps": {
            "primary": "value",
            "bounds": {
                "floor_m2": (">=", 140),
                "year_built": (">=", 1990),
                "lot_m2": (">=", 500),
            },
            "sweep_col": "floor_m2",
            "sweep_op": ">=",
            "sweep_values": [80, 100, 120, 140, 160, 200, 250, 300],
        },
        "lex": {"order": ["year_built", "floor_m2", "value", "lot_m2"], "tol": 0.05},
        "goal": {"value": 350_000, "floor_m2": 140, "year_built": 2000, "lot_m2": 600},
        "k": 8,
        "seed": SEED,
        "full_naive": False,
        "plot": {
            "x": "value",
            "y": "floor_m2",
            "color": "year_built",
            "size": "lot_m2",
            "xlabel": "Assessed value (CAD), minimise",
            "ylabel": "Floor area (m2), maximise",
            "clabel": "Year built, maximise (marker size = lot area, maximise)",
        },
        # Text and layout of the two figures in the report (US spelling; see m08_pipeline.run). Their height
        # is about 10% under the cars figures' so that the two house figures fit one page with captions.
        "print": {
            "title": "Québec City detached houses",
            "xlabel": "Assessed value (CAD), minimize",
            "ylabel": "Floor area (m²), maximize",
            "clabel": "Year built, maximize (marker size = lot area, maximize)",
            "x_tick_step": 200_000,
            "pareto_ax_h": 2.6,
            "weights_ax_h": 2.5,
        },
        "fig_pareto": FIG / "m08-houses-pareto.png",
        "fig_weights": FIG / "m08-houses-weights.png",
        "fig_diversity": FIG / "m08-houses-diversity.png",
    }
    out = mpl.run(spec)
    out["segment"] = {
        "rule": "assessed value <= 750000 CAD, lot <= 4000 m2, built 1900 or later, floor area 60-400 m2",
        "why": "a buyer's segment; without it a few mansions and rural lots stretch the min-max scale so the other houses crowd near 0",
        "rows_in_derived_file": len(df),
        "rows_in_segment": len(seg),
    }
    out["criteria_rationale"] = {
        "value": "minimise: the stand-in for price; assessed value is not an asking price",
        "floor_m2": "maximise: gross floor area of the main building (includes attic and integrated garage in the roll's definition)",
        "year_built": "maximise: newer construction means less deferred maintenance; the roll gives the original build year, renovations are invisible",
        "lot_m2": "maximise: more garden, parking and room to extend, at the cost of upkeep; larger is not always better, which is why it carries the lowest weight",
        "excluded": "number of storeys (a design choice, not a quality); land value and building value (components of value); location, which the roll gives only as a neighbourhood code",
    }
    return out


def main():
    source = {
        "note": "regenerated by m08_compromise.py; access dates and the raw-XML hash are kept from the run that downloaded the data",
    }
    results = {"seed": SEED, "cars": cars_run(source), "houses": houses_run(source)}
    SOURCE_JSON.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n")
    (RES / "m08-results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    )
    for name in ("cars", "houses"):
        r = results[name]
        print(
            name,
            "rows",
            r["n_alternatives"],
            "pareto",
            r["pareto_size"],
            "distinct winners",
            r["weight_sensitivity"]["distinct_winners"],
            "never-winner (1000 / 400000)",
            r["weight_sensitivity"]["pareto_never_winner_in_1000"],
            r["weight_sensitivity"]["pareto_never_winner_in_400000"],
        )
        print("  picks", r["top_picks"])


if __name__ == "__main__":
    main()
