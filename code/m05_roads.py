"""CCCS 610, session 5 (Getting around: building roads), assignment 2.

Basic stage : the class province (30 random towns, MST + TSP additions) with a
              geometric-distribution population per town, drawn on a log scale.
In-depth    : five edge costs that mix distance and population, measured on the
              same 30 towns, on 200 random provinces, and on 40 real Quebec towns.

Run: python -I m05_roads.py
Outputs (all under the assignment-2 folder, names start with m05-):
  figures/m05-*.png, results/m05-results.json, results/m05-real-source.json,
  data/m05-quebec-municipalities.json (raw Wikidata responses, fetched once).
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # -I does not add the script folder
import legible as lg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import networkx as nx
import numpy as np
import requests
from matplotlib.collections import LineCollection

# ----------------------------------------------------------------------------
# Paths and parameters
# ----------------------------------------------------------------------------
D = Path(__file__).resolve().parent.parent
FIG, RES, DATA = D / "figures", D / "results", D / "data"
RAW_PATH = DATA / "m05-quebec-municipalities.json"
SOURCE_PATH = RES / "m05-real-source.json"
RESULTS_PATH = RES / "m05-results.json"

N_TOWNS = 30  # class value
GEO_P = 0.1  # geometric parameter: draws are in units of POP_UNIT inhabitants
POP_UNIT = 1000  # one geometric unit = 1000 inhabitants (mean 10 units = 10 000)
PART1_SEED = 0  # the single province shown in the figures (fixed in advance)
SEEDS = list(range(200))  # the 200 provinces of the repeated experiment
N_REAL = 40  # real towns kept
S_MIN, S_MAX = 25.0, 500.0  # marker area (points^2) of smallest / largest town
EARTH_R_KM = 6371.0088

VARIANTS = {
    "a_distance": ("(a) distance only (class)", lambda d, pi, pj: d),
    "b_sum": ("(b) distance / (pop_i + pop_j)", lambda d, pi, pj: d / (pi + pj)),
    "c_product": ("(c) distance / (pop_i * pop_j)", lambda d, pi, pj: d / (pi * pj)),
    "d_geomean": (
        "(d) distance / sqrt(pop_i * pop_j)",
        lambda d, pi, pj: d / np.sqrt(pi * pj),
    ),
    "e_gravity": (
        "(e) gravity: distance^2 / (pop_i * pop_j)",
        lambda d, pi, pj: d**2 / (pi * pj),
    ),
}
# Panel titles in the report's notation (mathtext); VARIANTS keeps the plain-text
# names that results/m05-results.json records.
TITLE_MATH = {
    "a_distance": "(a) distance only (class)",
    "b_sum": r"(b) $d/(p_i+p_j)$",
    "c_product": r"(c) $d/(p_i \times p_j)$",
    "d_geomean": r"(d) $d/\sqrt{p_i \times p_j}$",
    "e_gravity": r"(e) gravity: $d^2/(p_i \times p_j)$",
}
SHORT = {
    "a_distance": "a dist",
    "b_sum": "b /sum",
    "c_product": "c /prod",
    "d_geomean": "d /sqrt",
    "e_gravity": "e grav",
}

UA = "cccs610-assignment/1.0 (sebastian.vielmas@mail.mcgill.ca)"
SPARQL_URL = "https://query.wikidata.org/sparql"
LICENCE_URL = "https://www.wikidata.org/wiki/Wikidata:Licensing"
LICENCE_API = (
    "https://www.wikidata.org/w/api.php?action=parse&page=Wikidata:Licensing"
    "&prop=wikitext&format=json&formatversion=2"
)

QUERY_MAIN = """SELECT ?item ?label ?pop ?coord (GROUP_CONCAT(DISTINCT ?tl; separator=" | ") AS ?types) WHERE {
  SERVICE wikibase:box {
    ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:cornerSouthWest "Point(-80.0 44.9)"^^geo:wktLiteral .
    bd:serviceParam wikibase:cornerNorthEast "Point(-57.0 52.0)"^^geo:wktLiteral .
  }
  ?item p:P1082 ?st .
  ?st ps:P1082 ?pop ; pq:P585 ?pit .
  FILTER(YEAR(?pit) = 2021 && xsd:integer(?pop) > 25000)
  ?item wdt:P17 wd:Q16 .
  ?item rdfs:label ?label . FILTER(LANG(?label) = "fr")
  OPTIONAL { ?item wdt:P31 ?t . ?t rdfs:label ?tl . FILTER(LANG(?tl) = "fr") }
}
GROUP BY ?item ?label ?pop ?coord
ORDER BY DESC(xsd:integer(?pop))
LIMIT 400
"""

QUERY_AUDIT = """SELECT DISTINCT ?item ?label ?pop ?coord WHERE {
  ?item wdt:P31/wdt:P279* wd:Q3327873 .
  ?item p:P1082 ?st .
  ?st ps:P1082 ?pop ; pq:P585 ?pit .
  FILTER(YEAR(?pit) = 2021 && xsd:integer(?pop) > 25000)
  ?item wdt:P625 ?coord .
  ?item rdfs:label ?label . FILTER(LANG(?label) = "fr")
}
ORDER BY DESC(xsd:integer(?pop))
LIMIT 400
"""

# A row is a Quebec municipality if one of its Wikidata classes (French labels) is here ...
INCLUDE_TYPES = {
    "ville du Québec",
    "municipalité du Québec",
    "territoire hors MRC au Québec",
}
# ... unless it is also typed as a sub-municipal unit.
EXCLUDE_TYPES = {
    "quartier",
    "secteur résidentiel",
    "secteur municipal",
    "arrondissement de Montréal",
}

QID_MONTREAL, QID_QUEBEC = "Q340", "Q2145"


def to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(f"not serialisable: {type(o)}")


# ----------------------------------------------------------------------------
# Province model, costs, MST, TSP, measures
# ----------------------------------------------------------------------------
def make_province(seed: int, n: int = N_TOWNS):
    """Positions uniform in the unit square, populations POP_UNIT * Geometric(GEO_P)."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    pop = POP_UNIT * rng.geometric(GEO_P, n)
    return pos, pop


def dist_matrix(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def complete_graph(dist, pop, variant):
    """Complete graph; edge attribute `distance` is real, `weight` is the cost.

    Costs are divided by their median so that tiny products stay well scaled for the
    matching step of Christofides; a constant factor changes no ranking.
    """
    n = len(pop)
    cost_fn = VARIANTS[variant][1]
    iu = np.triu_indices(n, 1)
    cost = cost_fn(dist[iu], pop[iu[0]].astype(float), pop[iu[1]].astype(float))
    cost = cost / np.median(cost)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        g.add_edge(int(i), int(j), distance=float(dist[i, j]), weight=float(cost[k]))
    return g


def edge_key(u, v):
    return (u, v) if u < v else (v, u)


def mst_and_tsp(g):
    """Class method: MST on `weight`, then add the TSP-cycle edges that are missing."""
    mst = nx.minimum_spanning_tree(g, weight="weight")
    mst_edges = {edge_key(u, v) for u, v in mst.edges()}
    cycle = nx.approximation.traveling_salesman_problem(g, weight="weight")
    added = []
    for u, v in itertools.pairwise(cycle):
        k = edge_key(u, v)
        if u != v and k not in mst_edges and k not in added:
            added.append(k)
    visited = cycle[:-1]
    tsp_info = {
        "cycle_length": len(cycle) - 1,
        "distinct_towns": len(set(visited)),
        "repeated_visits": len(visited) - len(set(visited)),
        "hamiltonian": len(visited) == len(set(visited)) == g.number_of_nodes(),
    }
    return sorted(mst_edges), added, tsp_info


def ranks(pop, k=5):
    order = sorted(range(len(pop)), key=lambda i: (-pop[i], i))
    small = sorted(range(len(pop)), key=lambda i: (pop[i], i))
    return order[:k], small[:k]


def road_graph(edges, dist, n):
    r = nx.Graph()
    r.add_nodes_from(range(n))
    for u, v in edges:
        r.add_edge(u, v, distance=float(dist[u, v]))
    return r


def measure(edges, dist, n, top5, bot5):
    r = road_graph(edges, dist, n)
    lengths = [dist[u, v] for u, v in edges]
    total = float(np.sum(lengths))
    touching = float(sum(dist[u, v] for u, v in edges if u in top5 or v in top5))
    deg = dict(r.degree())
    connected = nx.is_connected(r)
    detour = float("nan")
    if connected:
        sp = dict(nx.all_pairs_dijkstra_path_length(r, weight="distance"))
        detour = float(
            np.mean([sp[i][j] / dist[i, j] for i in range(n) for j in range(i + 1, n)])
        )
    return {
        "n_roads": len(edges),
        "length": total,
        "length_share_touching_top5": touching / total if total > 0 else float("nan"),
        "top5_degree_mean": float(np.mean([deg[i] for i in top5])),
        "bottom5_degree_mean": float(np.mean([deg[i] for i in bot5])),
        "max_degree": int(max(deg.values())),
        "longest_road": float(max(lengths)),
        "connected": bool(connected),
        "mean_detour": detour,
    }


def run_variant(dist, pop, variant):
    n = len(pop)
    g = complete_graph(dist, pop, variant)
    mst_edges, added, tsp_info = mst_and_tsp(g)
    top5, bot5 = ranks(pop)
    network = sorted(set(mst_edges) | set(added))
    return {
        "mst": measure(mst_edges, dist, n, top5, bot5),
        "network": measure(network, dist, n, top5, bot5),
        "tsp_added": {
            "n_roads": len(added),
            "length": float(sum(dist[u, v] for u, v in added)),
        },
        "tsp": tsp_info,
        "_mst_edges": mst_edges,
        "_tsp_added": added,
    }


def flat(res):
    out = {}
    for stage in ("mst", "network"):
        for k, v in res[stage].items():
            out[f"{stage}.{k}"] = float(v)
    out["tsp_added.n_roads"] = float(res["tsp_added"]["n_roads"])
    out["tsp_added.length"] = res["tsp_added"]["length"]
    out["tsp.hamiltonian"] = float(res["tsp"]["hamiltonian"])
    out["tsp.repeated_visits"] = float(res["tsp"]["repeated_visits"])
    return out


def summarize(values):
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    q = np.quantile(a, [0.25, 0.5, 0.75])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
        "min": float(a.min()),
        "q25": float(q[0]),
        "median": float(q[1]),
        "q75": float(q[2]),
        "max": float(a.max()),
    }


def public(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------
# Population statistics (skew)
# ----------------------------------------------------------------------------
def skew_stats(pop):
    p = np.sort(np.asarray(pop, dtype=float))[::-1]
    n = len(p)
    med = float(np.median(p))
    m = p.mean()
    sd = p.std(ddof=0)
    ranks_ = np.arange(1, n + 1)
    slope = float(np.polyfit(np.log(ranks_), np.log(p), 1)[0])
    return {
        "n": n,
        "mean": float(m),
        "median": med,
        "min": float(p[-1]),
        "max": float(p[0]),
        "max_over_median": float(p[0] / med),
        "max_over_second": float(p[0] / p[1]),
        "top1_share": float(p[0] / p.sum()),
        "top5_share": float(p[:5].sum() / p.sum()),
        "coef_variation": float(sd / m),
        "skewness": float(((p - m) ** 3).mean() / sd**3),
        "excess_kurtosis": float(((p - m) ** 4).mean() / sd**4 - 3),
        "share_of_towns_above_2x_mean": float((p > 2 * m).mean()),
        "loglog_rank_size_slope": slope,
    }


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def node_sizes(pop):
    """Marker area (points^2) = S_MIN + (S_MAX - S_MIN) * (log10(pop) - log10(pop_min)) /
    (log10(pop_max) - log10(pop_min)): the area grows linearly with log10 of the population."""
    lp = np.log10(np.asarray(pop, dtype=float))
    lo, hi = lp.min(), lp.max()
    if hi == lo:
        return np.full(len(lp), S_MIN)
    return S_MIN + (S_MAX - S_MIN) * (lp - lo) / (hi - lo)


def size_legend(
    ax, pop, anchor_y=-0.03, fontsize=7, columnspacing=2.5, handleheight=0.7
):
    """Three reference circles under the axes, outside the map.

    handleheight (in font sizes) must make the row at least as tall as the biggest circle, or that circle
    spills out of the frame and the crop clips it.
    """
    pop = np.asarray(pop, dtype=float)
    refs = [pop.min(), math.sqrt(pop.min() * pop.max()), pop.max()]
    sizes = node_sizes(np.array([pop.min(), pop.max()] + refs))[2:]
    handles = [
        ax.scatter([], [], s=s, c="lightgray", edgecolors="k", label=f"{round(r):,}")
        for r, s in zip(refs, sizes)
    ]
    ax.legend(
        handles=handles,
        title="population",
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=3,
        fontsize=fontsize,
        title_fontsize=fontsize,
        columnspacing=columnspacing,
        handletextpad=1.2,
        handleheight=handleheight,
        borderpad=0.8,
        framealpha=0.9,
        scatterpoints=1,
    )


def assert_indices_inside_own_circle(fig, ax, pos, pop):
    """Each drawn town number lies inside its own circle (number against number is checked by assert_legible)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    centres = ax.transData.transform(pos)
    radii = np.sqrt(node_sizes(pop)) / 2 * pt
    for t in ax.texts:
        if not t.get_text().isdigit():
            continue
        i = int(t.get_text())
        b = mtext.Text.get_window_extent(t, renderer)
        corners = [(b.x0, b.y0), (b.x0, b.y1), (b.x1, b.y0), (b.x1, b.y1)]
        if any(np.hypot(*(np.array(c) - centres[i])) > radii[i] for c in corners):
            raise AssertionError(f"town number {i} does not fit inside its circle")


def assert_titles_clear(fig, axes):
    """No panel title reaches into another panel's frame."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.ravel():
        t = mtext.Text.get_window_extent(ax.title, renderer)
        for other in axes.ravel():
            if other is not ax:
                b = other.get_window_extent(renderer)
                if lg.rects_overlap((t.x0, t.y0, t.x1, t.y1), (b.x0, b.y0, b.x1, b.y1)):
                    raise AssertionError("a panel title overlaps another panel's frame")


def draw_roads(ax, pos, mst_edges, tsp_edges, lw_mst=1.4, lw_tsp=2.6):
    if mst_edges:
        ax.add_collection(
            LineCollection(
                [[pos[u], pos[v]] for u, v in mst_edges],
                colors="black",
                linewidths=lw_mst,
                zorder=1,
            )
        )
    if tsp_edges:
        ax.add_collection(
            LineCollection(
                [[pos[u], pos[v]] for u, v in tsp_edges],
                colors="orange",
                linewidths=lw_tsp,
                zorder=2,
            )
        )


RING_LW = 1.8  # points: the red ring of a biggest town
EDGE_LW = 0.8  # points: the black outline of a town


def draw_province(
    ax,
    pos,
    pop,
    mst_edges=(),
    tsp_edges=(),
    index_labels=True,
    top5=(),
    title=None,
    title_size=8,
    index_size=5,
    index_only=None,
    index_halo=None,
    lift_numbered=False,
):
    """Roads, then the town circles, then the town numbers centred on their towns.

    With lift_numbered the numbered towns are redrawn filled above the others and each number may move
    off its centre, so a neighbour's outline never runs through a number (see `place_indices`).
    """
    draw_roads(ax, pos, list(mst_edges), list(tsp_edges))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=title_size)
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=node_sizes(pop),
        c="lightgray",
        edgecolors="k",
        linewidths=EDGE_LW,
        zorder=3,
    )
    if not lift_numbered:
        if len(top5):
            ax.scatter(
                pos[list(top5), 0],
                pos[list(top5), 1],
                s=node_sizes(pop)[list(top5)],
                facecolors="none",
                edgecolors="crimson",
                linewidths=RING_LW,
                zorder=4,
            )
        if index_labels:
            for i, (x, y) in enumerate(pos):
                if index_only is None or i in index_only:
                    ax.annotate(
                        str(i),
                        (x, y),
                        ha="center",
                        va="center",
                        fontsize=index_size,
                        zorder=5,
                        path_effects=index_halo,
                    )
        return
    numbered = lift_order(pop, index_only) if index_labels else []
    ring = set(top5)
    for k, i in enumerate(numbered):
        ax.scatter(
            [pos[i, 0]],
            [pos[i, 1]],
            s=node_sizes(pop)[i],
            c="lightgray",
            edgecolors="crimson" if i in ring else "k",
            linewidths=RING_LW if i in ring else EDGE_LW,
            zorder=4 + 0.01 * k,
        )
    for i in ring - set(numbered):
        ax.scatter(
            [pos[i, 0]],
            [pos[i, 1]],
            s=node_sizes(pop)[i],
            facecolors="none",
            edgecolors="crimson",
            linewidths=RING_LW,
            zorder=4,
        )


def lift_order(pop, index_only=None):
    """Drawing order of the numbered towns, lowest first: the smaller circle goes on top, so the bigger
    circle keeps the room for its own number."""
    towns = range(len(pop)) if index_only is None else index_only
    return sorted(towns, key=lambda i: (-pop[i], i))


def place_indices(ax, pos, pop, numbered, ring, size, halo):
    """Write each town number inside its own circle, clear of every circle drawn above it.

    Call it once the layout is final: it measures the axes in pixels. `numbered` is `lift_order`. A
    number is moved off the centre, by the least distance, until its box lies inside its own circle
    (minus the outline) and outside every later circle's outline; a town with no such place raises.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    pt = ax.figure.dpi / 72
    centres = ax.transData.transform(pos)
    radii = np.sqrt(node_sizes(pop)) / 2 * pt
    lw = {i: (RING_LW if i in ring else EDGE_LW) * pt for i in numbered}
    probe = ax.text(0, 0, "", fontsize=size)
    for k, i in enumerate(numbered):
        probe.set_text(str(i))
        e = mtext.Text.get_window_extent(probe, renderer)
        hw, hh = e.width / 2, e.height / 2
        above = numbered[k + 1 :]
        best = None
        step = 0.25 * pt
        reach = radii[i] / step
        for ix in np.arange(-reach, reach + 1):
            for iy in np.arange(-reach, reach + 1):
                dx, dy = ix * step, iy * step
                c = centres[i] + (dx, dy)
                corners = [
                    (c[0] + sx * hw, c[1] + sy * hh) for sx in (-1, 1) for sy in (-1, 1)
                ]
                if any(
                    np.hypot(*(np.array(q) - centres[i]))
                    > radii[i] - lw[i] / 2 - 0.5 * pt
                    for q in corners
                ):
                    continue
                box = (c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh)
                if any(
                    lg.rect_hits_circle(
                        box, centres[j], radii[j] + lw[j] / 2 + 0.5 * pt
                    )
                    for j in above
                ):
                    continue
                d = np.hypot(dx, dy)
                if best is None or d < best[0]:
                    best = (d, c)
        if best is None:
            raise RuntimeError(f"no place for the number of town {i} inside its circle")
        x, y = ax.transData.inverted().transform(best[1])
        ax.text(
            x,
            y,
            str(i),
            ha="center",
            va="center",
            fontsize=size,
            zorder=5 + 0.01 * k,
            path_effects=halo,
        )
    probe.remove()


# ----------------------------------------------------------------------------
# Real data (Wikidata) fetch / cache
# ----------------------------------------------------------------------------
def sparql(query):
    last = None
    for attempt in (1, 2):
        try:
            resp = requests.get(
                SPARQL_URL,
                params={"query": query},
                headers={"Accept": "application/sparql-results+json", "User-Agent": UA},
                timeout=180,
            )
            if resp.status_code == 200:
                return resp.json(), 200
            last = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(5)
    raise RuntimeError(f"SPARQL request failed twice at {SPARQL_URL}: {last}")


def fetch_raw():
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    main, status_main = sparql(QUERY_MAIN)
    raw = {
        "access_date_utc": stamp,
        "endpoint": SPARQL_URL,
        "query_main": QUERY_MAIN,
        "http_status_main": status_main,
        "main": main,
    }
    try:
        audit, status_audit = sparql(QUERY_AUDIT)
        raw.update({
            "query_audit": QUERY_AUDIT,
            "http_status_audit": status_audit,
            "audit_class": audit,
        })
    except RuntimeError as exc:
        raw.update({
            "query_audit": QUERY_AUDIT,
            "audit_class": None,
            "audit_error": str(exc),
        })
    lic = requests.get(LICENCE_API, headers={"User-Agent": UA}, timeout=60)
    lic.raise_for_status()
    text = lic.json()["parse"]["wikitext"]
    line = next(
        (
            ln
            for ln in text.splitlines()
            if ln.startswith("All data in Wikidata has a CC0 license.")
        ),
        None,
    )
    if line is None:
        raise RuntimeError(f"licence sentence not found on {LICENCE_API}")
    raw["licence"] = {
        "url": LICENCE_URL,
        "retrieved_from": LICENCE_API,
        "text": line[:200].split(" However,")[0],
    }
    RAW_PATH.write_text(
        json.dumps(raw, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return raw


def load_raw():
    if RAW_PATH.exists():
        return json.loads(RAW_PATH.read_text(encoding="utf-8"))
    return fetch_raw()


def parse_point(wkt):
    m = re.fullmatch(r"Point\(([-\d.]+) ([-\d.]+)\)", wkt)
    if not m:
        raise ValueError(f"unexpected coordinate literal {wkt!r}")
    return float(m.group(1)), float(m.group(2))


def select_towns(raw):
    """Filter the box query to Quebec municipalities and deduplicate by Wikidata item."""
    bindings = raw["main"]["results"]["bindings"]
    by_qid, drops = {}, []
    counts = {"response_rows": len(bindings)}
    for b in bindings:
        qid = b["item"]["value"].rsplit("/", 1)[1]
        types = {
            t.strip()
            for t in b.get("types", {}).get("value", "").split(" | ")
            if t.strip()
        }
        row = {
            "qid": qid,
            "name": b["label"]["value"],
            "pop": int(b["pop"]["value"]),
            "wkt": b["coord"]["value"],
            "types": sorted(types),
        }
        if not types & INCLUDE_TYPES:
            drops.append({
                **row,
                "reason": "no Quebec-municipality class (region, county, riding, agglomeration, other province)",
            })
            continue
        if types & EXCLUDE_TYPES:
            drops.append({
                **row,
                "reason": "sub-municipal unit: "
                + ", ".join(sorted(types & EXCLUDE_TYPES)),
            })
            continue
        if qid in by_qid:
            by_qid[qid]["wkts"].add(row["wkt"])
            continue
        by_qid[qid] = {**row, "wkts": {row["wkt"]}}
    counts["distinct_items_in_response"] = len({b["item"]["value"] for b in bindings})
    counts["rows_dropped_not_municipality"] = sum(
        1 for d in drops if d["reason"].startswith("no Quebec")
    )
    counts["rows_dropped_sub_municipal"] = sum(
        1 for d in drops if d["reason"].startswith("sub-municipal")
    )
    counts["quebec_municipalities_after_dedup_by_item"] = len(by_qid)
    counts["items_with_two_coordinate_values"] = sorted(
        q for q, r in by_qid.items() if len(r["wkts"]) > 1
    )
    towns = []
    for qid, r in by_qid.items():
        lon, lat = parse_point(
            min(r["wkts"])
        )  # deterministic choice among several P625 values
        towns.append({
            "qid": qid,
            "name": r["name"],
            "pop": r["pop"],
            "lon": lon,
            "lat": lat,
            "types": r["types"],
        })
    towns.sort(key=lambda t: (-t["pop"], t["qid"]))
    top = towns[:N_REAL]
    cutoff = top[-1]["pop"]
    counts["kept_top"] = len(top)
    pops = [t["pop"] for t in top]
    dup_pops = sorted({p for p in pops if pops.count(p) > 1})
    near_drops = [d for d in drops if d["pop"] >= cutoff]
    return top, towns, counts, near_drops, dup_pops


def project_km(towns):
    lon = np.array([t["lon"] for t in towns])
    lat = np.array([t["lat"] for t in towns])
    lon0, lat0 = lon.mean(), lat.mean()
    x = EARTH_R_KM * np.cos(np.radians(lat0)) * np.radians(lon - lon0)
    y = EARTH_R_KM * np.radians(lat - lat0)
    return np.column_stack([x, y]), float(lon0), float(lat0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def part1(results):
    pos, pop = make_province(PART1_SEED)
    stats = skew_stats(pop)
    print(
        f"[part 1] seed {PART1_SEED}: n={N_TOWNS}, p={GEO_P}, unit={POP_UNIT}; "
        f"mean={stats['mean']:.0f}, max={int(stats['max'])}, median={stats['median']:.0f}, "
        f"max/median={stats['max_over_median']:.2f}"
    )

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    draw_province(
        ax,
        pos,
        pop,
        title=f"Synthetic province: {N_TOWNS} towns, area ~ log10(population)",
    )
    size_legend(ax, pop)
    fig.savefig(FIG / "m05-basic-province.png", bbox_inches="tight")
    plt.close(fig)

    dist = dist_matrix(pos)
    res = run_variant(dist, pop, "a_distance")
    # Printed about 6.3 in wide: 7 pt town numbers, only on the five biggest towns, whose circles hold them.
    top5_fig1, _ = ranks(pop)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    draw_province(
        ax,
        pos,
        pop,
        res["_mst_edges"],
        res["_tsp_added"],
        title="Class MST (black) + TSP additions (orange), distance cost",
        index_size=7,
        index_only=set(top5_fig1),
        lift_numbered=True,
    )
    size_legend(ax, pop, fontsize=8, handleheight=3.4)
    place_indices(ax, pos, pop, lift_order(pop, set(top5_fig1)), set(), 7, None)
    assert_indices_inside_own_circle(fig, ax, pos, pop)
    lg.assert_legible(fig, "m05-basic-mst-tsp.png")
    lg.assert_fits(fig, "m05-basic-mst-tsp.png", 6.5, 0.1)
    fig.savefig(FIG / "m05-basic-mst-tsp.png", bbox_inches="tight")
    plt.close(fig)

    # skew over the 200 provinces, for 30 towns and for 40 towns (to match the real set)
    per_seed = {
        n: [skew_stats(make_province(s, n)[1]) for s in SEEDS]
        for n in (N_TOWNS, N_REAL)
    }
    agg = {
        n: {k: summarize([d[k] for d in v]) for k in v[0] if k != "n"}
        for n, v in per_seed.items()
    }
    results["part1"] = {
        "seed": PART1_SEED,
        "n_towns": N_TOWNS,
        "positions_xy": np.round(pos, 6).tolist(),
        "populations": pop.tolist(),
        "geometric_p": GEO_P,
        "population_unit_inhabitants": POP_UNIT,
        "choice_of_p": (
            "numpy Generator.geometric(p) returns k = 1, 2, 3, ... with mean 1/p, median about ln2/p and "
            "an exponentially thin tail. p = 0.1 gives a mean of 10 units; one unit is read as 1000 "
            "inhabitants, so the mean town has 10 000, the smallest possible has 1000 and the "
            "largest of 30 is typically near 35 000. The scale only multiplies all populations by one "
            "constant, which changes no MST or TSP result for any of the five costs and none of the "
            "ratios below."
        ),
        "marker_area_formula": (
            "area(points^2) = 25 + 475 * (log10(pop) - log10(pop_min)) / (log10(pop_max) - log10(pop_min)); "
            "area is linear in log10(population) and spans 25 (smallest town of the figure) to 500 "
            "(largest); the legend prints three reference populations."
        ),
        "summary_this_province": stats,
        "theory_geometric": {
            "ratio_max_over_median_approx": (
                f"(ln n + 0.577) / ln 2 = {(math.log(N_TOWNS) + 0.5772) / math.log(2):.2f} for n={N_TOWNS}"
            ),
            "tail": "P(X > x) = (1-p)^x, exponential decay: constant mean excess, excess kurtosis ~ 6, no power-law tail",
        },
        "over_200_seeds": {f"n={n}": a for n, a in agg.items()},
        "single_province_stage_a": {
            "mst": res["mst"],
            "network": res["network"],
            "tsp_added": res["tsp_added"],
            "tsp": res["tsp"],
        },
    }
    return pos, pop, dist


def part2(results, pos, pop, dist):
    # --- one province (the part-1 towns) ---
    single = {v: run_variant(dist, pop, v) for v in VARIANTS}
    top5, bot5 = ranks(pop)
    results["part2_single_province"] = {
        "seed": PART1_SEED,
        "top5_towns": top5,
        "bottom5_towns": bot5,
        "variants": {
            v: public(r)
            | {"mst_edges": r["_mst_edges"], "tsp_added_edges": r["_tsp_added"]}
            for v, r in single.items()
        },
    }
    for v, r in single.items():
        print(
            f"[part 2, seed 0] {v:11s} MST len {r['mst']['length']:.3f} top5deg {r['mst']['top5_degree_mean']:.2f} "
            f"| network len {r['network']['length']:.3f} top5deg {r['network']['top5_degree_mean']:.2f} "
            f"detour {r['network']['mean_detour']:.3f} | TSP repeats {r['tsp']['repeated_visits']}"
        )

    # Drawn at the width it prints at (6.5 in): 7 pt town numbers, only on the five biggest towns, whose
    # circles are large enough to hold them; the smaller circles are too small for 7 pt digits.
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 7.4), dpi=200)
    for ax, v in zip(axes.ravel(), list(VARIANTS)[:4]):
        r = single[v]
        t = (
            f"{TITLE_MATH[v]}\nMST+TSP: {r['network']['n_roads']} roads, "
            f"length {r['network']['length']:.2f},\n"
            f"top-5 mean degree {r['network']['top5_degree_mean']:.1f}, "
            f"max degree {r['network']['max_degree']}"
        )
        draw_province(
            ax,
            pos,
            pop,
            r["_mst_edges"],
            r["_tsp_added"],
            top5=top5,
            title=t,
            index_only=set(top5),
            lift_numbered=True,
        )
    size_legend(axes[1, 0], pop, fontsize=8, columnspacing=1.2, handleheight=3.4)
    fig.suptitle(
        "Same 30 towns and populations, four edge costs. Black: MST.\n"
        "Orange: TSP additions. Red ring: the 5 largest towns.\n"
        "Length is real distance, not cost.",
        fontsize=8,
    )
    fig.tight_layout(h_pad=2.5)
    halo = [pe.withStroke(linewidth=1.6, foreground="lightgray")]
    for ax in axes.ravel():
        place_indices(ax, pos, pop, lift_order(pop, set(top5)), set(top5), 7, halo)
        assert_indices_inside_own_circle(fig, ax, pos, pop)
    assert_titles_clear(fig, axes)
    lg.assert_legible(fig, "m05-indepth-costs.png")
    lg.assert_fits(fig, "m05-indepth-costs.png", 6.5, 0.04)
    fig.savefig(FIG / "m05-indepth-costs.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # --- 200 provinces ---
    table = {v: {} for v in VARIANTS}
    for s in SEEDS:
        p_, q_ = make_province(s)
        d_ = dist_matrix(p_)
        for v in VARIANTS:
            for k, val in flat(run_variant(d_, q_, v)).items():
                table[v].setdefault(k, []).append(val)
    summary = {
        v: {k: summarize(vals) for k, vals in table[v].items()} for v in VARIANTS
    }
    paired = {}
    for v in VARIANTS:
        if v == "a_distance":
            continue
        paired[v] = {}
        for k in table[v]:
            diff = np.array(table[v][k]) - np.array(table["a_distance"][k])
            diff = diff[~np.isnan(diff)]
            paired[v][k] = {
                "mean_diff_vs_a": float(diff.mean()),
                "sd_diff": float(diff.std(ddof=1)),
                "share_seeds_greater": float((diff > 0).mean()),
                "share_seeds_less": float((diff < 0).mean()),
            }
    results["part2_200_seeds"] = {
        "seeds": [SEEDS[0], SEEDS[-1]],
        "n_provinces": len(SEEDS),
        "summary": summary,
        "paired_vs_a_distance": paired,
    }

    panels = [
        ("mst.top5_degree_mean", "MST: mean degree of the 5 biggest towns"),
        ("network.top5_degree_mean", "MST+TSP: mean degree of the 5 biggest towns"),
        ("mst.mean_detour", "MST: mean detour (path / straight line)"),
        ("network.mean_detour", "MST+TSP: mean detour (path / straight line)"),
        ("network.length", "MST+TSP: total real road length"),
        (
            "network.length_share_touching_top5",
            "MST+TSP: share of length touching the 5 biggest",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    for ax, (k, title) in zip(axes.ravel(), panels):
        ax.boxplot(
            [table[v][k] for v in VARIANTS],
            tick_labels=[SHORT[v] for v in VARIANTS],
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "crimson",
                "markeredgecolor": "crimson",
            },
        )
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"{len(SEEDS)} random provinces (seeds {SEEDS[0]}-{SEEDS[-1]}), 30 towns each; "
        "box = quartiles, whiskers 1.5 IQR, red diamond = mean",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG / "m05-indepth-seeds.png", bbox_inches="tight")
    plt.close(fig)
    for v in VARIANTS:
        s = summary[v]
        print(
            f"[part 2, 200 seeds] {v:11s} MST top5deg {s['mst.top5_degree_mean']['mean']:.2f}"
            f"+-{s['mst.top5_degree_mean']['sd']:.2f} | net top5deg {s['network.top5_degree_mean']['mean']:.2f}"
            f"+-{s['network.top5_degree_mean']['sd']:.2f} | net detour {s['network.mean_detour']['mean']:.3f}"
            f"+-{s['network.mean_detour']['sd']:.3f} | net length {s['network.length']['mean']:.2f}"
            f"+-{s['network.length']['sd']:.2f}"
        )


def stage_graph(res, stage):
    edges = (
        res["_mst_edges"]
        if stage == "mst"
        else sorted(set(res["_mst_edges"]) | set(res["_tsp_added"]))
    )
    return edges


def route_report(edges, dist, names, n, i_mtl, i_qc):
    r = road_graph(edges, dist, n)
    path = nx.shortest_path(r, i_mtl, i_qc, weight="distance")
    length = nx.shortest_path_length(r, i_mtl, i_qc, weight="distance")
    straight = float(dist[i_mtl, i_qc])
    bc = nx.betweenness_centrality(r, weight="distance", normalized=True)
    deg = dict(r.degree())
    hub = max(range(n), key=lambda i: (deg[i], -i))
    return {
        "montreal_quebec_path": [names[i] for i in path],
        "montreal_quebec_hops": len(path) - 1,
        "montreal_quebec_path_km": float(length),
        "montreal_quebec_straight_km": straight,
        "montreal_quebec_detour": float(length / straight),
        "degree_montreal": int(deg[i_mtl]),
        "degree_quebec": int(deg[i_qc]),
        "betweenness_montreal": float(bc[i_mtl]),
        "betweenness_quebec": float(bc[i_qc]),
        "highest_degree_town": names[hub],
        "highest_degree": int(deg[hub]),
        "towns_with_degree_1": sum(1 for i in range(n) if deg[i] == 1),
    }


LABEL_OFFSETS = {  # (dx, dy) in points, horizontal alignment
    "Montréal": (-55, -38, "right"),
    "Laval": (-60, 30, "right"),
    "Longueuil": (15, -45, "right"),
    "Terrebonne": (35, 48, "left"),
    "Québec": (-45, 30, "right"),
    "Lévis": (40, -32, "left"),
    "Gatineau": (-30, 26, "right"),
    "Sherbrooke": (35, -30, "left"),
    "Saguenay": (-40, -30, "right"),
    "Trois-Rivières": (-50, 26, "right"),
}


QUEBEC_LABELS = 8  # towns named in the real-data figure, biggest first


def draw_quebec(ax, xy, pop, names, edges_mst, edges_tsp, title, label_top=10):
    draw_roads(ax, xy, edges_mst, edges_tsp, lw_mst=1.6, lw_tsp=2.8)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_sizes(pop),
        c="lightgray",
        edgecolors="k",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_aspect("equal")
    ax.margins(x=0.05, y=0.3)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("km east of centre", fontsize=8)
    ax.set_ylabel("km north of centre", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.2)


def label_quebec(fig, ax, xy, pop, names, label_top=10, size=7.5):
    """Name the biggest towns once the layout is final.

    The map is about 3 in wide, so every label takes the first offset (the hand-set one, scaled down, then
    rings of growing radius) where its box touches no circle, no other label, and its leader crosses no other
    leader. A label with no such place raises.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    box = ax.get_window_extent()
    centres = ax.transData.transform(xy)
    radii = (np.sqrt(node_sizes(pop)) / 2 + 0.5) * pt
    pad = 1.5 * pt
    probe = ax.text(0, 0, "", fontsize=size, fontweight="bold")
    placed = []  # (rect, leader segment)
    order = sorted(range(len(pop)), key=lambda i: (-pop[i], i))[:label_top]
    for i in order:
        probe.set_text(names[i])
        e = mtext.Text.get_window_extent(probe, renderer)
        w, h = e.width + 2 * pad, e.height + 2 * pad
        cands = []
        if names[i] in LABEL_OFFSETS:
            dx, dy, _ = LABEL_OFFSETS[names[i]]
            cands.append((0.6 * dx, 0.6 * dy))
        for r in (12, 18, 26, 36, 48, 62, 80, 100):
            for ang in range(0, 360, 15):
                cands.append((r * np.cos(np.radians(ang)), r * np.sin(np.radians(ang))))
        spot = None
        # First choice: no leader crosses another leader or runs through another label. If the cluster leaves
        # no such place, a leader may cross another leader; no label ever covers a circle or another label.
        for strict in (True, False):
            for dx, dy in cands:
                ha = "left" if dx >= 0 else "right"
                ax_ = centres[i][0] + dx * pt
                cy = centres[i][1] + dy * pt
                x0, x1 = (
                    (ax_ - pad, ax_ + w - pad)
                    if ha == "left"
                    else (ax_ - w + pad, ax_ + pad)
                )
                rect = (x0, cy - h / 2, x1, cy + h / 2)
                if not (
                    box.x0 < rect[0]
                    and rect[2] < box.x1
                    and box.y0 < rect[1]
                    and rect[3] < box.y1
                ):
                    continue
                if any(
                    lg.rect_hits_circle(rect, centres[j], radii[j])
                    for j in range(len(pop))
                ):
                    continue
                seg = (np.array([(x0 + x1) / 2, cy]), centres[i])
                if any(
                    lg.rects_overlap(rect, o)
                    or lg.seg_hits_rect(*seg, o)
                    or lg.seg_hits_rect(*s, rect)
                    or (strict and lg.segs_cross(*seg, *s))
                    for o, s in placed
                ):
                    continue
                spot = (dx, dy, ha, rect, seg)
                break
            if spot:
                break
        if spot is None:
            raise RuntimeError(f"no free place for the label {names[i]}")
        dx, dy, ha, rect, seg = spot
        placed.append((rect, seg))
        ax.annotate(
            names[i],
            xy[i],
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=size,
            fontweight="bold",
            zorder=6,
            arrowprops={"arrowstyle": "-", "color": "0.2", "lw": 0.7},
            bbox={
                "boxstyle": "round,pad=0.12",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
        )
    probe.remove()


def part3(results, geo_summary):
    raw = load_raw()
    top, towns_all, counts, near_drops, dup_pops = select_towns(raw)
    names = [t["name"] for t in top]
    pop = np.array([t["pop"] for t in top])
    xy, lon0, lat0 = project_km(top)
    dist = dist_matrix(xy)
    qids = [t["qid"] for t in top]
    i_mtl, i_qc = qids.index(QID_MONTREAL), qids.index(QID_QUEBEC)
    n = len(top)

    # audit: items the class-based query returns vs. the box query result
    audit = {"available": raw.get("audit_class") is not None}
    if audit["available"]:
        cls = {
            b["item"]["value"].rsplit("/", 1)[1]
            for b in raw["audit_class"]["results"]["bindings"]
        }
        audit["class_Q3327873_items_over_25000"] = len(cls)
        audit["kept_towns_missing_from_class_query"] = [
            f"{t['qid']} {t['name']}" for t in top if t["qid"] not in cls
        ]
        audit["class_query_items_not_kept"] = sorted(
            cls - {t["qid"] for t in towns_all}
        )
    else:
        audit["error"] = raw.get("audit_error")

    # near-coordinate pairs (flagged, not removed)
    near_pairs = [
        [names[i], names[j], round(float(dist[i, j]), 2)]
        for i in range(n)
        for j in range(i + 1, n)
        if dist[i, j] < 8.0
    ]

    source = {
        "purpose": "Population and coordinates of Quebec municipalities for the real-data run of m05_roads.py",
        "source": "Wikidata (https://www.wikidata.org), SPARQL endpoint " + SPARQL_URL,
        "licence": raw["licence"],
        "licence_line": f'CC0 1.0: "{raw["licence"]["text"]}" ({raw["licence"]["url"]})',
        "access_date_utc": raw["access_date_utc"],
        "http_status_main": raw["http_status_main"],
        "query_main_exact": raw["query_main"],
        "query_audit_exact": raw["query_audit"],
        "population_property": "P1082 with point in time P585 in 2021 (2021 census as entered in Wikidata)",
        "coordinate_property": "P625",
        "brief_defect": "The suggested class Q3374262 is 'miasteczko' (a Polish small-town class): the query with it "
        "returned Romanian and Ukrainian towns. The class of municipalities of Quebec is Q3327873 "
        "('local municipality of Quebec'); it is used in the audit query only (see below).",
        "why_box_query": "The Q3327873 class query omits Cote Saint-Luc (Q139588, typed only 'ville' and "
        "'territoire hors MRC au Quebec'), which lies inside the top 40. The main query "
        "therefore selects by a geographic box and a 2021 population over 25 000, and the Quebec "
        "municipalities are then picked out by their Wikidata classes in this script.",
        "include_rule_any_class": sorted(INCLUDE_TYPES),
        "exclude_rule_any_class": sorted(EXCLUDE_TYPES),
        "row_counts": counts,
        "rows_in_audit_class_query": (
            len(raw["audit_class"]["results"]["bindings"])
            if audit["available"]
            else None
        ),
        "audit_vs_class_query": audit,
        "dropped_rows_that_would_have_ranked_in_top_40": [
            {k: d[k] for k in ("qid", "name", "pop", "types", "reason")}
            for d in near_drops
        ],
        "duplicates_checked": {
            "same_item_repeated_in_response": "collapsed by Wikidata item (QID); items with two P625 values take the "
            "lexicographically smallest coordinate literal",
            "identical_population_values_among_kept_towns": dup_pops,
            "boroughs_vs_city": "borough items (arrondissements de Montreal, secteurs of Gatineau) and the "
            "administrative regions/agglomerations carrying the same name are dropped by class; "
            "the municipal 'ville' item is kept, so Montreal counts once at its municipal "
            "population, not once per borough.",
            "towns_closer_than_8_km_kept_as_distinct_municipalities": near_pairs,
        },
        "projection": {
            "type": "equirectangular",
            "lon0_mean": lon0,
            "lat0_mean": lat0,
            "radius_km": EARTH_R_KM,
            "formula": "x = R*cos(lat0)*(lon-lon0), y = R*(lat-lat0), angles in radians",
        },
        "towns": [
            {
                "rank": i + 1,
                "qid": t["qid"],
                "name": t["name"],
                "pop_2021": t["pop"],
                "lon": t["lon"],
                "lat": t["lat"],
                "x_km": float(xy[i, 0]),
                "y_km": float(xy[i, 1]),
            }
            for i, t in enumerate(top)
        ],
        "raw_file": str(RAW_PATH.relative_to(D)),
    }
    SOURCE_PATH.write_text(
        json.dumps(source, ensure_ascii=False, indent=1, default=to_jsonable) + "\n",
        encoding="utf-8",
    )
    print(
        f"[part 3] {n} real towns, cutoff {pop.min():,}; audit missing from class query: "
        f"{audit.get('kept_towns_missing_from_class_query')}"
    )

    real_stats = skew_stats(pop)
    # --- experiment on real towns ---
    out = {v: run_variant(dist, pop, v) for v in VARIANTS}
    real = {}
    for v, r in out.items():
        real[v] = public(r)
        for stage in ("mst", "network"):
            real[v][stage + "_route"] = route_report(
                stage_graph(r, stage), dist, names, n, i_mtl, i_qc
            )
        real[v]["mst_edges"] = [[names[u], names[w]] for u, w in r["_mst_edges"]]
        real[v]["tsp_added_edges"] = [[names[u], names[w]] for u, w in r["_tsp_added"]]
        rt = real[v]["mst_route"]
        print(
            f"[part 3] {v:11s} MST len {r['mst']['length']:.0f} km, top5 deg {r['mst']['top5_degree_mean']:.1f}, "
            f"MTL-QC {rt['montreal_quebec_path_km']:.0f} km (straight {rt['montreal_quebec_straight_km']:.0f}) "
            f"via {' > '.join(rt['montreal_quebec_path'])}"
        )

    # selection rule for the figure: pop-aware variants that raise the MST top-5 degree above (a);
    # among them the one with the smallest total real MST length.
    base_deg = out["a_distance"]["mst"]["top5_degree_mean"]
    cands = [
        v
        for v in VARIANTS
        if v != "a_distance" and out[v]["mst"]["top5_degree_mean"] > base_deg
    ]
    if cands:
        best = min(cands, key=lambda v: out[v]["mst"]["length"])
        rule = "among cost variants (b)-(e) whose MST raises the mean degree of the 5 biggest towns above variant (a), the smallest total real MST length"
    else:
        best = max(
            (v for v in VARIANTS if v != "a_distance"),
            key=lambda v: out[v]["mst"]["top5_degree_mean"],
        )
        rule = "no variant raised the top-5 degree; the one with the highest top-5 degree is shown"

    # Drawn at the width it prints at (6.5 in), the two maps side by side, so each is only about 2.4 in wide:
    # the biggest towns are named at 7.5 pt (the rest are circles only).
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.4), dpi=200)
    for ax, v in zip(axes, ("a_distance", best)):
        r = out[v]
        rt = real[v]["mst_route"]
        t = (
            f"{TITLE_MATH[v]}  (MST, black)\nMST length {r['mst']['length']:.0f} km;\n"
            f"Montréal-Québec by road {rt['montreal_quebec_path_km']:.0f} km\n"
            f"(straight line {rt['montreal_quebec_straight_km']:.0f} km)"
        )
        draw_quebec(ax, xy, pop, names, r["_mst_edges"], [], t)
    size_legend(axes[0], pop, anchor_y=-0.2, handleheight=3.6)
    fig.suptitle(
        r"40 most populous Québec municipalities (Wikidata, 2021): circle area ~ $\log_{10}$(population)",
        fontsize=8,
    )
    fig.tight_layout()
    for ax in axes:
        label_quebec(fig, ax, xy, pop, names, label_top=QUEBEC_LABELS)
    lg.assert_legible(fig, "m05-quebec-mst.png")
    lg.assert_fits(fig, "m05-quebec-mst.png", 6.5, 0.04)
    fig.savefig(FIG / "m05-quebec-mst.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # --- permutation null: same locations, populations shuffled ---
    null = {v: {} for v in VARIANTS}
    for s in SEEDS:
        shuffled = np.random.default_rng(s).permutation(pop)
        for v in VARIANTS:
            for k, val in flat(run_variant(dist, shuffled, v)).items():
                null[v].setdefault(k, []).append(val)
    keys = [
        "mst.top5_degree_mean",
        "network.top5_degree_mean",
        "mst.mean_detour",
        "network.mean_detour",
        "mst.length",
        "network.length",
        "mst.length_share_touching_top5",
        "network.length_share_touching_top5",
    ]
    null_summary = {}
    for v in VARIANTS:
        obs = flat(out[v])
        null_summary[v] = {
            k: summarize(null[v][k])
            | {
                "observed": obs[k],
                "share_of_shuffles_at_or_below_observed": float(
                    np.mean(np.array(null[v][k]) <= obs[k])
                ),
            }
            for k in keys
        }

    results["part3_real_quebec"] = {
        "n_towns": n,
        "population_stats_top40": real_stats,
        "geometric_comparison": geo_comparison(real_stats, geo_summary),
        "selected_variant_for_figure": best,
        "selection_rule": rule,
        "variants": real,
        "permutation_null_same_locations_shuffled_populations": null_summary,
        "note_on_real_roads": (
            "The comparison with the Autoroute 20 / Autoroute 40 corridors is a comparison with "
            "the reader's knowledge: no road-network data was retrieved or used here."
        ),
    }


def geo_comparison(real_stats, geo_summary):
    g40, g30 = geo_summary["n=40"], geo_summary["n=30"]
    pick = (
        "max_over_median",
        "max_over_second",
        "top1_share",
        "top5_share",
        "coef_variation",
        "skewness",
        "excess_kurtosis",
        "loglog_rank_size_slope",
    )
    return {
        "real_top40": {k: real_stats[k] for k in pick},
        "geometric_40_towns_mean_over_200_seeds": {k: g40[k]["mean"] for k in pick},
        "geometric_40_towns_sd_over_200_seeds": {k: g40[k]["sd"] for k in pick},
        "geometric_40_towns_max_over_200_seeds": {k: g40[k]["max"] for k in pick},
        "geometric_30_towns_mean_over_200_seeds": {k: g30[k]["mean"] for k in pick},
        "caveat": (
            "The real set is the 40 largest municipalities, so its median is far above the median municipality "
            "of the province; its largest/median ratio and top-5 share therefore understate the skew of the "
            "whole province. The geometric draws are compared at n = 40 on purpose."
        ),
    }


def main():
    for d in (FIG, RES, DATA):
        d.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "script": "code/m05_roads.py",
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "random": "numpy.random.default_rng(seed); province seed 0 for the figures, seeds 0-199 for the repeated runs",
            "variants": {k: v[0] for k, v in VARIANTS.items()},
            "weights": "edge attribute 'weight' = cost divided by its median over the complete graph (no ranking changes); "
            "'distance' = real Euclidean length (unit square, or km for Quebec) used for every length and detour measure",
            "tsp": (
                "networkx.approximation.traveling_salesman_problem with its default method (Christofides) on the "
                "complete graph, as in class. It is an approximation: the 1.5 guarantee needs the triangle "
                "inequality, which holds for (a) and generally fails for (b)-(e). The function first replaces each "
                "pair by its cheapest path, so for (b)-(e) the returned cycle can pass through a hub several times; "
                "'tsp.repeated_visits' counts those repeats and the edges of such a closed walk are still added."
            ),
            "stages": "mst = minimum spanning tree on the cost; network = MST plus the TSP-cycle edges not already built",
            "detour": "mean over all unordered town pairs of (shortest path over built roads on 'distance') / (straight line)",
            "top5_bottom5": "5 largest / 5 smallest populations, ties broken by town index",
        }
    }
    pos, pop, dist = part1(results)
    part2(results, pos, pop, dist)
    geo_summary = results["part1"]["over_200_seeds"]
    part3(results, geo_summary)
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=to_jsonable) + "\n",
        encoding="utf-8",
    )
    print("done:", RESULTS_PATH)


if __name__ == "__main__":
    main()
