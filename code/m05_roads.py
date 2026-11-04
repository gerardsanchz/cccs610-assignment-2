"""CCCS 610, session 5 (Getting around: building roads), assignment 2.

Basic stage: the class region (30 random towns, MST + TSP additions) with a
geometric-distribution population per town, drawn with circle area on a log scale.

Run: python -I m05_roads.py
Outputs (under the assignment-2 folder): figures/m05-basic-mst-tsp.png and
results/m05-results.json.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # -I does not add the script folder
import legible as lg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

# ----------------------------------------------------------------------------
# Paths and parameters
# ----------------------------------------------------------------------------
D = Path(__file__).resolve().parent.parent
FIG, RES = D / "figures", D / "results"
RESULTS_PATH = RES / "m05-results.json"

N_TOWNS = 30  # class value
GEO_P = 0.1  # geometric parameter: draws are in units of POP_UNIT inhabitants
POP_UNIT = 1000  # one geometric unit = 1000 inhabitants (mean 10 units = 10 000)
PART1_SEED = 4  # the single region shown in the figure; seed 0 puts towns 11 and 14 1.5% of the side of the square apart, so their circles overlap
SEEDS = list(range(200))  # the 200 regions of the repeated experiment
S_MIN, S_MAX = 25.0, 500.0  # marker area (points^2) of smallest / largest town
FIG1_S_MAX = (
    200.0  # largest marker area of the 7-inch Fig. 1, so that no two circles overlap
)

VARIANTS = {"a_distance": ("(a) distance only (class)", lambda d, pi, pj: d)}


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
def node_sizes(pop, s_max=None):
    """Marker area (points^2) = S_MIN + (S_MAX - S_MIN) * (log10(pop) - log10(pop_min)) /
    (log10(pop_max) - log10(pop_min)): the area grows linearly with log10 of the population."""
    s_max = S_MAX if s_max is None else s_max
    lp = np.log10(np.asarray(pop, dtype=float))
    lo, hi = lp.min(), lp.max()
    if hi == lo:
        return np.full(len(lp), S_MIN)
    return S_MIN + (s_max - S_MIN) * (lp - lo) / (hi - lo)


def size_legend(
    ax,
    pop,
    anchor_y=-0.03,
    fontsize=7,
    columnspacing=2.5,
    handleheight=0.7,
    loc="upper center",
    anchor_x=0.5,
    ncol=3,
    s_max=None,
):
    """Three reference circles outside the map (under it by default).

    handleheight (in font sizes) must make the row at least as tall as the biggest circle, or that circle
    spills out of the frame and the crop clips it.
    """
    pop = np.asarray(pop, dtype=float)
    refs = [pop.min(), math.sqrt(pop.min() * pop.max()), pop.max()]
    sizes = node_sizes(np.array([pop.min(), pop.max()] + refs), s_max)[2:]
    handles = [
        ax.scatter([], [], s=s, c="lightgray", edgecolors="k", label=f"{round(r):,}")
        for r, s in zip(refs, sizes)
    ]
    ax.legend(
        handles=handles,
        title="population",
        loc=loc,
        bbox_to_anchor=(anchor_x, anchor_y),
        ncol=ncol,
        fontsize=fontsize,
        title_fontsize=fontsize,
        columnspacing=columnspacing,
        handletextpad=1.2,
        handleheight=handleheight,
        borderpad=0.8,
        framealpha=0.9,
        scatterpoints=1,
    )


def assert_no_overlap(fig, ax, pos, pop, s_max=None):
    """No two town circles of the figure overlap."""
    fig.canvas.draw()
    pt = fig.dpi / 72
    centres = ax.transData.transform(pos)
    radii = np.sqrt(node_sizes(pop, s_max)) / 2 * pt
    for i, j in itertools.combinations(range(len(pop)), 2):
        if np.hypot(*(centres[i] - centres[j])) < radii[i] + radii[j]:
            raise AssertionError(f"the circles of towns {i} and {j} overlap")


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


EDGE_LW = 0.8  # points: the black outline of a town


def draw_province(
    ax,
    pos,
    pop,
    mst_edges=(),
    tsp_edges=(),
    title=None,
    title_size=8,
    s_max=None,
):
    """Roads, then the town circles, with area linear in log10 of the population."""
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
        s=node_sizes(pop, s_max),
        c="lightgray",
        edgecolors="k",
        linewidths=EDGE_LW,
        zorder=3,
    )


# ----------------------------------------------------------------------------
# Basic stage
# ----------------------------------------------------------------------------
def part1(results):
    pos, pop = make_province(PART1_SEED)
    stats = skew_stats(pop)
    print(
        f"[part 1] seed {PART1_SEED}: n={N_TOWNS}, p={GEO_P}, unit={POP_UNIT}; "
        f"mean={stats['mean']:.0f}, max={int(stats['max'])}, median={stats['median']:.0f}, "
        f"max/median={stats['max_over_median']:.2f}"
    )

    dist = dist_matrix(pos)
    res = run_variant(dist, pop, "a_distance")
    # Printed 7 in wide, the full text width: the map on the left, the size legend on the right. No town
    # numbers: nothing in the report refers to a town by number.
    fig, ax = plt.subplots(figsize=(7.0, 6.6), dpi=150)
    draw_province(
        ax,
        pos,
        pop,
        res["_mst_edges"],
        res["_tsp_added"],
        title="Class MST (black) + TSP additions (orange), distance cost",
        title_size=10,
        s_max=FIG1_S_MAX,
    )
    size_legend(
        ax,
        pop,
        anchor_y=0.5,
        anchor_x=1.03,
        loc="center left",
        ncol=1,
        fontsize=9,
        handleheight=2.4,
        s_max=FIG1_S_MAX,
    )
    assert_no_overlap(fig, ax, pos, pop, FIG1_S_MAX)
    lg.assert_legible(fig, "m05-basic-mst-tsp.png")
    lg.assert_fits(fig, "m05-basic-mst-tsp.png", 7.0, 0.1)
    fig.savefig(FIG / "m05-basic-mst-tsp.png", bbox_inches="tight")
    plt.close(fig)

    # skew over the 200 regions of 30 towns
    per_seed = {
        n: [skew_stats(make_province(s, n)[1]) for s in SEEDS] for n in (N_TOWNS,)
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
            f"area(points^2) = {S_MIN:g} + {S_MAX - S_MIN:g} * (log10(pop) - log10(pop_min)) / (log10(pop_max) - log10(pop_min)); "
            f"area is linear in log10(population) and spans {S_MIN:g} (smallest town of the figure) to {S_MAX:g} "
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


def main():
    for d in (FIG, RES):
        d.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "script": "code/m05_roads.py",
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "random": "numpy.random.default_rng(seed); province seed 4 for the figures, seeds 0-199 for the repeated runs",
            "variants": {k: v[0] for k, v in VARIANTS.items()},
            "weights": "edge attribute 'weight' = cost divided by its median over the complete graph (no ranking changes); "
            "'distance' = real Euclidean length in the unit square, used for every length and detour measure",
            "tsp": (
                "networkx.approximation.traveling_salesman_problem with its default method (Christofides) on the "
                "complete graph, as in class; it is an approximation. 'tsp.repeated_visits' counts the towns that "
                "the closed walk visits more than once, and the edges of the walk are added to the tree."
            ),
            "stages": "mst = minimum spanning tree on the cost; network = MST plus the TSP-cycle edges not already built",
            "detour": "mean over all unordered town pairs of (shortest path over built roads on 'distance') / (straight line)",
            "top5_bottom5": "5 largest / 5 smallest populations, ties broken by town index",
        }
    }
    part1(results)
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=to_jsonable) + "\n",
        encoding="utf-8",
    )
    print("done:", RESULTS_PATH)


if __name__ == "__main__":
    main()
