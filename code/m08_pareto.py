"""Multi-objective helpers for session 8.

Convention: every criterion is turned into a cost in [0, 1] where 0 is the best value present and 1 the worst,
so "smaller is better" holds for all columns. This is the class recipe (normalise, multiply the maximised
criteria by -1) written as one step: (high - x) / span for a maximised column equals 1 + (-1)(x - low)/span
shifted by a constant, which changes neither dominance nor weighted-sum rankings.
"""

import numpy as np


def normalise(df, criteria):
    """criteria: list of (column, 'min' | 'max'). Returns an (n, d) array of costs in [0, 1]."""
    cols = []
    for col, direction in criteria:
        x = df[col].to_numpy(dtype=float)
        lo, hi = x.min(), x.max()
        if hi == lo:
            raise ValueError(f"{col} is constant, it cannot discriminate")
        if direction == "min":
            cols.append((x - lo) / (hi - lo))
        elif direction == "max":
            cols.append((hi - x) / (hi - lo))
        else:
            raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")
    return np.column_stack(cols)


def dominates(challenger, challenged):
    """Smaller is better: at least as good everywhere and strictly better somewhere (class definition)."""
    if all(v <= w for v, w in zip(challenger, challenged, strict=True)):
        return any(v < w for v, w in zip(challenger, challenged, strict=True))
    return False


def nondominated_naive(m):
    """Every pair is challenged in pure Python. Returns a boolean mask."""
    n = len(m)
    rows = [list(r) for r in m]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if j != i and dominates(rows[j], rows[i]):
                keep[i] = False
                break
    return keep


def nondominated_class_loop(m):
    """The class loop as written: car i is only challenged by the cars after it (range(i + 1, n))."""
    n = len(m)
    rows = [list(r) for r in m]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(rows[j], rows[i]):
                keep[i] = False
                break
    return keep


def nondominated_fast(m):
    """Sort lexicographically, then test each point against the front built so far (vectorised).

    A dominator is never later than its victim in lexicographic order, and dominance is transitive, so
    comparing against the current front alone is enough. Exact duplicates do not dominate each other.
    """
    m = np.asarray(m, dtype=float)
    order = np.lexsort(m.T[::-1])
    front = np.empty_like(m)
    nf = 0
    keep = np.zeros(len(m), dtype=bool)
    for idx in order:
        p = m[idx]
        if nf:
            f = front[:nf]
            if np.any(np.all(f <= p, axis=1) & np.any(f < p, axis=1)):
                continue
        front[nf] = p
        nf += 1
        keep[idx] = True
    return keep


def weight_draws(rng, d, n, alpha=1.0):
    return rng.dirichlet(np.full(d, alpha), size=n)


def weighted_winners(nc, weights, chunk=2000):
    """Index (into nc) of the lowest weighted cost for each weight vector."""
    out = np.empty(len(weights), dtype=int)
    for s in range(0, len(weights), chunk):
        out[s : s + chunk] = np.argmin(weights[s : s + chunk] @ nc.T, axis=1)
    return out


def epsilon_constraint(df, primary, bounds, normalised_cost):
    """Minimise `primary` (a column) subject to bounds {column: ('>=' | '<=', value)}.

    Ties on the primary column are broken by the smallest sum of the normalised costs so the pick is not
    merely weakly Pareto. Returns the row position or None when infeasible.
    """
    ok = np.ones(len(df), dtype=bool)
    for col, (op, v) in bounds.items():
        x = df[col].to_numpy(dtype=float)
        ok &= (x >= v) if op == ">=" else (x <= v)
    if not ok.any():
        return None, 0
    idx = np.flatnonzero(ok)
    prim = df[primary].to_numpy(dtype=float)[idx]
    best = idx[prim == prim.min()]
    pick = best[np.argmin(normalised_cost[best].sum(axis=1))]
    return int(pick), int(ok.sum())


def lexicographic(nc, order, tol):
    """Filter candidates criterion by criterion (priority `order`), keeping those within `tol` of the best.

    Returns (final position into nc, list of survivor counts after each step).
    """
    alive = np.arange(len(nc))
    counts = []
    for c in order:
        col = nc[alive, c]
        alive = alive[col <= col.min() + tol]
        counts.append(len(alive))
    final = alive[np.argmin(nc[alive, order[-1]])]
    return int(final), counts


def topsis(nc):
    """Closeness to the ideal (all 0) relative to the nadir (all 1): d_nadir / (d_ideal + d_nadir). Higher is better."""
    d_ideal = np.linalg.norm(nc, axis=1)
    d_nadir = np.linalg.norm(1.0 - nc, axis=1)
    return d_nadir / (d_ideal + d_nadir), d_ideal


def reference_point(nc, goal, weights=None, rho=1e-3):
    """Achievement scalarising function: max_i w_i (f_i - g_i) + rho * sum_i w_i (f_i - g_i). Lower is better."""
    w = np.ones(nc.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    gap = w * (nc - np.asarray(goal, dtype=float))
    return gap.max(axis=1) + rho * gap.sum(axis=1)


def pairwise(nf):
    diff = nf[:, None, :] - nf[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def nearest_neighbour(nf):
    dm = pairwise(nf)
    np.fill_diagonal(dm, np.inf)
    return dm.min(axis=1)


def farthest_point_sampling(nf, k, start):
    """Greedy max-min selection of k points starting at index `start`."""
    chosen = [start]
    dist = np.linalg.norm(nf - nf[start], axis=1)
    while len(chosen) < min(k, len(nf)):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(nf - nf[nxt], axis=1))
    return chosen


def covering_radius(nf, chosen):
    """Largest distance from any front point to its nearest chosen representative."""
    d = np.linalg.norm(nf[:, None, :] - nf[chosen][None, :, :], axis=2).min(axis=1)
    return float(d.max()), float(d.mean())
