"""One Pareto + preference-handling pipeline, run on the cars and on the houses.

Every number it returns is computed here; the figures are drawn from the same arrays.
"""

import matplotlib

matplotlib.use("Agg")
import legible as lg
import m08_pareto as mp
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import numpy as np

MAX_LABELS = 12

# Figures that carry text for the page are drawn at the width they print at (A4, single column), so a size
# set here is the size on paper; 7 pt is the floor.
PRINT_W = 6.5
PRINT_DPI = 200
PRINT_RED = "#b30000"
PRINT_RC = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.formatter.useoffset": False,
}


def _cost_of(df, criteria, values):
    """Normalised cost of a raw value per criterion (same scale as mp.normalise)."""
    out = []
    for col, direction in criteria:
        x = df[col].to_numpy(dtype=float)
        lo, hi = x.min(), x.max()
        v = values[col]
        out.append((v - lo) / (hi - lo) if direction == "min" else (hi - v) / (hi - lo))
    return np.array(out)


def _describe(df, i, cols):
    r = df.iloc[i]
    d = {"id": int(r["id"]), "label": r["label"]}
    for c in cols:
        d[c] = float(r[c])
    return d


def annotate_points(ax, xs, ys, texts, colors, fontsize=7.5):
    """Ring and number each point, then list the numbered labels in a column beside the axes.

    The number beside the point and the number in the column carry the match, so the leader lines can
    cross without making the figure ambiguous.
    """
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    order = sorted(range(len(xs)), key=lambda k: ys[k])
    n = len(order)
    ax.scatter(
        xs, ys, s=130, facecolors="none", edgecolors=colors, linewidths=1.4, zorder=5
    )
    for rank, k in enumerate(order):
        frac = 0.04 + 0.92 * (rank + 0.5) / n
        ax.annotate(
            f"{k + 1}. {texts[k]}",
            xy=(xs[k], ys[k]),
            xycoords="data",
            xytext=(1.015, frac),
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=fontsize,
            color=colors[k],
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.4,
                "color": colors[k],
                "alpha": 0.5,
                "shrinkA": 0,
                "shrinkB": 5,
            },
        )
        ax.annotate(
            str(k + 1),
            xy=(xs[k], ys[k]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=colors[k],
            zorder=6,
        )


_seg_hits_circle = lg.seg_hits_circle
_seg_hits_rect = lg.seg_hits_rect
_rect_hits_circle = lg.rect_hits_circle
_segs_cross = lg.segs_cross
_rects_overlap = lg.rects_overlap


def annotate_legible(
    ax,
    xs,
    ys,
    texts,
    color,
    label_size=7,
    num_size=8,
    gap_in=0.12,
    pad_in=0.04,
    markers=None,
):
    """Ring each point, number it beside the point and list the numbered labels in a column right of the axes.

    The axes position must already be final: leaders are laid out in pixels. Slots are assigned by height and
    then moved until no leader crosses another, passes through another point's ring or runs over an x tick label. Each rank number takes the offset where it touches no ring,
    no leader and no other number, and the fewest of the plotted markers `markers` (data x, data y, marker area
    in points^2), nearest first. Any collision left raises instead of printing a cluttered label.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(xs)
    ring_r = (130**0.5 / 2 + 0.6) * pt
    ax.scatter(
        xs, ys, s=130, facecolors="none", edgecolors=color, linewidths=1.2, zorder=5
    )
    box = ax.get_window_extent()
    pts = ax.transData.transform(np.c_[xs, ys])
    x_end = box.x1 + gap_in * fig.dpi
    min_gap = 1.6 * label_size * pt

    def segments(slot_y):
        return [(pts[k], np.array([x_end, slot_y[k]])) for k in range(n)]

    # Pixel boxes of the drawn x tick labels and the x axis label, padded: a leader may leave the frame below
    # the axes only where it crosses none of them.
    tick_boxes = []
    for lab in ax.xaxis.get_majorticklabels() + [ax.xaxis.label]:
        if lab.get_text() and not lg._outside_view(ax.xaxis, lab, *ax.get_xlim()):
            e = mtext.Text.get_window_extent(lab, renderer)
            tick_boxes.append((
                e.x0 - 2 * pt,
                e.y0 - 2 * pt,
                e.x1 + 2 * pt,
                e.y1 + 2 * pt,
            ))

    def issues(slot_y):
        segs, found = segments(slot_y), []
        for a, (p, q) in enumerate(segs):
            if q[1] < box.y0 and any(_seg_hits_rect(p, q, tb) for tb in tick_boxes):
                found.append(f"leader {a + 1} through an x tick label")
            for k in range(n):
                # A ring that already touches the leader's own ring cannot be avoided.
                if (
                    k != a
                    and np.hypot(*(pts[k] - pts[a])) > 2 * ring_r
                    and _seg_hits_circle(p, q, pts[k], ring_r + 1.5 * pt)
                ):
                    found.append(f"leader {a + 1} through ring {k + 1}")
            for s in range(a + 1, n):
                if _segs_cross(*segs[a], *segs[s]):
                    found.append(f"leaders {a + 1} and {s + 1} cross")
        return found

    def spaced(slot_y):
        s = np.sort(slot_y)
        return bool((np.diff(s) >= min_gap).all())

    # Start from evenly spaced slots in height order, then anneal (own fixed random stream, so the analysis
    # draws are untouched): swap two labels' slots or nudge one slot, accept any move that does not add
    # collisions and a few that do, and stop at the first layout with none. Spacing is kept by every move.
    order = sorted(range(n), key=lambda k: pts[k][1])
    slot_y = np.empty(n)
    for r, k in enumerate(order):
        slot_y[k] = box.y0 + box.height * (0.04 + 0.92 * (r + 0.5) / n)
    best = len(issues(slot_y))
    rs = np.random.default_rng(0)
    lo, hi = box.y0 - 0.06 * box.height, box.y1 + 0.06 * box.height
    steps = 40000
    for it in range(steps):
        if not best:
            break
        trial = slot_y.copy()
        if rs.random() < 0.5:
            a, b = rs.integers(n, size=2)
            trial[a], trial[b] = slot_y[b], slot_y[a]
        else:
            k = int(rs.integers(n))
            trial[k] = np.clip(slot_y[k] + rs.normal(0, 25 * pt), lo, hi)
        if not spaced(trial):
            continue
        c = len(issues(trial))
        temperature = 0.6 * (1 - it / steps) + 0.02
        if c <= best or rs.random() < np.exp((best - c) / temperature):
            slot_y, best = trial, c
    if best:
        raise RuntimeError(
            f"leader collisions remain after moving slots: {issues(slot_y)}"
        )

    for k in range(n):
        yf = (slot_y[k] - box.y0) / box.height
        ax.annotate(
            "",
            xy=(xs[k], ys[k]),
            xytext=((x_end - box.x0) / box.width, yf),
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.7,
                "color": color,
                "alpha": 0.9,
                "shrinkA": 0,
                "shrinkB": ring_r / pt,
            },
        )
        ax.text(
            (x_end - box.x0) / box.width + pad_in * fig.dpi / box.width,
            yf,
            f"{k + 1}. {texts[k]}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=label_size,
            color=color,
            clip_on=False,
        )

    segs = segments(slot_y)
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    probe = ax.text(0, 0, "00", fontsize=num_size, fontweight="bold")
    placed = []
    if markers is None:
        mk_c, mk_r = np.empty((0, 2)), np.empty(0)
    else:
        mk_c = ax.transData.transform(np.c_[markers[0], markers[1]])
        mk_r = np.sqrt(np.asarray(markers[2], dtype=float)) / 2 * pt
        mk_r = np.broadcast_to(mk_r, len(mk_c))
    angles = (
        45,
        135,
        -45,
        -135,
        90,
        -90,
        0,
        180,
        22,
        158,
        -22,
        -158,
        68,
        112,
        -68,
        -112,
    )
    for k in range(n):
        label = str(k + 1)
        probe.set_text(label)
        ext = mtext.Text.get_window_extent(probe, renderer)
        hw, hh = ext.width / 2, ext.height / 2
        spot = None  # (markers touched, offset points, dx, dy, rect): fewest markers, then nearest
        for radius in (9, 11, 13, 16, 20, 24):
            for ang in angles:
                dx, dy = (
                    radius * pt * np.cos(np.radians(ang)),
                    radius * pt * np.sin(np.radians(ang)),
                )
                c = pts[k] + (dx, dy)
                rect = (c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh)
                if not (
                    box.x0 + pt < rect[0]
                    and rect[2] < box.x1 - pt
                    and box.y0 + pt < rect[1]
                    and rect[3] < box.y1 - pt
                ):
                    continue
                if any(
                    _rect_hits_circle(rect, pts[j], ring_r + 2 * pt) for j in range(n)
                ):
                    continue
                if any(_seg_hits_rect(p, q, rect) for p, q in segs):
                    continue
                if any(_rects_overlap(rect, o, 3 * pt) for o in placed):
                    continue
                hits = sum(
                    _rect_hits_circle(rect, mk_c[j], mk_r[j] + 0.5 * pt)
                    for j in range(len(mk_c))
                )
                if spot is None or hits < spot[0]:
                    spot = (hits, dx / pt, dy / pt, rect)
            if spot is not None and spot[0] == 0:
                break
        if spot is None:
            raise RuntimeError(f"no free place for the rank number {label}")
        spot = spot[1:]
        placed.append(spot[2])
        ax.annotate(
            label,
            xy=(xs[k], ys[k]),
            xytext=spot[:2],
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=num_size,
            fontweight="bold",
            color=color,
            zorder=6,
            path_effects=halo,
        )
    probe.remove()


def _fit_inches(fig, left, bottom, width, height):
    return [
        left / fig.get_figwidth(),
        bottom / fig.get_figheight(),
        width / fig.get_figwidth(),
        height / fig.get_figheight(),
    ]


def pareto_figure(plot, df, mask, fpos, chosen, title, path):
    """Pareto scatter at print size: 6.5 in wide, 7 pt labels, 8 pt numbers, cropped to its content.

    A first pass without labels measures how far the y label and ticks reach left of the axes, so the axes
    take every inch the label column leaves and the saved image is 6.5 in wide.
    """
    texts = [df.iloc[fpos[i]]["label"] for i in chosen]
    pad, ax_h, bottom = 0.04, plot["pareto_ax_h"], 1.0

    def build(left, ax_w):
        fig = plt.figure(
            figsize=(left + ax_w + 2.5, bottom + ax_h + 0.5), dpi=PRINT_DPI
        )
        ax = fig.add_axes(_fit_inches(fig, left, bottom, ax_w, ax_h))
        sc = _plot_axes(ax, df, mask, plot)
        _thousands_ticks(ax, plot)
        ax.margins(0.05)
        cax = fig.add_axes(_fit_inches(fig, left + 0.15 * ax_w, 0.34, 0.7 * ax_w, 0.11))
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label(plot["clabel"])
        ax.set_title(title, loc="left", pad=4)
        return fig, ax, sc

    with plt.rc_context(PRINT_RC):
        fig, ax, _ = build(0.6, 3.0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ink_left = 0.6 - fig.get_tightbbox(renderer).x0
        probe = fig.text(0, 0, "", fontsize=7)
        widest = 0.0
        for j, s in enumerate(texts):
            probe.set_text(f"{j + 1}. {s}")
            widest = max(
                widest, mtext.Text.get_window_extent(probe, renderer).width / fig.dpi
            )
        plt.close(fig)
        ax_w = PRINT_W - 2 * pad - ink_left - 0.12 - 0.04 - widest
        fig, ax, sc = build(ink_left + pad, ax_w)
        xs = df[plot["x"]].to_numpy(dtype=float)[fpos[chosen]]
        ys = df[plot["y"]].to_numpy(dtype=float)[fpos[chosen]]
        front = df.iloc[fpos]
        sizes = np.broadcast_to(sc.get_sizes(), len(front))
        annotate_legible(
            ax,
            xs,
            ys,
            texts,
            PRINT_RED,
            markers=(front[plot["x"]], front[plot["y"]], sizes),
        )
        floor = lg.assert_legible(fig, path.name)
        lg.assert_fits(fig, path.name, PRINT_W, pad)
        fig.savefig(path, dpi=PRINT_DPI, bbox_inches="tight", pad_inches=pad)
        plt.close(fig)
    return floor


def weights_figure(
    plot, title, f, counts, big_counts, shown, bar_labels, n_winners, nfront, path
):
    """Left: wins of the most frequent winners. Right: where the Pareto points sit. 6.5 in wide, 7 pt and up."""
    bottom, ax_h, pad = 1.1, plot["weights_ax_h"], 0.04
    with plt.rc_context(PRINT_RC):
        fig = plt.figure(figsize=(PRINT_W, bottom + ax_h + 0.5), dpi=PRINT_DPI)
        renderer = fig.canvas.get_renderer()
        probe = fig.text(0, 0, "", fontsize=7)
        widest = 0.0
        for s in bar_labels:
            probe.set_text(s)
            widest = max(
                widest, mtext.Text.get_window_extent(probe, renderer).width / fig.dpi
            )
        probe.remove()
        # The bar names sit above their bars, so the bar panel is as wide as the longest name and the
        # scatter keeps the rest of the 6.5 in.
        l_left, l_w = 0.1, widest + 0.08
        cb_left = PRINT_W - pad - 0.55
        r_left = l_left + l_w + 0.65
        a1 = fig.add_axes(_fit_inches(fig, l_left, bottom, l_w, ax_h))
        a2 = fig.add_axes(
            _fit_inches(fig, r_left, bottom, cb_left - 0.08 - r_left, ax_h)
        )
        cax = fig.add_axes(_fit_inches(fig, cb_left, bottom, 0.1, ax_h))
        rows = np.arange(len(shown))[::-1]
        a1.barh(rows, [counts[i] for i in shown], height=0.34, color="tab:blue")
        for y, s in zip(rows, bar_labels, strict=True):
            a1.text(
                0.01,
                y + 0.24,
                s,
                transform=a1.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=7,
            )
        a1.set_yticks([])
        a1.set_ylim(-0.5, len(shown) - 0.1)
        a1.spines[["left", "right", "top"]].set_visible(False)
        a1.set_xlabel("wins in 1000 random\nweight vectors")
        a1.set_title(
            f"{n_winners} distinct winners among\n{nfront} Pareto candidates",
            loc="left",
        )
        a2.scatter(
            f[plot["x"]],
            f[plot["y"]],
            s=18,
            c="0.8",
            edgecolors="none",
            label="never wins (1000 draws)",
        )
        won = counts > 0
        s2 = a2.scatter(
            f[plot["x"]][won],
            f[plot["y"]][won],
            s=30 + 12 * counts[won] ** 0.5,
            c=counts[won],
            cmap="viridis",
            edgecolors="k",
            linewidths=0.4,
            label="wins",
        )
        never_big = big_counts == 0
        if never_big.any():
            a2.scatter(
                f[plot["x"]][never_big],
                f[plot["y"]][never_big],
                marker="x",
                s=36,
                c="red",
                label=f"no winner in 400000 draws ({int(never_big.sum())})",
            )
        a2.set_xlabel(plot["xlabel"])
        a2.set_ylabel(plot["ylabel"])
        _thousands_ticks(a2, plot)
        # Row spacing: the "wins" marker is bigger than a text row and must not touch the rows around it.
        a2.legend(
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            frameon=False,
            labelspacing=1.0,
            handleheight=1.6,
        )
        cb = fig.colorbar(s2, cax=cax, label="wins")
        cb.ax.tick_params(labelsize=7.5)
        a2.set_title(f"{title}: where the\nweighted-sum winners sit", loc="left")
        lg.assert_legible(fig, path.name)
        lg.assert_fits(fig, path.name, PRINT_W, pad)
        fig.savefig(path, dpi=PRINT_DPI, bbox_inches="tight", pad_inches=pad)
        plt.close(fig)


def _thousands_ticks(ax, plot):
    """Value ticks as 200k, 400k, ... when the print spec gives a step: a tick every 100 000 prints seven
    six-digit labels that run together, and a leader needs a gap between labels to cross the tick row."""
    step = plot["x_tick_step"]
    if step:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k")
        )


def _plot_axes(ax, df, front_mask, plot, color_vals=None):
    xcol, ycol = plot["x"], plot["y"]
    ax.scatter(
        df[xcol], df[ycol], s=4, c="0.82", linewidths=0, label="all alternatives"
    )
    f = df[front_mask]
    sizes = (
        40
        if plot.get("size") is None
        else 14 + 90 * (f[plot["size"]] / f[plot["size"]].max())
    )
    sc = ax.scatter(
        f[xcol],
        f[ycol],
        s=sizes,
        c=f[plot["color"]],
        cmap="viridis",
        edgecolors="k",
        linewidths=0.4,
    )
    ax.set_xlabel(plot["xlabel"])
    ax.set_ylabel(plot["ylabel"])
    return sc


def _choose_labels(extreme_pos, pick_pos, rep_pos):
    chosen = []
    for group in (extreme_pos, pick_pos, rep_pos):
        for p in group:
            if p not in chosen and len(chosen) < MAX_LABELS:
                chosen.append(p)
    return chosen


def run(spec):
    df = spec["df"].reset_index(drop=True).copy()
    df["id"] = df[spec["id_col"]]
    criteria = spec["criteria"]
    cols = [c for c, _ in criteria]
    rng = np.random.default_rng(spec["seed"])
    res = {
        "name": spec["name"],
        "n_alternatives": len(df),
        "criteria": [{"column": c, "direction": d} for c, d in criteria],
    }

    n_all = mp.normalise(df, criteria)

    # 1. Pareto set, fast versus naive on a random subsample of 300, plus the class loop as written.
    mask = mp.nondominated_fast(n_all)
    sample = np.sort(rng.choice(len(df), size=min(300, len(df)), replace=False))
    sub = n_all[sample]
    fast_sub, naive_sub, class_sub = (
        mp.nondominated_fast(sub),
        mp.nondominated_naive(sub),
        mp.nondominated_class_loop(sub),
    )
    check = {
        "subsample_size": len(sample),
        "naive_pareto_size": int(naive_sub.sum()),
        "fast_pareto_size": int(fast_sub.sum()),
        "fast_equals_naive": bool((fast_sub == naive_sub).all()),
        "class_loop_pareto_size": int(class_sub.sum()),
        "class_loop_wrongly_kept": int((class_sub & ~naive_sub).sum()),
    }
    if not check["fast_equals_naive"]:
        raise AssertionError(
            "fast Pareto filter disagrees with the naive pairwise result"
        )
    if spec.get("full_naive"):
        full_naive = mp.nondominated_naive(n_all)
        check["full_naive_equals_fast"] = bool((full_naive == mask).all())
        check["full_class_loop_wrongly_kept"] = int(
            (mp.nondominated_class_loop(n_all) & ~full_naive).sum()
        )
        if not check["full_naive_equals_fast"]:
            raise AssertionError(
                "fast Pareto filter disagrees with the naive pairwise result on the full data"
            )
    res["pareto_check"] = check

    fpos = np.flatnonzero(mask)  # positions in df
    nf = n_all[fpos]
    nfront = len(fpos)
    res["pareto_size"] = nfront
    res["pareto_share_percent"] = round(100 * nfront / len(df), 2)
    corr = df[spec["corr_cols"]].corr().round(3)
    res["correlations"] = {
        a: {b: float(corr.loc[a, b]) for b in corr.columns} for a in corr.index
    }

    # 2. Weighted sum (class) and its sensitivity.
    w = np.array([spec["weights"][c] for c in cols])
    if abs(w.sum() - 1) > 1e-9:
        raise ValueError("weights must sum to 1")
    score = nf @ w
    order = np.argsort(score, kind="stable")
    class_pick = int(order[0])
    res["weighted_sum"] = {
        "weights": spec["weights"],
        "top5": [
            {**_describe(df, fpos[i], cols), "score": float(score[i])}
            for i in order[:5]
        ],
    }
    # Alternatives with the same cost vector tie for every weight vector, so wins are counted per vector.
    uniq, inv = np.unique(nf, axis=0, return_inverse=True)
    inv = inv.ravel()
    gsize = np.bincount(inv)
    rep = np.array([np.flatnonzero(inv == g)[0] for g in range(len(uniq))])
    draws = mp.weight_draws(rng, len(cols), 1000, 1.0)
    win = mp.weighted_winners(uniq, draws)
    ucounts = np.bincount(win, minlength=len(uniq))
    counts = ucounts[inv]
    curve = {n: len(np.unique(win[:n])) for n in (10, 30, 100, 300, 1000)}
    big = np.vstack([
        mp.weight_draws(rng, len(cols), 200_000, 1.0),
        mp.weight_draws(rng, len(cols), 200_000, 0.3),
    ])
    big_counts = np.bincount(mp.weighted_winners(uniq, big), minlength=len(uniq))[inv]
    ranked = rep[np.argsort(-ucounts, kind="stable")]
    res["weight_sensitivity"] = {
        "draws": 1000,
        "distribution": "Dirichlet(1,...,1), uniform on the simplex",
        "pareto_candidates": nfront,
        "distinct_cost_vectors_among_candidates": len(uniq),
        "distinct_winners": int((ucounts > 0).sum()),
        "pareto_never_winner_in_1000": int((counts == 0).sum()),
        "distinct_winners_vs_draws": curve,
        "top_winners": [
            {
                **_describe(df, fpos[i], cols),
                "wins": int(counts[i]),
                "alternatives_sharing_this_cost_vector": int(gsize[inv[i]]),
            }
            for i in ranked[:10]
            if counts[i] > 0
        ],
        "share_won_by_most_frequent_percent": round(100 * counts.max() / 1000, 1),
        "class_pick_wins_in_draws": int(counts[class_pick]),
        "reachability_draws": len(big),
        "reachability_draws_note": "200000 draws from Dirichlet(1) plus 200000 from Dirichlet(0.3); a sample, so the never-winner count is an upper bound on the points no weights can reach. Points with the same cost vector tie and share one count.",
        "pareto_never_winner_in_400000": int((big_counts == 0).sum()),
        "never_winner_in_400000_labels": [
            df.iloc[fpos[i]]["label"] for i in np.flatnonzero(big_counts == 0)
        ][:50],
    }

    # 3. Epsilon-constraint.
    eps = spec["eps"]
    pick, feasible = mp.epsilon_constraint(df, eps["primary"], eps["bounds"], n_all)
    if pick is None:
        raise ValueError("epsilon-constraint bounds are infeasible")
    sweep = []
    for v in eps["sweep_values"]:
        b = {**eps["bounds"], eps["sweep_col"]: (eps["sweep_op"], v)}
        p, nfeas = mp.epsilon_constraint(df, eps["primary"], b, n_all)
        sweep.append({
            "bound": v,
            "feasible": nfeas,
            **(
                {
                    "label": df.iloc[p]["label"],
                    eps["primary"]: float(df.iloc[p][eps["primary"]]),
                }
                if p is not None
                else {}
            ),
        })
    res["epsilon_constraint"] = {
        "minimise": eps["primary"],
        "bounds": {c: list(b) for c, b in eps["bounds"].items()},
        "feasible_alternatives": feasible,
        "pick": _describe(df, pick, cols),
        "pick_is_pareto": bool(mask[pick]),
        "sweep_of": f"{eps['sweep_col']} {eps['sweep_op']} bound, other bounds fixed",
        "sweep": sweep,
    }

    # 4. Lexicographic ordering over the Pareto candidates.
    lex = spec["lex"]
    lorder = [cols.index(c) for c in lex["order"]]
    lex_res = {"priority_order": lex["order"], "variants": {}}
    for tol in (0.0, lex["tol"]):
        p, cnt = mp.lexicographic(nf, lorder, tol)
        lex_res["variants"][f"tolerance_{tol}"] = {
            "survivors_after_each_step": cnt,
            "pick": _describe(df, fpos[p], cols),
        }
    lex_pick = mp.lexicographic(nf, lorder, lex["tol"])[0]
    lex_res["tolerance_note"] = (
        "a criterion keeps every candidate within the tolerance (in normalised units, 0 = best, 1 = worst) of the best one"
    )
    res["lexicographic"] = lex_res

    # 5. TOPSIS, Chebyshev distance to the ideal point, goal (reference) point.
    close, d_ideal = mp.topsis(nf)
    cheb = nf.max(axis=1)
    goal = _cost_of(df, criteria, spec["goal"])
    ref = mp.reference_point(nf, goal)
    t_pick, c_pick, r_pick = (
        int(np.argmax(close)),
        int(np.argmin(cheb)),
        int(np.argmin(ref)),
    )
    res["distance_methods"] = {
        "topsis_closeness": {
            "pick": _describe(df, fpos[t_pick], cols),
            "closeness": float(close[t_pick]),
        },
        "euclid_distance_to_ideal": {
            "pick": _describe(df, fpos[int(np.argmin(d_ideal))], cols),
            "distance": float(d_ideal.min()),
        },
        "chebyshev_to_ideal": {
            "pick": _describe(df, fpos[c_pick], cols),
            "max_cost": float(cheb[c_pick]),
        },
        "goal_point": {
            "goal": spec["goal"],
            "pick": _describe(df, fpos[r_pick], cols),
            "achievement": float(ref[r_pick]),
        },
    }

    # 6. Diversity of the Pareto set and k farthest-point representatives.
    nn = mp.nearest_neighbour(nf)
    gaps = {c: float(np.max(np.diff(np.sort(nf[:, j])))) for j, c in enumerate(cols)}
    k = spec["k"]
    start = int(np.argmin(nf[:, 0]))
    reps = mp.farthest_point_sampling(nf, k, start)
    cover_fps = mp.covering_radius(nf, reps)
    topk = [int(i) for i in order[:k]]
    cover_top = mp.covering_radius(nf, topk)
    rand_cov = [
        mp.covering_radius(nf, list(rng.choice(nfront, size=k, replace=False)))[0]
        for _ in range(200)
    ]
    res["diversity"] = {
        "front_size": nfront,
        "mean_nearest_neighbour_distance": float(nn.mean()),
        "median_nearest_neighbour_distance": float(np.median(nn)),
        "max_nearest_neighbour_distance": float(nn.max()),
        "max_possible_distance": float(np.sqrt(len(cols))),
        "points_with_identical_cost_vector": int((nn == 0).sum()),
        "max_gap_per_criterion_normalised": gaps,
        "k": k,
        "fps_start": df.iloc[fpos[start]]["label"] + f" (lowest {cols[0]})",
        "fps_representatives": [_describe(df, fpos[i], cols) for i in reps],
        "covering_radius_max_and_mean": {
            "farthest_point_sampling": [cover_fps[0], cover_fps[1]],
            f"top_{k}_by_weighted_sum": [cover_top[0], cover_top[1]],
            "random_k_median_of_200": [float(np.median(rand_cov))],
        },
    }
    res["all_pareto"] = [
        _describe(df, i, cols) for i in fpos[np.argsort(score, kind="stable")]
    ]

    picks = {
        "weighted_sum": fpos[class_pick],
        "epsilon_constraint": pick,
        "lexicographic": fpos[lex_pick],
        "topsis": fpos[t_pick],
        "chebyshev_ideal": fpos[c_pick],
        "goal_point": fpos[r_pick],
    }
    res["top_picks"] = {m: df.iloc[p]["label"] for m, p in picks.items()}
    res["distinct_picks"] = len(set(picks.values()))

    # Figures.
    plot = spec["plot"]
    ext = []
    for j, (c, d) in enumerate(criteria):
        ext.append(int(np.argmin(nf[:, j] + 1e-3 * nf.sum(axis=1))))
    chosen = _choose_labels(
        ext,
        [int(np.flatnonzero(fpos == p)[0]) for p in dict.fromkeys(picks.values())],
        reps,
    )
    res["labelled_in_pareto_figure"] = [df.iloc[fpos[i]]["label"] for i in chosen]

    # The two print figures take their text (US spelling, accents, units) from spec["print"]; the
    # diversity figure keeps the labels in spec["plot"].
    pplot = {**plot, **spec["print"]}
    pareto_figure(
        pplot,
        df,
        mask,
        fpos,
        chosen,
        f"{pplot['title']}: {nfront} Pareto-optimal of {len(df)}\n({len(chosen)} labeled; all {nfront} listed in the results JSON)",
        spec["fig_pareto"],
    )

    shown = [int(i) for i in ranked[:12] if counts[i] > 0]
    f = df.iloc[fpos]
    weights_figure(
        pplot,
        pplot["title"],
        f,
        counts,
        big_counts,
        shown,
        [
            df.iloc[fpos[i]]["label"]
            + (f" (+{gsize[inv[i]] - 1} tied)" if gsize[inv[i]] > 1 else "")
            for i in shown
        ],
        res["weight_sensitivity"]["distinct_winners"],
        nfront,
        spec["fig_weights"],
    )

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(15, 6.5), dpi=110, gridspec_kw={"width_ratios": [1.9, 1]}
    )
    a1.scatter(df[plot["x"]], df[plot["y"]], s=4, c="0.85", linewidths=0)
    a1.scatter(
        f[plot["x"]],
        f[plot["y"]],
        s=22,
        c="0.55",
        edgecolors="none",
        label=f"Pareto set ({nfront})",
    )
    a1.scatter(
        f[plot["x"]].iloc[topk],
        f[plot["y"]].iloc[topk],
        marker="^",
        s=70,
        c="tab:blue",
        edgecolors="k",
        linewidths=0.4,
        label=f"top {k} by weighted sum",
    )
    a1.scatter(
        f[plot["x"]].iloc[reps],
        f[plot["y"]].iloc[reps],
        marker="o",
        s=70,
        c="tab:red",
        edgecolors="k",
        linewidths=0.4,
        label=f"{k} farthest-point representatives",
    )
    a1.set_xlabel(plot["xlabel"])
    a1.set_ylabel(plot["ylabel"])
    a1.legend(fontsize=8, loc="best")
    a1.margins(0.04)
    fig.subplots_adjust(left=0.06, right=0.97, wspace=0.55)
    annotate_points(
        a1,
        f[plot["x"]].to_numpy()[reps],
        f[plot["y"]].to_numpy()[reps],
        [df.iloc[fpos[i]]["label"] for i in reps],
        ["tab:red"] * len(reps),
    )
    names = ["farthest-point", f"top {k} weighted", "random k (median)"]
    vals = [cover_fps[0], cover_top[0], float(np.median(rand_cov))]
    a2.bar(names, vals, color=["tab:red", "tab:blue", "0.6"])
    for i, v in enumerate(vals):
        a2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    a2.set_ylabel(
        "covering radius (max distance from a Pareto point to its representative)"
    )
    a2.tick_params(axis="x", labelsize=7)
    a2.set_title(
        f"mean NN distance {nn.mean():.3f}, largest {nn.max():.3f}", fontsize=9
    )
    fig.savefig(spec["fig_diversity"])
    plt.close(fig)
    return res
