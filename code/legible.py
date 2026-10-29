"""Print-size checks for figures: no drawn text may overlap another drawn text, and no text may fall under 7 pt.

A figure is drawn at the width it is printed at (6.5 in on an A4 page), so a font size set in code is the
size on paper. `assert_legible` draws the figure once and raises with the offending strings when two
pieces of text overlap or a size is below the floor; nothing is silently accepted.
"""

import matplotlib.text as mtext
import numpy as np

MIN_PT = 7.0


# Pixel geometry shared by the label placers. A box is (x0, y0, x1, y1), a point or segment end is (x, y).
def seg_hits_circle(p, q, c, r):
    d = q - p
    t = np.clip(np.dot(c - p, d) / np.dot(d, d), 0.0, 1.0)
    return bool(np.hypot(*(p + t * d - c)) < r)


def seg_hits_rect(p, q, box):
    x0, y0, x1, y1 = box
    t = np.linspace(0.0, 1.0, 400)[:, None]
    s = p + t * (q - p)
    return bool(
        ((s[:, 0] >= x0) & (s[:, 0] <= x1) & (s[:, 1] >= y0) & (s[:, 1] <= y1)).any()
    )


def rect_hits_circle(box, c, r):
    x0, y0, x1, y1 = box
    return bool(
        np.hypot(np.clip(c[0], x0, x1) - c[0], np.clip(c[1], y0, y1) - c[1]) < r
    )


def segs_cross(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def rects_overlap(a, b, pad=0.0):
    return not (
        a[2] + pad <= b[0]
        or b[2] + pad <= a[0]
        or a[3] + pad <= b[1]
        or b[3] + pad <= a[1]
    )


def _outside_view(axis, text_obj, lo, hi):
    """True for a tick label whose tick lies outside the view limits (matplotlib does not draw it)."""
    labels = axis.get_majorticklabels()
    locs = axis.get_majorticklocs()
    span = abs(hi - lo)
    for lab, loc in zip(labels, locs):
        if lab is text_obj:
            return not (min(lo, hi) - 1e-9 * span <= loc <= max(lo, hi) + 1e-9 * span)
    return False


def drawn_texts(fig):
    """The visible, non-empty Text artists a draw of fig would paint, with their pixel boxes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    skipped = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for lab in axis.get_majorticklabels() + axis.get_minorticklabels():
                if _outside_view(axis, lab, *lim):
                    skipped.add(id(lab))
    out = []
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible() or id(t) in skipped:
            continue
        if not t.get_figure():
            continue
        # Text.get_window_extent, not the Annotation override, which also covers the arrow.
        out.append((t, mtext.Text.get_window_extent(t, renderer)))
    return out


def assert_fits(fig, name, width_in, pad_in):
    """Raise when the cropped image would be wider than the print width, which would shrink its text."""
    fig.canvas.draw()
    got = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad_in
    if got > width_in + 0.005:
        raise AssertionError(
            f"{name}: cropped width {got:.3f} in exceeds {width_in} in"
        )
    return got


def assert_legible(fig, name, min_pt=MIN_PT):
    problems = []
    texts = drawn_texts(fig)
    for t, _ in texts:
        if t.get_fontsize() < min_pt:
            problems.append(f"size {t.get_fontsize():.1f} pt: {t.get_text()!r}")
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1 :]:
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w > 0.5 and h > 0.5:
                problems.append(
                    f"overlap {w:.1f}x{h:.1f} px: {a.get_text()!r} / {b.get_text()!r}"
                )
    if problems:
        raise AssertionError(f"{name}: " + "; ".join(problems))
    return min(t.get_fontsize() for t, _ in texts)
