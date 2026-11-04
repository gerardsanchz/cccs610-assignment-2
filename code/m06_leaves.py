"""Session 6 hands-on: how much of the ground do red maple leaves cover?

Follows the class notebook (saturate, red rule, median/mode/min/max, blur,
brightness, contrast, yes/no mask, composite) and then replaces the red rule
with an HSV classification. Run from anywhere:

    python -I m06_leaves.py

Inputs : ../data/leaves-original.jpg (Wikimedia Commons, CC0)
Outputs: ../data/leaves-1500.jpg, ../figures/m06-*.png, ../figures/m06-steps*.gif,
         ../results/m06.json
"""

import hashlib
import io
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # -I does not add the script folder
import legible as lg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# A 24-megapixel photo is below Pillow's decompression-bomb limit; turning the
# warning into an error means a different, oversized file fails loudly.
warnings.simplefilter("error", Image.DecompressionBombWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG, RES = ROOT / "data", ROOT / "figures", ROOT / "results"
for folder in (DATA, FIG, RES):
    folder.mkdir(exist_ok=True)

SOURCE = {
    "page": "https://commons.wikimedia.org/wiki/File:Autumn_leaves_on_the_ground_in_Quebec_City.jpg",
    "direct_image": "https://upload.wikimedia.org/wikipedia/commons/f/f2/Autumn_leaves_on_the_ground_in_Quebec_City.jpg",
    "author": "Wilfredor (Wikimedia Commons user; Wikidata Q115464159)",
    "licence_url": "http://creativecommons.org/publicdomain/zero/1.0/deed.en",
    "licence_usage_terms": "Creative Commons Zero, Public Domain Dedication",
    "licence_short": "CC0 1.0",
    "description_on_commons": "Fallen maple leaves in Quebec City in early October. Quebec City, Quebec, Canada. "
    "Photographed at Vieux-Limoilou.",
    "date_taken": "2018-10-06 12:25:28",
    "accessed": "2026-10-28",
    "licence_text_provenance": "Commons API extmetadata (LicenseUrl, UsageTerms); the rendered file page was not fetched.",
}

# Class-method parameters (M06 notebook).
SAT_FACTOR = 5
CLASS_THRESHOLD = 200
SWEEP = list(range(120, 241, 20))

# HSV rule parameters. PIL scales hue, saturation and value to 0-255; hue 255
# wraps to 0, so hue is re-centred on red: hh = h - 256 when h >= 128.
SAT_CUT = 140  # valley between the pale-leaf and vivid-leaf saturation modes
VAL_CUT = 96  # valley between the asphalt mode and the leaf mode
RED_MAX = 10  # hue <= 10 (about 14 degrees) counts as red
ORANGE_MAX = 40  # hue 11-40 (about 15-56 degrees) is orange/yellow
GREEN_MAX = 130  # hue 41-130 (about 58-183 degrees) is green
GREEN_MIN_SAT = 40


def pct(boolean_array):
    return round(100.0 * float(np.mean(boolean_array)), 3)


def class_rule_mask(rgb, threshold):
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return (r > threshold) & (r > 2 * g) & (r > 2 * b)


def class_rule_image(saturated, threshold):
    """The notebook's per-pixel loop, kept literal."""
    reds = saturated.copy()
    pixels = reds.load()
    w, h = reds.size
    for row in range(h):
        for column in range(w):
            (red, green, blue) = pixels[column, row]
            if red > threshold and red > 2 * green and red > 2 * blue:
                pixels[column, row] = (red, green, 0)
            else:
                pixels[column, row] = (0, 0, 0)
    return reds


def mask_percent(mask_l):
    """Class formula: sum of pixel values over the all-white maximum."""
    return round(
        100.0
        * float(np.asarray(mask_l, dtype=np.int64).sum())
        / (255 * mask_l.size[0] * mask_l.size[1]),
        3,
    )


def mask_from_bool(b):
    return Image.fromarray((b * 255).astype(np.uint8), mode="L")


def save_png(img, name, widths=(1500, 1200, 1000, 900, 800, 700, 600), limit=1_500_000):
    """Save a PNG under the size limit by stepping the width down."""
    for width in widths:
        out = (
            img
            if img.size[0] <= width
            else img.resize(
                (width, round(img.size[1] * width / img.size[0])), Image.LANCZOS
            )
        )
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        if buf.tell() < limit:
            (FIG / name).write_bytes(buf.getvalue())
            return {
                "file": name,
                "width": out.size[0],
                "height": out.size[1],
                "bytes": buf.tell(),
            }
    raise RuntimeError(f"{name} stays above {limit} bytes at width {widths[-1]}")


# ---------------------------------------------------------------- load
original_path = DATA / "leaves-original.jpg"
full = Image.open(original_path)
full.load()
full = full.convert("RGB")
sha = hashlib.sha256(original_path.read_bytes()).hexdigest()
small = full.resize((1500, round(full.size[1] * 1500 / full.size[0])), Image.LANCZOS)
small.save(DATA / "leaves-1500.jpg", quality=90)
small = Image.open(DATA / "leaves-1500.jpg").convert(
    "RGB"
)  # work on exactly what is shipped
W, H = small.size
results = {
    "source": SOURCE,
    "image": {
        "original_file": "data/leaves-original.jpg",
        "original_sha256": sha,
        "original_size_px": list(full.size),
        "original_bytes": original_path.stat().st_size,
        "working_file": "data/leaves-1500.jpg",
        "working_size_px": [W, H],
        "working_bytes": (DATA / "leaves-1500.jpg").stat().st_size,
        "resampling": "LANCZOS, JPEG quality 90",
    },
    "problems": [],
}
rgb = np.asarray(small)

# ---------------------------------------------------------------- 1. class method
saturated = ImageEnhance.Color(small).enhance(SAT_FACTOR)
sat_rgb = np.asarray(saturated)
class_img = class_rule_image(saturated, CLASS_THRESHOLD)
class_np = class_rule_mask(sat_rgb, CLASS_THRESHOLD)
# The vectorised rule must equal the notebook's loop; checked on a crop.
crop_box = (500, 300, 800, 500)
crop_loop = np.asarray(class_rule_image(saturated.crop(crop_box), CLASS_THRESHOLD)).any(
    axis=2
)
crop_np = class_np[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
assert (crop_loop == crop_np).all(), (
    "vectorised class rule differs from the notebook loop"
)
assert (np.asarray(class_img).any(axis=2) == class_np).all()

unsat_class_img = class_rule_image(small, CLASS_THRESHOLD)
unsat_np = class_rule_mask(rgb, CLASS_THRESHOLD)

factor_sweep = {}
for factor in (1, 1.5, 2, 3, 5):
    arr = np.asarray(ImageEnhance.Color(small).enhance(factor))
    factor_sweep[str(factor)] = {str(t): pct(class_rule_mask(arr, t)) for t in SWEEP}
results["class_method"] = {
    "saturation_factor": SAT_FACTOR,
    "rule": "red > T and red > 2*green and red > 2*blue",
    "threshold": CLASS_THRESHOLD,
    "percent_marked_saturated_x5": pct(class_np),
    "percent_marked_unsaturated": pct(unsat_np),
    "percent_with_red_above_200_only_unsaturated": pct(rgb[..., 0] > 200),
    "sweep_percent_marked_by_threshold": {
        "saturation_x5": factor_sweep["5"],
        "no_saturation": factor_sweep["1"],
    },
    "sweep_percent_marked_by_saturation_factor_and_threshold": factor_sweep,
    "vectorised_equals_notebook_loop_on_crop": True,
}

# ---------------------------------------------------------------- 2. HSV rule
hsv = np.asarray(small.convert("HSV")).astype(np.int32)
h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
hh = np.where(h >= 128, h - 256, h)

hist_s = np.histogram(s, bins=32, range=(0, 256))[0]
hist_v = np.histogram(v, bins=32, range=(0, 256))[0]
hist_hh = np.histogram(hh[(s >= SAT_CUT) & (v >= VAL_CUT)], bins=np.arange(-16, 61, 4))[
    0
]
print("saturation histogram, 32 bins of 8 (0-255):", hist_s.tolist())
print("value histogram, 32 bins of 8 (0-255):", hist_v.tolist())
print("hue histogram of vivid pixels, 4-wide bins from -16:", hist_hh.tolist())
# valleys: the lowest count between the two modes, read from the histograms
valley_s = int(np.argmin(hist_s[12:22]) + 12) * 8
valley_v = int(np.argmin(hist_v[8:16]) + 8) * 8
print(
    "saturation valley bin starts at",
    valley_s,
    "; value valley bin starts at",
    valley_v,
)

dark = v < VAL_CUT
vivid = ~dark & (s >= SAT_CUT)
red = vivid & (hh <= RED_MAX)
orange = vivid & (hh > RED_MAX) & (hh <= ORANGE_MAX)
green = ~dark & (hh > ORANGE_MAX) & (hh <= GREEN_MAX) & (s >= GREEN_MIN_SAT)
other = ~(dark | red | orange | green)
cls = np.full((H, W), 3, np.uint8)  # 3 = other (dark or pale or desaturated)
cls[red], cls[orange], cls[green] = 0, 1, 2
# red, orange and green exclude dark pixels and each other, so the classes partition the frame
assert (red.astype(int) + orange + green + dark + other == 1).all()
pale = other & ~dark
results["hsv_rule"] = {
    "cutoffs": {
        "saturation_min_vivid": SAT_CUT,
        "value_min": VAL_CUT,
        "red_hue_max_centred": RED_MAX,
        "orange_yellow_hue_max": ORANGE_MAX,
        "green_hue_max": GREEN_MAX,
        "green_min_saturation": GREEN_MIN_SAT,
        "hue_scale": "PIL 0-255 (255 = 360 degrees), centred on red (hue>=128 minus 256)",
    },
    "histograms": {
        "saturation_32_bins_of_8": hist_s.tolist(),
        "value_32_bins_of_8": hist_v.tolist(),
        "hue_of_vivid_pixels_4_wide_bins_from_minus_16": hist_hh.tolist(),
        "saturation_valley_bin_start": valley_s,
        "value_valley_bin_start": valley_v,
    },
    "percent": {
        "red": pct(cls == 0),
        "orange_yellow": pct(cls == 1),
        "green": pct(cls == 2),
        "brown_dark_other": pct(cls == 3),
        "other_split_dark_v_below_cut": pct(dark & ~(red | orange | green)),
        "other_split_pale_or_desaturated": pct(pale),
    },
    "class_rule_percent_for_comparison": {
        "saturated_x5": pct(class_np),
        "unsaturated": pct(unsat_np),
    },
}
# agreement between the class rule (no saturation) and the HSV red class
results["hsv_rule"]["agreement"] = {
    "class_rule_unsaturated_pixels_that_are_hsv_red_percent_of_class": round(
        100 * float((unsat_np & red).sum() / max(unsat_np.sum(), 1)), 2
    ),
    "hsv_red_pixels_caught_by_unsaturated_class_rule_percent_of_hsv_red": round(
        100 * float((unsat_np & red).sum() / max(red.sum(), 1)), 2
    ),
    "class_rule_unsaturated_pixels_in_each_hsv_class_percent_of_class": {
        n: round(100 * float((unsat_np & (cls == i)).sum() / max(unsat_np.sum(), 1)), 2)
        for i, n in enumerate(["red", "orange_yellow", "green", "brown_dark_other"])
    },
}

# sensitivity of the HSV red class to its own cutoffs
sens = {"saturation_cut": {}, "red_hue_max": {}, "value_cut": {}}
for sc in range(100, 181, 20):
    sens["saturation_cut"][str(sc)] = pct((v >= VAL_CUT) & (s >= sc) & (hh <= RED_MAX))
for hm in (4, 6, 8, 10, 12, 14, 16):
    sens["red_hue_max"][str(hm)] = pct((v >= VAL_CUT) & (s >= SAT_CUT) & (hh <= hm))
for vc in (60, 80, 96, 112, 130):
    sens["value_cut"][str(vc)] = pct((v >= vc) & (s >= SAT_CUT) & (hh <= RED_MAX))
results["hsv_rule"]["red_percent_sensitivity"] = sens

# unreliable classes: measured on the whole frame and on boxes read off the figure
rois = {
    "asphalt_gap_between_leaves_center_left": (500, 560, 570, 650),
    "deep_crimson_leaves_in_shadow_lower_center": (780, 730, 880, 900),
    "olive_green_leaf_top_right": (1385, 70, 1490, 170),
    "large_pale_peach_leaf_center": (640, 420, 790, 540),
}


def roi_shares(box):
    x0, y0, x1, y1 = box
    c = cls[y0:y1, x0:x1]
    d = dark[y0:y1, x0:x1]
    return {
        "box_xyxy_in_1500px": list(box),
        "percent_red": pct(c == 0),
        "percent_orange_yellow": pct(c == 1),
        "percent_green": pct(c == 2),
        "percent_other": pct(c == 3),
        "percent_dark_v_below_cut": pct(d),
        "mean_rgb": [
            round(float(x), 1) for x in rgb[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        ],
    }


results["hsv_rule"]["unreliable"] = {
    "regions": {k: roi_shares(b) for k, b in rois.items()},
    "global": {
        "dark_pixels_with_vivid_red_or_orange_hue_percent_of_frame": pct(
            (v < VAL_CUT) & (s >= SAT_CUT) & (hh <= ORANGE_MAX) & (hh > -16)
        ),
        "pale_red_hue_pixels_s_60_to_139_hue_le_10_percent_of_frame": pct(
            (v >= VAL_CUT) & (s >= 60) & (s < SAT_CUT) & (hh <= RED_MAX) & (hh >= -16)
        ),
        "near_white_glare_v_ge_235_s_lt_60_percent_of_frame": pct(
            (v >= 235) & (s < 60)
        ),
        "saturation_within_20_of_cut_percent_of_frame": pct(np.abs(s - SAT_CUT) <= 20),
    },
}

# ---------------------------------------------------------------- 3. cleaning chain
raw_mask = mask_from_bool(red)
steps = [("raw HSV red mask", raw_mask)]
chain = [
    ("median 5", lambda m: m.filter(ImageFilter.MedianFilter(size=5))),
    ("min 5 (thin)", lambda m: m.filter(ImageFilter.MinFilter(size=5))),
    ("max 5 (fatten)", lambda m: m.filter(ImageFilter.MaxFilter(size=5))),
    ("gaussian blur r=8", lambda m: m.filter(ImageFilter.GaussianBlur(radius=8))),
    ("contrast x4", lambda m: ImageEnhance.Contrast(m).enhance(4)),
]
cur = raw_mask
for name, fn in chain:
    cur = fn(cur)
    steps.append((name, cur))
final_mask = cur.point(
    lambda p: 255 if p > CLASS_THRESHOLD else 0
)  # the class's yes/no cut
steps.append(("yes/no cut value > 200", final_mask))
prev_bin = np.asarray(steps[0][1]) > 127
cleaning = []
for name, m in steps:
    arr = np.asarray(m)
    binary = arr > 127
    cleaning.append({
        "step": name,
        "percent_marked_class_formula_mean_intensity": mask_percent(m),
        "percent_pixels_above_127": pct(binary),
        "percent_of_frame_flipped_vs_previous_step": pct(binary != prev_bin),
    })
    prev_bin = binary
results["cleaning"] = {
    "order": [n for n, _ in steps],
    "steps": cleaning,
    "final_percent_marked": mask_percent(final_mask),
    "note": "Intermediate percentages use the class formula (sum of pixel values / all-white maximum); "
    "blur and contrast produce grays, so that formula and the count above 127 can differ.",
}
# The notebook also brightens x3 after the blur; on a binary mask that grows the area, so it is measured but not used.
blurred_mask = steps[4][1]
class_style = ImageEnhance.Contrast(
    ImageEnhance.Brightness(blurred_mask).enhance(3)
).enhance(4)
class_style_cut = class_style.point(lambda p: 255 if p > CLASS_THRESHOLD else 0)
results["cleaning"]["variant_with_class_brightness_x3_before_contrast"] = {
    "final_percent_marked": mask_percent(class_style_cut),
    "percent_after_brightness_x3": mask_percent(
        ImageEnhance.Brightness(blurred_mask).enhance(3)
    ),
}

# ---------------------------------------------------------------- final highlight images
highlight = Image.new("RGB", small.size, (0, 190, 255))
tinted = Image.blend(small, highlight, 0.55)
overlay = Image.composite(tinted, small, final_mask)
gray_flat = Image.new("RGB", small.size, (128, 128, 128))
gray_desat = ImageEnhance.Color(small).enhance(0)
gray_base = Image.blend(gray_desat, gray_flat, 0.6)
grayed = Image.composite(small, gray_base, final_mask)
flat_class_style = Image.composite(
    small, gray_flat, final_mask
)  # class notebook's blank-gray variant

# colour-coded HSV class map
palette = np.array(
    [(215, 25, 28), (255, 190, 0), (45, 160, 60), (110, 95, 90)], np.uint8
)
hsv_img = Image.fromarray(palette[cls], "RGB")

side = Image.new("RGB", (W * 2 + 20, H), (255, 255, 255))
side.paste(small, (0, 0))
side.paste(grayed, (W + 20, 0))

figs = {}
figs["m06-01-original.png"] = small
figs["m06-02-saturated.png"] = saturated
figs["m06-03-class-rule.png"] = class_img
figs["m06-03b-class-rule-unsaturated.png"] = unsat_class_img
figs["m06-04-hsv-rule.png"] = hsv_img
figs["m06-05-cleaned-mask.png"] = final_mask.convert("RGB")
figs["m06-06-overlay.png"] = overlay
figs["m06-07-grayed.png"] = grayed
figs["m06-08-side-by-side.png"] = side
figs["m06-09-grayed-flat-class-style.png"] = flat_class_style
png_info = {name: save_png(img, name) for name, img in figs.items()}

# ---------------------------------------------------------------- contact sheet
panels = [
    ("1. Original", small),
    ("2. Saturation x5", saturated),
    (f"3. Class rule on x5 image:\n{pct(class_np):.1f}% marked", class_img),
    (f"3b. Class rule, no saturation:\n{pct(unsat_np):.1f}% marked", unsat_class_img),
    (
        (
            "4. HSV classes (% of pixels):\n"
            f"red {pct(cls == 0):.1f}, orange/yellow {pct(cls == 1):.1f},\n"
            f"green {pct(cls == 2):.1f}, other {pct(cls == 3):.1f}"
        ),
        hsv_img,
    ),
    (
        f"5. Cleaned red mask:\n{mask_percent(final_mask):.1f}% marked",
        final_mask.convert("RGB"),
    ),
    ("6. Overlay (cyan = red leaves)", overlay),
    ("7. Non-red grayed out", grayed),
]
# Drawn at the width it prints at (6.5 in), so the font sizes below are the sizes on paper.
SHEET_PT = 7
fig, axes = plt.subplots(3, 3, figsize=(6.5, 5.4))
for ax, (title, img) in zip(axes.ravel(), panels):
    ax.imshow(img.resize((600, round(600 * H / W)), Image.LANCZOS))
    ax.set_title(title, fontsize=SHEET_PT)
    ax.set_xticks([])  # not axis("off"): its hidden tick labels would still count as text
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
ax = axes.ravel()[8]
ax.plot(np.arange(32) * 8 + 4, hist_s / (H * W) * 100, label="saturation")
ax.plot(np.arange(32) * 8 + 4, hist_v / (H * W) * 100, label="value")
ax.axvline(SAT_CUT, color="C0", ls="--")
ax.axvline(VAL_CUT, color="C1", ls="--")
ax.set_title(f"8. Histograms; cutoffs\nS={SAT_CUT}, V={VAL_CUT}", fontsize=SHEET_PT)
ax.set_xlabel("PIL value 0-255", fontsize=SHEET_PT)
ax.set_ylabel("% of pixels per bin", fontsize=SHEET_PT)
ax.tick_params(labelsize=SHEET_PT)
ax.legend(fontsize=SHEET_PT)
fig.tight_layout()
lg.assert_legible(fig, "m06-steps.png")
lg.assert_fits(fig, "m06-steps.png", 6.5, 0)
best = None
for dpi in (200, 170, 140, 110, 80):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    if buf.tell() < 1_500_000:
        best = (dpi, buf.getvalue())
        break
plt.close(fig)
if best is None:
    raise RuntimeError("contact sheet stays above 1.5 MB at 80 dpi")
(FIG / "m06-steps.png").write_bytes(best[1])
png_info["m06-steps.png"] = {
    "file": "m06-steps.png",
    "dpi": best[0],
    "bytes": len(best[1]),
}
results["figures"] = png_info

# ---------------------------------------------------------------- 4. GIFs
STAGES = [
    ("1 original", small),
    ("2 saturation x5", saturated),
    ("3 class rule (red > 200)", class_img),
    ("4 HSV classes", hsv_img),
    ("5 cleaned red mask", final_mask.convert("RGB")),
    ("6 overlay: red leaves in cyan", overlay),
]
font_cache = {}


def captioned(img, width, label):
    frame = img.resize((width, round(width * img.size[1] / img.size[0])), Image.LANCZOS)
    size = max(12, width // 28)
    if size not in font_cache:
        font_cache[size] = ImageFont.load_default(size=size)
    bar = size + 10
    canvas = Image.new("RGB", (width, frame.size[1] + bar), (20, 20, 20))
    canvas.paste(frame, (0, bar))
    ImageDraw.Draw(canvas).text(
        (8, 4), label, fill=(255, 255, 255), font=font_cache[size]
    )
    return canvas.quantize(colors=128, dither=Image.Dither.FLOYDSTEINBERG)


def write_gif(name, width, duration_ms, loop):
    frames = [captioned(img, width, label) for label, img in STAGES]
    kwargs = {
        "save_all": True,
        "append_images": frames[1:],
        "duration": duration_ms,
        "optimize": True,
    }
    if loop is not None:
        kwargs["loop"] = loop
    frames[0].save(FIG / name, format="GIF", **kwargs)


def read_back(name):
    with Image.open(FIG / name) as g:
        durations = []
        for i in range(g.n_frames):
            g.seek(i)
            durations.append(g.info.get("duration"))
        g.seek(0)
        return {
            "file": name,
            "bytes": (FIG / name).stat().st_size,
            "size_px": list(g.size),
            "n_frames": g.n_frames,
            "durations_ms": durations,
            "info_loop": g.info.get("loop", "absent"),
        }


write_gif("m06-steps.gif", 600, 1200, 0)
write_gif("m06-steps-once.gif", 400, 600, None)
# for the record: what PIL writes for loop=1, kept in memory only
probe = io.BytesIO()
frames_probe = [captioned(img, 300, label) for label, img in STAGES]
frames_probe[0].save(
    probe,
    format="GIF",
    save_all=True,
    append_images=frames_probe[1:],
    duration=600,
    loop=1,
)
probe.seek(0)
with Image.open(probe) as g:
    loop1_info = g.info.get("loop", "absent")
results["gifs"] = {
    "m06-steps.gif": {
        **read_back("m06-steps.gif"),
        "saved_with": "width 600, duration=1200, loop=0, optimize=True",
    },
    "m06-steps-once.gif": {
        **read_back("m06-steps-once.gif"),
        "saved_with": "width 400, duration=600, loop omitted, optimize=True",
    },
    "loop_equals_1_probe_info_loop_read_back": loop1_info,
    "parameters": {
        "speed": "duration= milliseconds per frame (a list gives one value per frame)",
        "size": "resize frames before saving (width); optimize=True and fewer palette colours (quantize colors=128) shrink bytes",
        "looping": "loop=0 writes the NETSCAPE extension with count 0 = repeat forever; omitting loop writes no extension and "
        "viewers play the animation once; loop=N writes count N, which browsers treat as N repeats after the first play",
    },
}
results["problems"].append(
    "Playback of loop=1 was not tested in a browser; the once-GIF omits loop, which every viewer plays once."
)
results["problems"].append(
    "Saturation x5 makes the class red rule mark most of the frame (see class_method), so frame 3 of the GIF is the rule failing, shown as the class defines it."
)

(RES / "m06.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print(
    json.dumps({k: results[k] for k in ("class_method", "cleaning")}, indent=1)[:6000]
)
print(json.dumps(results["hsv_rule"]["percent"], indent=1))
print(json.dumps(results["hsv_rule"]["unreliable"], indent=1))
print(json.dumps(results["gifs"], indent=1))
print(json.dumps(results["figures"], indent=1))
