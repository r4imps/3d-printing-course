"""probe_analyze.py - find the step where the suction cup touches, from a Probe run.

The hub's Probe button (Picker -> Test) writes probes/<run>/probe.csv and a tool camera frame per step.
This script reads one or more runs and looks for contact in three independent signals:

  camera  - the cup is fixed to the camera, so in the image it stays still while the table moves.
            A change inside the cup's region (the black rubber lip) means the cup deformed.
            lip_change = share of the lip's outline band (4 px either side of its edge) whose pixels
            flipped dark <-> light since the previous step. Step to step, not against the first
            frame: the XIAO changes its exposure and the background slides past, which adds up
            over a run but stays small between two steps.
  vacuum  - with suction on, the cup seals when it presses on a flat surface: vacuum_adc jumps.
  pose    - the robot lagging behind its target (actual - target) when it pushes on something.

Contact is the first step where a signal leaves its noise band (mean + k * std of the first
--free steps, which must still be in the air) and stays out for the next step too.
It does not move the robot; it only reads the files. Needs numpy, Pillow and matplotlib
(all already in the hub's sim venv); no OpenCV.

    cd 3d-print/lab2/slot-picker
    ..\\..\\sim\\.venv\\Scripts\\python probe_analyze.py probes\\<run>           # one run
    ..\\..\\sim\\.venv\\Scripts\\python probe_analyze.py --all                   # every run in probes/
    ..\\..\\sim\\.venv\\Scripts\\python probe_analyze.py probes\\<run> --roi-check   # only draw the cup ROI

Writes into the run folder: analysis.csv, analysis.png (signals against travel), cup_strip.jpg
(the cup region of every step side by side) and roi_check.jpg (ROI and lip mask on the first frame).
Check roi_check.jpg first: if the box does not sit on the black lip, pass --roi x0,y0,x1,y1
(full-size pixels of the 1600 x 1200 frame).

Status 13.11.26: run on the 24.10.26 Z runs, which never touched anything: no camera contact found
(correct). Synthetic deformations of their frames (lip edge moved 1-14 px) are only weakly visible
(lip_change 0.084 against 0.02-0.06 in the air), so the camera limit is not trusted yet; a real
2 mm press should move the edge ~30 px. Tune --free / -k / the 0.03 floor on the first real
contact runs. vacuum_adc and pose_lag_mm have not been tried on real data either.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PROBES = HERE / "probes"

# Cup lip in the 1600 x 1200 tool camera frame (XIAO on xiao-square-mount-v2), from the 24.10.26 runs.
DEFAULT_ROI = (1090, 660, 1440, 1110)
SCALE = 2  # work at half size: plenty for a lip that is ~300 px wide, and 4x faster


def load_rows(run):
    with (run / "probe.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def gray_roi(path, roi):
    img = Image.open(path).convert("L").crop(roi)
    img = img.resize((img.width // SCALE, img.height // SCALE), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)
    # the XIAO adjusts exposure itself: scale so the bright parts (cup body, table) are 1.0
    return a / max(float(np.percentile(a, 90)), 1.0)


def dilate(mask, r):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def largest_blob(mask):
    """Largest 4-connected True region (the lip); drops rig beams and shadows that also are dark."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    best = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, blob = [(y0, x0)], []
        while stack:
            y, x = stack.pop()
            blob.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(blob) > len(best):
            best = blob
    out = np.zeros_like(mask)
    if best:
        ys, xs = zip(*best)
        out[list(ys), list(xs)] = True
    return out


def lip_mask(ref):
    """The black rubber lip in the reference ROI: Otsu threshold, then only the largest dark blob.
    Returns (lip, threshold); the lip's surroundings are dilate(lip) & ~lip."""
    hist, edges = np.histogram(ref, bins=128)
    p = hist / hist.sum()
    mids = (edges[:-1] + edges[1:]) / 2
    w, mu = np.cumsum(p), np.cumsum(p * mids)
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-9)
    t = mids[int(np.argmax(between))]
    return largest_blob(ref < t), t


def first_out(values, free, k, floor):
    """First index where the value leaves mean + k*std of values[:free] (at least floor above the
    mean) and the next value is out too (or it is the last). None if it never does."""
    base = [v for v in values[:free] if v is not None]
    if len(base) < 2:
        return None, None
    m, s = float(np.mean(base)), float(np.std(base))
    lim = m + max(k * s, floor)
    out = [v is not None and v > lim for v in values]
    for i in range(free, len(values)):
        if out[i] and (i + 1 >= len(values) or out[i + 1]):
            return i, lim
    return None, lim


def analyse(run, roi, free, k, roi_check_only=False):
    rows = load_rows(run)
    frames = [(i, run / r["frame"]) for i, r in enumerate(rows) if r["frame"].endswith(".jpg")
              and (run / r["frame"]).exists()]
    if not frames:
        print(f"{run.name}: no frames, camera signal skipped")
    ref_i = frames[0][0] if frames else None
    ref = gray_roi(frames[0][1], roi) if frames else None
    mask, thr = lip_mask(ref) if frames else (None, None)

    if frames:
        # outline band: 4 px either side of the lip's edge, where a deformation shows
        band = dilate(mask, 4) & dilate(~mask, 4)
        img = Image.open(frames[0][1]).convert("RGB")
        over = np.zeros((img.height, img.width), bool)
        big = np.kron(mask, np.ones((SCALE, SCALE), bool))
        over[roi[1]:roi[1] + big.shape[0], roi[0]:roi[0] + big.shape[1]] = big
        a = np.asarray(img).copy()
        a[over] = (0.5 * a[over] + [0, 127, 0]).astype(np.uint8)
        img = Image.fromarray(a)
        ImageDraw.Draw(img).rectangle(roi, outline=(255, 0, 0), width=4)
        img.save(run / "roi_check.jpg", quality=85)
    if roi_check_only:
        print(f"{run.name}: roi_check.jpg written (red = ROI, green = lip mask)")
        return None

    t0 = [num(rows[0][c]) for c in ("target_x", "target_y", "target_z")]
    out, crops, prev = [], [], ref
    for i, r in enumerate(rows):
        tgt = [num(r[c]) for c in ("target_x", "target_y", "target_z")]
        act = [num(r[c]) for c in ("x", "y", "z")]
        travel = math.dist(tgt, t0) if None not in tgt + t0 else None
        lag = math.dist(tgt, act) if None not in tgt + act else None
        change = None
        p = run / r["frame"]
        if mask is not None and r["frame"].endswith(".jpg") and p.exists():
            cur = gray_roi(p, roi)
            # pixels around the lip that flipped dark <-> light since the previous step: the lip edge
            # moved. Against the previous step, not the first, so background slowly sliding past
            # the lip edge in the air does not add up (24.10.26 air runs: it grew 0.013 per 0.5 mm).
            change = float(((cur < thr) != (prev < thr))[band].mean())
            prev = cur
            crops.append((r["step"], Image.open(p).convert("L").crop(roi)))
        out.append({"step": r["step"], "travel_mm": travel, "z": act[2], "lip_change": change,
                    "vacuum_adc": num(r.get("vacuum_adc")), "adc": num(r.get("adc")),
                    "pose_lag_mm": lag, "pump": r.get("pump", "")})

    found = {}
    for key, floor in (("lip_change", 0.03), ("vacuum_adc", 15.0), ("pose_lag_mm", 0.05)):
        i, lim = first_out([o[key] for o in out], free, k, floor)
        found[key] = (i, lim)

    with (run / "analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) + ["contact"], lineterminator="\n")
        w.writeheader()
        for i, o in enumerate(out):
            hits = [k_ for k_, (j, _) in found.items() if j == i]
            w.writerow({**{k_: ("" if v is None else round(v, 4) if isinstance(v, float) else v)
                           for k_, v in o.items()}, "contact": " ".join(hits)})

    plot(run, out, found)
    if crops:
        strip(run, crops, {found["lip_change"][0]} - {None}, out)

    print(f"{run.name}  ({len(rows)} steps, ROI {roi}, reference frame step {rows[ref_i]['step'] if frames else '-'})")
    for key, (i, lim) in found.items():
        if lim is None:
            print(f"  {key:13s} no data")
        elif i is None:
            print(f"  {key:13s} no contact (limit {lim:.3f})")
        else:
            o = out[i]
            print(f"  {key:13s} CONTACT at step {o['step']}: travel {o['travel_mm']:.2f} mm, "
                  f"z {o['z']:.3f}, value {o[key]:.3f} > {lim:.3f}")
    return found


def plot(run, out, found):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [o["travel_mm"] for o in out]
    keys = [("lip_change", "lip edge change (0..1)"),
            ("vacuum_adc", "vacuum (ADC codes)"), ("pose_lag_mm", "robot lag (mm)")]
    fig, axs = plt.subplots(len(keys), 1, figsize=(7, 9), sharex=True)
    for ax, (k_, label) in zip(axs, keys):
        pts = [(a, o[k_]) for a, o in zip(x, out) if a is not None and o[k_] is not None]
        if pts:
            ax.plot(*zip(*pts), "o-", ms=3)
        i, lim = found[k_]
        if lim is not None:
            ax.axhline(lim, color="tab:red", lw=0.8, ls="--")
        if i is not None and x[i] is not None:
            ax.axvline(x[i], color="tab:red", lw=1.2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("travel from start (mm)")
    fig.suptitle(run.name)
    fig.tight_layout()
    fig.savefig(run / "analysis.png", dpi=110)
    plt.close(fig)


def strip(run, crops, hits, out):
    w, h = crops[0][1].width // 3, crops[0][1].height // 3
    cols = min(len(crops), 8)
    rows_n = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * w, rows_n * (h + 16)), "white")
    d = ImageDraw.Draw(sheet)
    by_step = {o["step"]: o for o in out}
    hit_steps = {out[i]["step"] for i in hits}
    for n, (step, im) in enumerate(crops):
        cx, cy = (n % cols) * w, (n // cols) * (h + 16)
        sheet.paste(im.resize((w, h)).convert("RGB"), (cx, cy + 16))
        tr = by_step[step]["travel_mm"]
        d.text((cx + 3, cy + 2), f"{step}  {tr:.1f} mm" if tr is not None else str(step),
               fill="red" if step in hit_steps else "black")
        if step in hit_steps:
            d.rectangle((cx, cy + 16, cx + w - 1, cy + 16 + h - 1), outline="red", width=3)
    sheet.save(run / "cup_strip.jpg", quality=85)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", type=Path, help="probe run folders (probes/<run>)")
    ap.add_argument("--all", action="store_true", help="every run under probes/")
    ap.add_argument("--roi", help="cup region x0,y0,x1,y1 in full-size pixels (default %s)" % (DEFAULT_ROI,))
    ap.add_argument("--free", type=int, default=3, help="first N steps are surely in the air (default 3)")
    ap.add_argument("-k", type=float, default=5.0, help="noise band in standard deviations (default 5)")
    ap.add_argument("--roi-check", action="store_true", help="only write roi_check.jpg")
    a = ap.parse_args(argv)
    runs = sorted(p for p in PROBES.iterdir() if (p / "probe.csv").exists()) if a.all else a.runs
    if not runs:
        ap.error("give a run folder or --all")
    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else DEFAULT_ROI
    if len(roi) != 4 or roi[2] <= roi[0] or roi[3] <= roi[1]:
        ap.error("--roi x0,y0,x1,y1 with x1 > x0 and y1 > y0")
    for run in runs:
        if not (run / "probe.csv").exists():
            print(f"{run}: no probe.csv", file=sys.stderr)
            continue
        analyse(run, roi, a.free, a.k, a.roi_check)


if __name__ == "__main__":
    main()
