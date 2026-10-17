#!/usr/bin/env python3
"""Residual of a dust observer frame against its Monte-Carlo truth (`--mc`).

usage: mc_residual.py OUT_DIR/mc_NNN.f32 WIDTH HEIGHT [residual.png]

Reads the raw pair the observer writes beside `mc_NNN.png` (render then truth, f32 little endian,
both unit-sum and 5x5 box filtered) and prints, over the region where the truth is above 1e-4 of
its peak: the 2-D rms of render/truth - 1, its low-pass (9-px box) and high-pass parts (the
high-pass part is the striation / lattice amplitude), and the striation spacing from the
autocorrelation of the residual along image rows. Writes the residual image when asked
(grey = equal, +-30 % to white / black, outside the region dark).
"""
import sys
import numpy as np


def box(x, r):
    k = 2 * r + 1
    c = np.cumsum(np.pad(x, ((r + 1, r), (r + 1, r))), axis=0)
    c = np.cumsum(c, axis=1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / k ** 2


def main():
    path, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    raw = np.fromfile(path, dtype=np.float32).astype(np.float64)
    a = raw[: w * h].reshape(h, w)
    b = raw[w * h :].reshape(h, w)
    peak = b.max()
    m = b >= 1e-4 * peak
    res = np.where(m, a / np.maximum(b, 1e-300) - 1.0, 0.0)
    lo = box(res, 4)
    hp = res - lo
    rel = np.sqrt(np.sum((a[m] - b[m]) ** 2) / max(np.sum(b[m] ** 2), 1e-300))
    print(
        f"region {int(m.sum())} px: rel rms {rel:.3f}; residual rms {res[m].std():.3f} "
        f"(low-pass {lo[m].std():.3f}, high-pass {hp[m].std():.3f}); ratio p05/p50/p95 "
        f"{np.percentile(a[m] / np.maximum(b[m], 1e-300), 5):.2f}/"
        f"{np.percentile(a[m] / np.maximum(b[m], 1e-300), 50):.2f}/"
        f"{np.percentile(a[m] / np.maximum(b[m], 1e-300), 95):.2f}"
    )
    # striation spacing: the first autocorrelation peak of the high-pass residual along rows
    # through the region, pooled over the rows that hold at least 60 region pixels
    lags = []
    for y in range(h):
        xs = np.where(m[y])[0]
        if len(xs) < 60:
            continue
        r = hp[y, xs[0] : xs[-1] + 1]
        r = r - r.mean()
        if not (r.std() > 0):
            continue
        ac = np.correlate(r, r, "full")[len(r) - 1 :]
        ac /= ac[0]
        for k in range(2, min(60, len(ac) - 1)):
            if ac[k] > ac[k - 1] and ac[k] > ac[k + 1] and ac[k] > 0.05:
                lags.append(k)
                break
    if lags:
        print(
            f"striation spacing along rows: median {np.median(lags):.1f} px "
            f"(p25 {np.percentile(lags, 25):.1f}, p75 {np.percentile(lags, 75):.1f}, {len(lags)} rows)"
        )
    else:
        print("striation spacing along rows: no periodic residual found")
    if len(sys.argv) > 4:
        from PIL import Image

        im = np.full((h, w), 20, np.uint8)
        im[m] = (np.clip(res[m], -0.3, 0.3) / 0.6 * 255 + 127).astype(np.uint8)
        Image.fromarray(im).save(sys.argv[4])
        print("wrote", sys.argv[4])


if __name__ == "__main__":
    main()
