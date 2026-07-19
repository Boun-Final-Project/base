"""Visualize the synth wind (potential flow + curl-noise) on a real champ map.

Big, legible 2-panel comparison on one wall-rich map: pure potential flow vs the
GADEN-tuned curl-noise overlay. color = speed, white = streamlines, cyan arrows =
local wind sampled on a coarse grid, red ★ = source, yellow ⇒ = map-mean wind.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "train"))
from reinforcement_learning.envs.map_generator import MapGenerator
from wind_field_potflow import WindField

SPEED_RANGE = (0.05, 0.5)     # GADEN-tuned


def make_field(occ_grid, curl_amp, seed):
    wf = WindField(speed_range=SPEED_RANGE, max_speed=2.0,
                   curl_noise_amplitude=curl_amp, curl_noise_scale=4.0)
    wf.randomize(occ_grid, np.random.default_rng(seed))
    return wf.Ux.copy(), wf.Uy.copy(), wf.speed, wf.direction


def frac_off(occ, Ux, Uy):
    loc = np.stack([Ux[~occ], Uy[~occ]], 1)
    nz = np.linalg.norm(loc, axis=1) > 1e-6
    if nz.sum() < 10:
        return 0.0
    mu = loc[nz].mean(0); mn = mu / (np.linalg.norm(mu) + 1e-9)
    c = (loc[nz] @ mn) / np.linalg.norm(loc[nz], axis=1)
    return (c < 0.5).mean() * 100


def panel(ax, occ, Ux, Uy, res, src, title):
    H, W = occ.shape
    extent = [0, W * res, 0, H * res]
    spd = np.hypot(Ux, Uy)
    vmax = max(np.percentile(spd[~occ], 98), 1e-3)
    im = ax.imshow(np.ma.array(spd, mask=occ), origin="lower", extent=extent,
                   cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
    wall = np.zeros((H, W, 4)); wall[occ] = [0.12, 0.12, 0.12, 1.0]
    ax.imshow(wall, origin="lower", extent=extent, aspect="equal")
    xs = (np.arange(W) + 0.5) * res; ys = (np.arange(H) + 0.5) * res
    U = np.ma.array(Ux, mask=occ).filled(0.0); V = np.ma.array(Uy, mask=occ).filled(0.0)
    ax.streamplot(xs, ys, U, V, color="white", density=1.4, linewidth=0.7, arrowsize=0.8)
    # coarse local-wind arrows (cyan) — show local direction varying with geometry
    step = max(2, W // 22)
    XX, YY = np.meshgrid(xs[::step], ys[::step])
    uu = Ux[::step, ::step]; vv = Uy[::step, ::step]; mm = ~occ[::step, ::step]
    ax.quiver(XX[mm], YY[mm], uu[mm], vv[mm], color="cyan", scale=6, width=0.004,
              alpha=0.9)
    # map-mean wind arrow (yellow) drawn from map center
    mu = np.stack([Ux[~occ], Uy[~occ]], 1).mean(0)
    cx, cy = W * res * 0.5, H * res * 0.08
    ax.quiver([cx], [cy], [mu[0]], [mu[1]], color="yellow", scale=3, width=0.012,
              label="map-mean wind")
    if src is not None:
        ax.plot(src[0], src[1], "*", color="red", ms=22, mec="k", mew=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, W * res); ax.set_ylim(0, H * res)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def main():
    # search (template, seed) for a wall-rich map where curl makes local!=mean clearly.
    # Use the SAME seed for selection and render so speed sample is consistent.
    mg = MapGenerator(rng=np.random.default_rng(7))
    best = None
    for _ in range(120):
        tid = int(np.random.default_rng().integers(0, 1))  # placeholder; set below
    # deterministic search
    rng_pick = np.random.default_rng(2)
    for tid in (8, 5, 9, 4):                       # dense_multi_room, multi_room, hybrid, maze
        for _try in range(20):
            o = mg.generate(template_id=tid)
            occ = np.asarray(o["grid"].grid) != 0
            seed = int(rng_pick.integers(0, 10000))
            Ux, Uy, sp, dr = make_field(o["grid"], curl_amp=0.6, seed=seed)
            f = frac_off(occ, Ux, Uy)
            # prefer low-ish mean speed (so curl is visible) and high variation
            score = f - 8.0 * max(0.0, sp - 0.35)
            if best is None or score > best[0]:
                best = (score, o, occ, seed, tid)
    _, o, occ, seed, tid = best
    res = o["grid"].resolution
    src = np.array(o["source_pos"], float)

    Ux0, Uy0, sp, dr = make_field(o["grid"], curl_amp=0.0, seed=seed)
    Ux1, Uy1, _, _ = make_field(o["grid"], curl_amp=0.6, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    panel(axes[0], occ, Ux0, Uy0, res, src,
          f"potential flow only (curl=0)\nmean {sp:.2f} m/s @ {np.rad2deg(dr):.0f}°   "
          f"local-vs-mean off: {frac_off(occ,Ux0,Uy0):.0f}%  (smooth, no eddies)")
    im = panel(axes[1], occ, Ux1, Uy1, res, src,
          f"+ curl-noise eddies (curl=0.6, GADEN-tuned)\n"
          f"local-vs-mean off: {frac_off(occ,Ux1,Uy1):.0f}%  (recirculation -> local != mean)")
    fig.suptitle("Synth wind on a dense_multi_room map — 'uniform wind with varying'\n"
                 "color=speed · white=streamlines · cyan=local wind vectors · "
                 "yellow=map-mean · ★=gas source", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.7, label="wind speed (m/s)", pad=0.015)
    out = os.path.join(ROOT, "synth_wind_viz.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out, "| variation curl0=%.0f%% curl0.6=%.0f%%" % (
        frac_off(occ, Ux0, Uy0), frac_off(occ, Ux1, Uy1)))


if __name__ == "__main__":
    main()
