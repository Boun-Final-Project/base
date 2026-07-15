"""Honest comparison: synth wind (potential flow + curl) vs REAL GADEN CFD wind,
on the SAME map geometry. Side-by-side render + quantitative gap metrics.

Answers "is the synth wind realistic enough?" with numbers instead of guessing.
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(ROOT), "train")
sys.path.insert(0, ROOT); sys.path.insert(0, TRAIN)
from reinforcement_learning.test import gaden_loader as gl
from reinforcement_learning.envs.occupancy_grid import OccupancyGrid
from wind_field_potflow import WindField

MAP = "many_rooms"
GADEN_ROOT = Path(TRAIN).parent.parent / "gaden_scenarios"


def real_gaden_field(map_dir, grid, origin):
    """Return real GADEN wind as [H,W,2] sampled at every cell center."""
    wf = gl.load_gaden_wind_field(map_dir, grid, origin)
    H, W = grid.grid.shape; res = grid.resolution
    field = getattr(wf, "field", None)
    if field is not None and np.asarray(field).shape[:2] == (H, W):
        return np.asarray(field)[..., :2].astype(np.float32)
    xs = origin[0] + (np.arange(W) + 0.5) * res
    ys = origin[1] + (np.arange(H) + 0.5) * res
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], 1)
    uv = wf.query(pts).reshape(H, W, 2)
    return uv.astype(np.float32)


def metrics(occ, Ux, Uy):
    loc = np.stack([Ux[~occ], Uy[~occ]], 1)
    nz = np.linalg.norm(loc, axis=1) > 1e-6
    loc = loc[nz]
    if len(loc) < 10:
        return dict(off=0, rev=0, cv=0, meanspd=0, calm=100.0)
    spd = np.linalg.norm(loc, axis=1)
    mu = loc.mean(0); mn = mu / (np.linalg.norm(mu) + 1e-9)
    cos = (loc @ mn) / spd
    allspd = np.linalg.norm(np.stack([Ux[~occ], Uy[~occ]], 1), axis=1)
    return dict(
        off=float((cos < 0.5).mean() * 100),       # % cells >60deg off mean
        rev=float((cos < 0.0).mean() * 100),        # % cells REVERSED vs mean (recirculation)
        cv=float(spd.std() / (spd.mean() + 1e-9)),  # speed coefficient of variation
        meanspd=float(spd.mean()),
        calm=float((allspd < 0.02).mean() * 100),   # % near-calm free cells (stagnant pockets)
    )


def panel(ax, occ, Ux, Uy, res, title):
    H, W = occ.shape
    ext = [0, W * res, 0, H * res]
    spd = np.hypot(Ux, Uy)
    vmax = max(np.percentile(spd[~occ], 98), 1e-3)
    im = ax.imshow(np.ma.array(spd, mask=occ), origin="lower", extent=ext,
                   cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
    wall = np.zeros((H, W, 4)); wall[occ] = [0.12, 0.12, 0.12, 1]
    ax.imshow(wall, origin="lower", extent=ext, aspect="equal")
    xs = (np.arange(W) + 0.5) * res; ys = (np.arange(H) + 0.5) * res
    U = np.ma.array(Ux, mask=occ).filled(0.0); V = np.ma.array(Uy, mask=occ).filled(0.0)
    ax.streamplot(xs, ys, U, V, color="white", density=1.4, linewidth=0.7, arrowsize=0.8)
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    return im


def main():
    map_dir = gl.resolve_map_dir(GADEN_ROOT, MAP)
    grid, origin = gl.load_gaden_grid(map_dir)
    occ = np.asarray(grid.grid) != 0
    res = grid.resolution
    H, W = occ.shape
    print(f"{MAP}: grid {H}x{W} res {res} free {(~occ).sum()} origin {origin}")

    real = real_gaden_field(map_dir, grid, origin)
    rUx, rUy = real[..., 0], real[..., 1]

    # synth on the SAME geometry; match the real mean speed for fairness
    rspd = np.hypot(rUx, rUy)[~occ]
    real_mean_spd = float(rspd[rspd > 1e-6].mean()) if (rspd > 1e-6).any() else 0.3
    fields = {}
    for amp in (0.0, 0.6, 1.2):
        wf = WindField(speed_range=(real_mean_spd, real_mean_spd), max_speed=2.0,
                       curl_noise_amplitude=amp, curl_noise_scale=4.0)
        wf.randomize(grid, np.random.default_rng(7))
        fields[amp] = (wf.Ux.copy(), wf.Uy.copy())

    print(f"\n{'field':28s} {'meanSpd':>7} {'off>60°':>8} {'reversed':>9} {'speedCV':>8} {'calm%':>6}")
    mr = metrics(occ, rUx, rUy)
    print(f"{'REAL GADEN CFD':28s} {mr['meanspd']:7.3f} {mr['off']:7.0f}% {mr['rev']:8.0f}% "
          f"{mr['cv']:8.2f} {mr['calm']:5.0f}%")
    for amp, (ux, uy) in fields.items():
        m = metrics(occ, ux, uy)
        print(f"{'synth potflow curl=%.1f'%amp:28s} {m['meanspd']:7.3f} {m['off']:7.0f}% "
              f"{m['rev']:8.0f}% {m['cv']:8.2f} {m['calm']:5.0f}%")

    # render: real + the 3 synth variants
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    panel(axes[0, 0], occ, rUx, rUy, res,
          f"REAL GADEN CFD ({MAP})\noff-mean {mr['off']:.0f}% · reversed {mr['rev']:.0f}% · "
          f"CV {mr['cv']:.2f} · calm {mr['calm']:.0f}%")
    for ax, amp in zip([axes[0, 1], axes[1, 0], axes[1, 1]], (0.0, 0.6, 1.2)):
        ux, uy = fields[amp]; m = metrics(occ, ux, uy)
        panel(ax, occ, ux, uy, res,
              f"synth potential flow, curl={amp}\noff-mean {m['off']:.0f}% · reversed {m['rev']:.0f}% · "
              f"CV {m['cv']:.2f} · calm {m['calm']:.0f}%")
    fig.suptitle(f"Real GADEN CFD vs synth wind on the same {MAP} geometry\n"
                 "color=speed · white=streamlines · 'reversed'=cells blowing against the map-mean "
                 "(geometry-locked recirculation)", fontsize=12)
    fig.colorbar(panel(axes[0, 0], occ, rUx, rUy, res, axes[0, 0].get_title()),
                 ax=axes, shrink=0.5, label="wind speed (m/s)", pad=0.02)
    out = os.path.join(ROOT, "synth_vs_gaden.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
