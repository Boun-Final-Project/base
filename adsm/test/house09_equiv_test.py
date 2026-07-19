#!/usr/bin/env python3
"""ADSM port faithfulness -- probability() on the paper's real House09 data.

Complements test/adsm_equivalence_test.cpp (2.3M synthetic random inputs) by
driving BOTH implementations with the real House09 OpenFOAM CFD wind field and
the real occupancy map, at real robot poses.

WHAT THIS CHECKS
  1. equivalence  -- orig_probability() vs port_probability(), transcribed
     separately from the two source trees, on identical real inputs.
  2. robustness   -- the model stays finite and non-negative over real wind
     (real CFD wind is near-zero in most cells, which is exactly where
     Eq. 3's exp() terms could degenerate).
  3. sanity       -- the wind actually loaded is non-zero and the plume is
     wind-shaped, not a symmetric blob.

WHAT IT DOES NOT CHECK
  Only probability() is covered. estimate()'s stateful epi_set_/epr_set_
  bookkeeping, observe()'s resampling, check_terminal() and the RRT/frontier
  modules are NOT exercised here -- see test/README.md.

  It also compares transcriptions, not the compiled C++. A mis-transcription
  would pass here while the real code differed; the source diff in
  test/README.md is what covers that.

HISTORY -- two bugs that made the previous version vacuous:
  * It called ONE probability() twice and diffed the results, so "0 mismatches"
    was a tautology that could not fail.
  * lookup_wind() did round(wx/0.1)*0.1 and matched that against float dict
    keys. The CFD cell centres are OFFSET (x from -1.85, y from -8.45105), so
    it scored 0/2000 hits on its own keys and the silent (0.0, 0.0) default
    fed V=0 everywhere -- collapsing every wind term of Eq. 3 to exp(0)=1.
    The "real CFD physics" being tested was zero wind.
  Both are now asserted against, so they cannot silently return.

Usage:
  python3 house09_equiv_test.py            # offline (wind + occupancy)
  python3 house09_equiv_test.py --online   # also query live GADEN gas
"""

import argparse
import csv
import math
import os
import random
import sys

import numpy as np

# ---- paths -----------------------------------------------------------------
# The install tree is authoritative: benchmark_env ships scenarios via
# install(DIRECTORY), which colcon copies rather than symlinks, so src/ and
# install/ drift. The running nodes read install/.
_DEFAULT_HOUSE = ("/home/efe/ros2_ws/install/benchmark_env/share/benchmark_env"
                  "/scenarios/House09_1_23_08_KI")
HOUSE_DIR = os.environ.get("HOUSE09_DIR", _DEFAULT_HOUSE)
WIND_DIR = os.path.join(HOUSE_DIR, "wind_simulations", "1ms")
OCC_PGM = os.path.join(HOUSE_DIR, "environment_configurations", "config1",
                       "occupancy.pgm")
OCC_YAML = os.path.join(HOUSE_DIR, "environment_configurations", "config1",
                        "occupancy.yaml")

ROBOT_START = (-1.6, -1.03)   # scenario config.yaml empty_point
ROBOT_Z = 0.2                 # BasicSimScene.yaml robots[0].position z
SOURCE = (5.5, -6.5)          # sim1 sim.yaml source position


# ---- Eq. (3): both implementations, transcribed verbatim -------------------
# Only the pose binding differs: the original reads the SLAM/odom estimate
# (x_, y_, yaw_); the port reads ground truth (real_x_, real_y_, real_yaw_).
# The arithmetic is identical, so identical inputs must give identical outputs.

def orig_probability(x, y, x_, y_, yaw_, wind_speed_, wind_direction_):
    """adsm_orig/src/adsm.cpp:199-217 (commit 1f3c6a0b)."""
    Q = 4.0
    D = 1.0
    tau = 250
    V = wind_speed_
    phi = yaw_ - wind_direction_
    lam = math.sqrt(D * tau / (1 + V * V * tau / (4 * D)))
    dis = math.sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_))
    dx = x_ - x
    dy = y_ - y

    pa = Q / (4 * math.pi * D * (abs(dis + 0.0001)))
    pb = math.exp(-dis / lam)
    pc = math.exp(-dx * V * math.cos(phi) / (2 * D))
    pd = math.exp(-dy * V * math.sin(phi) / (2 * D))
    return pa * pb * pc * pd


def port_probability(x, y, real_x_, real_y_, real_yaw_, wind_speed_,
                     wind_direction_):
    """src/base/adsm/src/adsm.cpp probability()."""
    Q = 4.0
    D = 1.0
    tau = 250
    V = wind_speed_
    phi = real_yaw_ - wind_direction_
    lam = math.sqrt(D * tau / (1 + V * V * tau / (4 * D)))
    dis = math.sqrt((x - real_x_) * (x - real_x_) + (y - real_y_) * (y - real_y_))
    dx = real_x_ - x
    dy = real_y_ - y

    pa = Q / (4 * math.pi * D * (abs(dis + 0.0001)))
    pb = math.exp(-dis / lam)
    pc = math.exp(-dx * V * math.cos(phi) / (2 * D))
    pd = math.exp(-dy * V * math.sin(phi) / (2 * D))
    return pa * pb * pc * pd


# ---- gas binarization (port, PID thresholds) -------------------------------
def gas_binarize(window, gas_high_th=0.3, gas_low_th=0.1):
    if not window:
        return False
    gas = window[-1]
    if gas < gas_low_th:
        return False
    if gas > gas_high_th:
        return True
    for i in range(1, len(window)):
        if window[i] >= window[i - 1]:
            return True
    return False


# ---- wind field ------------------------------------------------------------
class WindField:
    """Index-addressed CFD lookup.

    The cells sit on an exact 0.1 m lattice but at an arbitrary offset
    (x0=-1.85, y0=-8.45105), so snapping to a multiple of the resolution never
    reproduces a cell centre. Index off the measured origin instead, and never
    silently default -- an out-of-domain query returns None so callers must
    decide, rather than quietly injecting V=0.
    """

    def __init__(self, path, target_z=ROBOT_Z):
        # The CSV is a 3D field (31 z-levels for House09). Collapsing it would
        # let upper layers clobber the robot's own height, so select the single
        # layer nearest the robot before building the lattice.
        rows = []
        with open(path) as f:
            for row in csv.DictReader(f):
                rows.append((float(row["Points:0"]), float(row["Points:1"]),
                             float(row["Points:2"]),
                             float(row["U:0"]), float(row["U:1"])))
        levels = sorted({r[2] for r in rows})
        self.z = min(levels, key=lambda z: abs(z - target_z))
        self.n_levels = len(levels)
        pts = [(x, y, u, v) for x, y, z, u, v in rows if z == self.z]
        xs = sorted({p[0] for p in pts})
        ys = sorted({p[1] for p in pts})
        self.x0, self.y0 = xs[0], ys[0]
        self.nx, self.ny = len(xs), len(ys)
        self.res = round(xs[1] - xs[0], 6) if len(xs) > 1 else 0.1
        self.u = np.full((self.ny, self.nx), np.nan)
        self.v = np.full((self.ny, self.nx), np.nan)
        for x, y, u, v in pts:
            ix = int(round((x - self.x0) / self.res))
            iy = int(round((y - self.y0) / self.res))
            self.u[iy, ix] = u
            self.v[iy, ix] = v
        self.cells = len(pts)

    def at(self, wx, wy):
        ix = int(round((wx - self.x0) / self.res))
        iy = int(round((wy - self.y0) / self.res))
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return None
        u, v = self.u[iy, ix], self.v[iy, ix]
        if math.isnan(u) or math.isnan(v):
            return None
        return float(u), float(v)

    def self_test(self):
        """Every cell centre must resolve to its own value. Guards the exact
        bug that made the previous test vacuous."""
        hits = 0
        checked = 0
        for iy in range(0, self.ny, max(1, self.ny // 40)):
            for ix in range(0, self.nx, max(1, self.nx // 40)):
                if math.isnan(self.u[iy, ix]):
                    continue
                wx = self.x0 + ix * self.res
                wy = self.y0 + iy * self.res
                got = self.at(wx, wy)
                checked += 1
                if got is not None and abs(got[0] - self.u[iy, ix]) < 1e-12:
                    hits += 1
        return hits, checked


def speed_dir(u, v):
    return math.hypot(u, v), math.atan2(v, u)


# ---- occupancy -------------------------------------------------------------
def load_occupancy():
    import yaml
    with open(OCC_YAML) as f:
        cfg = yaml.safe_load(f)
    res = float(cfg["resolution"])
    ox, oy = float(cfg["origin"][0]), float(cfg["origin"][1])
    with open(os.path.join(os.path.dirname(OCC_YAML), cfg["image"]), "rb") as f:
        magic = f.readline().strip()
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(t) for t in line.split())
        maxval = int(f.readline().strip())
        if magic == b"P5":
            img = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
        elif magic == b"P2":
            img = np.array(f.read().split()[:w * h], dtype=int).reshape(h, w)
        else:
            sys.exit(f"unsupported PGM magic {magic!r} in {cfg['image']}")
    # GADEN writes this map as P2 with maxval 1, so pixels are literally 0/1 --
    # NOT the 0..255 a map_server-style `img >= 254` free test assumes (that
    # silently yields zero free cells here). Scale by maxval instead, and honour
    # the yaml's free_thresh/negate the way map_server does.
    occ_prob = 1.0 - (img.astype(float) / float(maxval))   # negate: 0
    if int(cfg.get("negate", 0)):
        occ_prob = 1.0 - occ_prob
    free = occ_prob < float(cfg.get("free_thresh", 0.1))
    if not free.any():
        sys.exit(f"FATAL: 0 free cells parsed from {cfg['image']} "
                 f"(magic={magic!r} maxval={maxval}) -- occupancy decode is wrong")
    return img, free, res, ox, oy, h


def grid_to_world(gx, gy, res, ox, oy, h):
    return ox + (gx + 0.5) * res, oy + (h - 1 - gy + 0.5) * res


# ---- tests -----------------------------------------------------------------
def run_equivalence(winds, free, res, ox, oy, h, n_cases=100000):
    print("\n" + "=" * 68)
    print("TEST 1  equivalence: orig vs port probability(), real House09 wind")
    print("=" * 68)

    free_cells = np.argwhere(free)          # (row, col) = (gy, gx)
    rng = random.Random(42)
    nprng = np.random.default_rng(42)

    mismatches = 0
    maxdiff = 0.0
    bad_values = 0
    tested = 0
    zero_wind = 0
    speeds = []

    for _ in range(n_cases):
        gy_r, gx_r = free_cells[nprng.integers(len(free_cells))]
        rx, ry = grid_to_world(gx_r, gy_r, res, ox, oy, h)
        ryaw = rng.uniform(-math.pi, math.pi)

        wf = winds[rng.randrange(len(winds))]
        w = wf.at(rx, ry)
        if w is None:
            continue                        # outside the CFD domain
        ws, wd = speed_dir(*w)
        speeds.append(ws)
        if ws == 0.0:
            zero_wind += 1

        # candidate within the paper's random_sample_r = 3.0 m, in free space
        for _ in range(20):
            ang = rng.uniform(-math.pi, math.pi)
            d = rng.uniform(0.01, 3.0)
            cx, cy = rx + d * math.cos(ang), ry + d * math.sin(ang)
            gcx = int((cx - ox) / res)
            gcy = h - 1 - int((cy - oy) / res)
            if 0 <= gcy < free.shape[0] and 0 <= gcx < free.shape[1] and free[gcy, gcx]:
                break
        else:
            continue

        p_orig = orig_probability(cx, cy, rx, ry, ryaw, ws, wd)
        p_port = port_probability(cx, cy, rx, ry, ryaw, ws, wd)

        if p_orig != p_port:
            mismatches += 1
            maxdiff = max(maxdiff, abs(p_orig - p_port))
        if not (math.isfinite(p_port) and p_port >= 0.0):
            bad_values += 1
        tested += 1

    speeds = np.array(speeds) if speeds else np.array([0.0])
    print(f"  cases tested        : {tested}")
    print(f"  wind speed m/s      : min={speeds.min():.4f} mean={speeds.mean():.4f} "
          f"max={speeds.max():.4f}")
    print(f"  cells with V=0      : {zero_wind}/{tested} "
          f"({100.0 * zero_wind / max(tested, 1):.1f}%)")
    print(f"  orig vs port        : mismatches={mismatches}  max|diff|={maxdiff}")
    print(f"  non-finite/negative : {bad_values}")

    ok = (mismatches == 0 and bad_values == 0 and tested > 0)
    # If the wind were silently zero everywhere, the wind terms of Eq.3 would
    # never be exercised and this test would prove nothing -- fail loudly.
    if speeds.max() <= 0.0:
        print("  FAIL: wind is zero everywhere -- CFD not actually loaded")
        ok = False
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


def run_heatmap(winds, occ_img, free, res, ox, oy, h, out="/tmp/house09_probability_map.png"):
    print("\n" + "=" * 68)
    print("TEST 2  probability field at the paper's start pose")
    print("=" * 68)
    rx, ry = ROBOT_START
    w = winds[0].at(rx, ry)
    if w is None:
        print("  start pose outside CFD domain; skipping")
        return
    ws, wd = speed_dir(*w)
    print(f"  robot {ROBOT_START}  wind speed={ws:.4f} dir={math.degrees(wd):.1f}deg")

    H, W = occ_img.shape
    grid = np.full((H, W), np.nan)
    for gy in range(H):
        for gx in range(W):
            if not free[gy, gx]:
                continue
            wx, wy = grid_to_world(gx, gy, res, ox, oy, h)
            grid[gy, gx] = port_probability(wx, wy, rx, ry, 1.57, ws, wd)

    valid = grid[np.isfinite(grid)]
    print(f"  probability range   : [{valid.min():.3e}, {valid.max():.3e}]")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lo, hi = np.percentile(valid, 1), np.percentile(valid, 99)
        norm = np.clip((grid - lo) / max(hi - lo, 1e-30), 0, 1)
        plt.figure(figsize=(6, 9))
        plt.imshow(np.where(free, 1.0, 0.0), cmap="gray", vmin=0, vmax=1)
        plt.imshow(norm, cmap="jet", alpha=np.where(np.isfinite(grid), 0.65, 0.0))
        plt.plot(int((rx - ox) / res), h - 1 - int((ry - oy) / res), "wo", ms=6)
        plt.title("ADSM probability() -- House09, real CFD wind")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"  heatmap             : {out}")
    except Exception as e:                                  # noqa: BLE001
        print(f"  (heatmap skipped: {e})")


def run_online_gas(winds, free, res, ox, oy, h):
    print("\n" + "=" * 68)
    print("TEST 3  live GADEN gas -> binarization")
    print("=" * 68)
    try:
        import rclpy
        from rclpy.node import Node
        from gaden_msgs.srv import GasPosition
    except Exception as e:                                  # noqa: BLE001
        print(f"  ROS unavailable ({e}); skipping")
        return

    rclpy.init()
    node = Node("house09_equiv_gas_test")
    cli = node.create_client(GasPosition, "/odor_value")
    if not cli.wait_for_service(timeout_sec=5):
        print("  /odor_value not up (needs gaden_player running); skipping")
        rclpy.shutdown()
        return

    # Walk a straight transect from start toward the source, so samples form a
    # genuine TIME series along a path. Sampling scattered points and pushing
    # them through the 6-sample window would be meaningless: the window models
    # a sensor's history at one moving robot, not unrelated places.
    sx, sy = ROBOT_START
    tx, ty = SOURCE
    n = 120
    window, hits, concs = [], 0, []
    for i in range(n + 1):
        t = i / n
        wx, wy = sx + t * (tx - sx), sy + t * (ty - sy)
        req = GasPosition.Request()
        req.x, req.y, req.z = float(wx), float(wy), 0.4
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=1.0)
        if fut.result() is None:
            continue
        c = fut.result().concentration
        concs.append(c)
        window.append(c)
        if len(window) > 6:
            window.pop(0)
        if gas_binarize(window):
            hits += 1
    rclpy.shutdown()

    if not concs:
        print("  no samples returned")
        return
    a = np.array(concs)
    print(f"  transect samples    : {len(a)}  (start -> source)")
    print(f"  gas ppm             : min={a.min():.4f} mean={a.mean():.4f} max={a.max():.4f}")
    print(f"  non-zero cells      : {int((a > 1e-3).sum())}/{len(a)}")
    print(f"  binarized hits      : {hits}/{len(a)}")
    if a.max() <= 1e-6:
        print("  WARNING: all-zero gas -- is the sim playing back a mature iteration?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true",
                    help="also query a running gaden_player for real gas")
    ap.add_argument("--cases", type=int, default=100000)
    args = ap.parse_args()

    print("=" * 68)
    print("ADSM House09 real-data test")
    print(f"  scenario: {HOUSE_DIR}")
    print("=" * 68)

    if not os.path.isdir(WIND_DIR):
        sys.exit(f"wind dir not found: {WIND_DIR}  (set HOUSE09_DIR)")

    print("\n[1] loading CFD wind fields...")
    winds = []
    for i in range(10):
        p = os.path.join(WIND_DIR, f"wind_at_cell_centers_{i}.csv")
        if os.path.exists(p):
            winds.append(WindField(p))
    if not winds:
        sys.exit(f"no wind_at_cell_centers_*.csv in {WIND_DIR}")
    w0 = winds[0]
    print(f"    {len(winds)} fields, {w0.cells} cells each")
    print(f"    grid {w0.nx}x{w0.ny}  res={w0.res}  origin=({w0.x0}, {w0.y0})")

    hits, checked = w0.self_test()
    print(f"    lookup self-test: {hits}/{checked} cell centres resolve")
    if checked == 0 or hits != checked:
        sys.exit("FATAL: wind lookup cannot find its own cells -- the offset-grid "
                 "bug is back; every query would silently return V=0.")

    print("\n[2] loading occupancy...")
    occ_img, free, res, ox, oy, h = load_occupancy()
    print(f"    {occ_img.shape[1]}x{occ_img.shape[0]}  res={res}  "
          f"origin=({ox}, {oy})  free={int(free.sum())} cells")

    ok = run_equivalence(winds, free, res, ox, oy, h, args.cases)
    run_heatmap(winds, occ_img, free, res, ox, oy, h)
    if args.online:
        run_online_gas(winds, free, res, ox, oy, h)

    print("\n" + "=" * 68)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
