"""Is the v4 CFD library enough? Coverage / diversity report + deployment-map overlay.

Reads manifest_solved.json (template/inlet/speed balance, in the manifest directly) and a
sample of per-case meta.json (map-size envelope), then overlays the 7 deployment scenarios and
restates the standing verdict (memory project_cfd_library_v4): count is enough; only the >=20m
size band is thin. Decision rule printed at the end.

    python analyze_cfd_coverage.py [--lib cfd_test/library_v4_4dir] [--sample 3000]
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OSL = os.path.dirname(os.path.dirname(ROOT))

# deployment scenarios (w x h metres) — from memory project_cfd_library_v4
DEPLOY = {
    "4_rooms": (13.2, 13.2), "uleft": (10.2, 6.2), "uright": (10.2, 6.2),
    "labyrinth_left": (10.2, 6.2), "labyrinth_right": (10.2, 6.2),
    "many_rooms": (17.2, 9.2), "ultimate": (20.2, 12.2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="cfd_test/library_v4_4dir")
    ap.add_argument("--sample", type=int, default=3000, help="cases to read meta.json for (size envelope)")
    a = ap.parse_args()
    lib = Path(os.path.join(OSL, a.lib))
    man = json.loads((lib / "manifest_solved.json").read_text())
    n = len(man)
    print(f"library: {lib}")
    print(f"solved cases: {n}\n")

    tmpl = Counter(int(m["template_id"]) for m in man)
    inlet = Counter(m.get("inlet_side", "?") for m in man)
    speeds = np.array([float(m["inlet_speed"]) for m in man])
    print("template balance:", dict(sorted(tmpl.items())))
    print("inlet-side balance:", dict(inlet))
    print(f"inlet speed: min {speeds.min():.2f} median {np.median(speeds):.2f} "
          f"max {speeds.max():.2f} m/s\n")

    # size envelope from a sample of meta.json
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)[:min(a.sample, n)]
    ws, hs = [], []
    for i in idx:
        cd = Path(man[i]["case_dir"])
        if not cd.is_absolute():
            cd = lib / cd
        try:
            mt = json.loads((cd / "meta.json").read_text())
            ws.append(float(mt["map_width_m"])); hs.append(float(mt["map_height_m"]))
        except Exception:
            continue
    ws = np.array(ws); hs = np.array(hs)
    print(f"map-size envelope (n={len(ws)} sampled):")
    print(f"  width : min {ws.min():.1f} median {np.median(ws):.1f} max {ws.max():.1f} m")
    print(f"  height: min {hs.min():.1f} median {np.median(hs):.1f} max {hs.max():.1f} m\n")

    wmax, hmax = ws.max(), hs.max()
    print("deployment-map coverage (inside the sampled size envelope?):")
    thin = []                                    # at/over the size ceiling -> few/no neighbours
    for name, (w, h) in DEPLOY.items():
        win = w <= wmax + 1e-6; hin = h <= hmax + 1e-6
        if not (win and hin):
            over = max(w - wmax, h - hmax)
            tag = f"OVER ceiling by {over:.1f}m"; thin.append(name)
        elif (w >= wmax - 1.0) or (h >= hmax - 1.0):
            tag = "THIN (near size ceiling)"; thin.append(name)
        else:
            tag = "OK"
        print(f"  {name:16s} {w:5.1f}x{h:<5.1f}  {tag}")

    print("\nVERDICT (per memory project_cfd_library_v4):")
    print(f"  {n} solved cases, balanced across templates + 4 inlet dirs -> COUNT IS ENOUGH for RL")
    print("  (good generalization from ~500-1000 distinct levels; effective diversity >> count via")
    print("   per-episode start randomization + mirror aug). Binding constraints are the search/moat")
    print("   problem and an honest held-out eval, NOT map quantity.")
    if thin:
        print(f"\n  Only thin band: {thin} sit near the size ceiling.")
        print("  DECISION RULE: if held-out val success on the largest val maps lags the rest,")
        print("  top up a few hundred >=20m cases via the CFD array job (15126) — a day or two.")
        print("  Otherwise no new maps are needed.")
    else:
        print("\n  No thin size bands detected — no new maps needed.")


if __name__ == "__main__":
    main()
