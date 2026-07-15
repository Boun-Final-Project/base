"""Correlate the discriminative held-out CFD val (eval_cfd_holdout_disc.py sweep) against the
GADEN TEST scores from jobs 25316/25317/25318. If val rank-correlates with GADEN, the val is a
legitimate checkpoint selector and we never touch the test set for selection.

Usage: python corr_val_vs_gaden.py disc_val_sweep.txt
"""
import re
import sys
import numpy as np

# GADEN test overall success by update (jobs 25316 sweep @10ep; 25317 @20ep for upd4000/6000).
# steps = upd * 262144.  (10-ep unless noted; upd4000/6000 have 20-ep confirmations.)
GADEN = {   # steps : overall %
    131072000: 69, 262144000: 61, 393216000: 77, 524288000: 76,
    655360000: 73, 786432000: 71, 917504000: 73, 1048576000: 80,   # 20-ep
    1179648000: 79, 1310720000: 73, 1441792000: 77, 1572864000: 77, # upd6000 20-ep
    1703936000: 63, 1835008000: 61, 1966080000: 76, 2097152000: 66,
}

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])

def main(path):
    steps_re = re.compile(r"localwind_agent_(\d+)\.pt")
    rows = []
    for ln in open(path):
        m = steps_re.search(ln)
        if not m:
            continue
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", ln.split(".pt")[1])
        # order after name: mean_mind, s15, s25, s35, s50
        steps = int(m.group(1))
        mean_mind = float(nums[0]); s15, s25, s35, s50 = (float(v) for v in nums[1:5])
        rows.append((steps, mean_mind, s15, s25, s35, s50))
    rows.sort()
    print(f"{'upd':>5} {'steps(M)':>8} {'mean_mind':>9} {'s@.15':>6} {'s@.25':>6} {'s@.35':>6} {'s@.50':>6} {'GADEN':>6}")
    G, MM, S15, S25, S35 = [], [], [], [], []
    for steps, mm, s15, s25, s35, s50 in rows:
        g = GADEN.get(steps)
        upd = steps // 262144
        print(f"{upd:>5} {steps/1e6:>8.0f} {mm:>9.3f} {s15:>5.0f}% {s25:>5.0f}% {s35:>5.0f}% {s50:>5.0f}% {('%d%%'%g) if g else '  -':>6}")
        if g is not None:
            G.append(g); MM.append(mm); S15.append(s15); S25.append(s25); S35.append(s35)
    G = np.array(G, float)
    print(f"\nCorrelation of each val metric vs GADEN test overall (n={len(G)}):")
    print(f"  mean_mind (neg -> lower dist = better): spearman={spearman(-np.array(MM), G):+.3f}  pearson={pearson(-np.array(MM), G):+.3f}")
    for name, arr in [("s@0.15", S15), ("s@0.25", S25), ("s@0.35", S35)]:
        a = np.array(arr, float)
        print(f"  {name:>7}: spearman={spearman(a, G):+.3f}  pearson={pearson(a, G):+.3f}")
    # which checkpoint each metric would SELECT vs GADEN's pick
    best_gaden = rows[int(np.argmax([GADEN.get(r[0], -1) for r in rows]))]
    best_mind  = rows[int(np.argmin([r[1] for r in rows]))]
    best_s15   = rows[int(np.argmax([r[2] for r in rows]))]
    print(f"\nGADEN-test would pick   : upd {best_gaden[0]//262144}  (GADEN {GADEN.get(best_gaden[0])}%)")
    print(f"mean_mind val picks     : upd {best_mind[0]//262144}  (GADEN {GADEN.get(best_mind[0])}%)")
    print(f"s@0.15 val picks        : upd {best_s15[0]//262144}  (GADEN {GADEN.get(best_s15[0])}%)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "disc_val_sweep.txt")
