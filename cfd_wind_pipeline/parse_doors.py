"""Parse a binary STL with multiple disjoint surfaces (doors), report each one's
bounding box so we can identify inlet/outlet positions."""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def parse_binary_stl(path):
    """Return triangles as (N, 3, 3) array of vertices."""
    with open(path, 'rb') as f:
        header = f.read(80)
        n_tri = struct.unpack('<I', f.read(4))[0]
        tris = np.zeros((n_tri, 3, 3), dtype=np.float32)
        for i in range(n_tri):
            f.read(12)  # normal
            for j in range(3):
                tris[i, j] = struct.unpack('<fff', f.read(12))
            f.read(2)  # attribute
    return tris


def cluster_into_components(tris, tol=0.01):
    """Connected components by shared vertices. Returns list of triangle-index arrays."""
    # Build vertex → triangle index mapping via voxelized vertex coordinates
    n = len(tris)
    # Round each vertex to tol cells for hashing
    keys = (tris / tol).round().astype(np.int64)
    vert_to_tri = {}
    for ti in range(n):
        for vi in range(3):
            k = tuple(keys[ti, vi])
            vert_to_tri.setdefault(k, []).append(ti)
    # Union-find over triangles sharing any vertex
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for verts in vert_to_tri.values():
        for v in verts[1:]:
            union(verts[0], v)
    # Group triangles by root
    groups = {}
    for ti in range(n):
        groups.setdefault(find(ti), []).append(ti)
    return list(groups.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stl', required=True)
    args = p.parse_args()

    tris = parse_binary_stl(args.stl)
    print(f"{len(tris)} triangles total")
    groups = cluster_into_components(tris)
    print(f"{len(groups)} connected components (\"doors\")")
    print()
    print(f"{'Door':>4s}  {'nTri':>5s}  {'x_min':>7s} {'x_max':>7s}  "
          f"{'y_min':>7s} {'y_max':>7s}  {'z_min':>7s} {'z_max':>7s}  {'wall_side':>10s}")
    door_info = []
    for i, group in enumerate(sorted(groups, key=len, reverse=True)):
        verts = tris[group].reshape(-1, 3)
        mn = verts.min(axis=0)
        mx = verts.max(axis=0)
        # Which side of the bounding box does it sit on?
        # We'll learn this after seeing the global bbox
        door_info.append((i, len(group), mn, mx))

    # Global bbox
    all_verts = tris.reshape(-1, 3)
    gmn = all_verts.min(axis=0)
    gmx = all_verts.max(axis=0)
    print(f"global bbox: x=[{gmn[0]:.2f},{gmx[0]:.2f}] y=[{gmn[1]:.2f},{gmx[1]:.2f}] z=[{gmn[2]:.2f},{gmx[2]:.2f}]")
    print()
    for i, n_t, mn, mx in door_info:
        # Identify which wall side this door is on (smallest gap to global bbox edge)
        side = ""
        if mn[0] - gmn[0] < 0.2: side = "WEST"
        elif gmx[0] - mx[0] < 0.2: side = "EAST"
        elif mn[1] - gmn[1] < 0.2: side = "SOUTH"
        elif gmx[1] - mx[1] < 0.2: side = "NORTH"
        else: side = "INTERIOR"
        print(f"{i:>4d}  {n_t:>5d}  {mn[0]:>7.2f} {mx[0]:>7.2f}  "
              f"{mn[1]:>7.2f} {mx[1]:>7.2f}  {mn[2]:>7.2f} {mx[2]:>7.2f}  {side:>10s}  "
              f"size {mx[0]-mn[0]:.2f}x{mx[1]-mn[1]:.2f}x{mx[2]-mn[2]:.2f}")


if __name__ == '__main__':
    main()
