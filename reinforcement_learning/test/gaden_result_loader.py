"""
Replay real GADEN gas fields in Python (maximal fidelity, no surrogate plume).

GADEN's filament_simulator writes one snapshot per saveDeltaTime to
``<sim>/result/iteration_<N>``. With ``preCalculateConcentrations: true`` in
sim.yaml, each snapshot stores the fully-computed per-cell gas concentration
[ppm] — the exact field the deployment ROS node samples via the player. This
loader reads those files directly so the Python eval harness can query real
GADEN concentrations instead of the approximate FilamentPlume.

  >>> seq = GadenResultSequence("/path/<sim>/result")
  >>> seq.load_iteration(200)               # snapshot index (== start_time/saveDeltaTime)
  >>> seq.concentration_at(x, y, z=0.5)     # ppm at world coords (env frame)

WHY: Results/GADEN_EVAL_FINDINGS.md §4 — big-map eval failures are a
gas-availability/fidelity problem in the surrogate plume, not a policy problem.
Replaying GADEN's own field removes the surrogate from the loop entirely and
answers e.g. "does real gas actually reach the ultimate robot start?".

------------------------------------------------------------------------------
FILE FORMAT  (gaden_core post-3.0, verified against
gaden_core/include/gaden/internal/Serialization.hpp + src/RunningSimulation.cpp
SaveResults() + src/PlaybackSimulation.cpp LoadLogfile())

Uncompressed header (little-endian):
    char    resultIdentifier[13]   # "GADEN_RESULT\0"  (sizeof includes NUL)
    uint8   compressionMode        # 0=UNCOMPRESSED 1=ZLIB 2=LIBBSC
    uint64  uncompressedSize
Then a compressed blob. After decompression, the body (BufferWriter order):
    int32   versionMajor
    int32   versionMinor
    Description  (40 bytes POD):
        int32 dim.x, dim.y, dim.z
        float min.x, min.y, min.z
        float max.x, max.y, max.z
        float cellSize
    GasSource (variable): string sourceType + per-type fields
                          + Vector3 sourcePosition + int gasType
    Constants (8 bytes): float totalMolesInFilament, float numMolesAllGasesIncm3
    int32   windIndex
    string  mode            # size_t length + bytes; "concentrations" or "filaments"
    vector  payload         # size_t count + count*float32 (concentrations)
                            # or count*Filament (filaments; not handled here)

A "string"/"vector" is size_t(uint64) length-prefix + raw bytes.

Cell ordering (gaden indexFrom3D): index = x + y*nx + z*nx*ny
  → reshape flat float array as (nz, ny, nx).

This loader targets concentration mode. It parses version+Description from the
front (giving grid dims/cellSize/minCoord), then extracts the float grid from
the TAIL — the concentration vector is the last thing written, so the final
``nx*ny*nz`` floats (preceded by a uint64 count == nx*ny*nz) are the grid. This
avoids parsing the variable-length GasSource. We assert the tail count matches
numCells as an integrity check.

LIBBSC compression (files > 5 MB) is NOT decodable with stdlib; those raise a
clear error (use ZLIB by keeping per-snapshot size down, or decode via gaden).
Most single-gas 2D-ish maps compress under 5 MB → ZLIB.

UNVALIDATED on this box: no iteration_* files exist here. Generate one on the
GADEN machine (preCalculateConcentrations: true, saveResults: true) and verify
against the player's SampleConcentrations before trusting.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_MAGIC = b"GADEN_RESULT\x00"          # 13 bytes (sizeof includes NUL terminator)
_DESC_BYTES = 3 * 4 + 3 * 4 + 3 * 4 + 4   # Vec3i + Vec3 + Vec3 + float = 40


class GadenResultSequence:
    """Reads ``result/iteration_<N>`` concentration snapshots for one sim."""

    def __init__(self, result_dir: str | Path):
        self.result_dir = Path(result_dir)
        if not self.result_dir.is_dir():
            raise FileNotFoundError(f"result dir not found: {self.result_dir}")
        # Per-iteration state (filled by load_iteration).
        self.dims = None            # (nx, ny, nz)
        self.min_coord = None       # (x, y, z) metres — env-frame origin
        self.cell_size = None       # metres
        self.grid = None            # (nz, ny, nx) float32 ppm
        self._loaded_iter = None

    # ------------------------------------------------------------------
    def iteration_path(self, n: int) -> Path:
        return self.result_dir / f"iteration_{n}"

    def available_iterations(self):
        out = []
        for p in self.result_dir.glob("iteration_*"):
            try:
                out.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return sorted(out)

    # ------------------------------------------------------------------
    def load_iteration(self, n: int):
        """Parse snapshot *n* into ``self.grid`` + geometry. Returns self."""
        path = self.iteration_path(n)
        if not path.is_file():
            avail = self.available_iterations()
            raise FileNotFoundError(
                f"{path} not found. Available: {avail[:3]}..{avail[-3:] if avail else []}"
            )
        raw = path.read_bytes()
        body = self._decompress(raw)
        self._parse_body(body)
        self._loaded_iter = n
        return self

    # ------------------------------------------------------------------
    def _decompress(self, raw: bytes) -> bytes:
        if raw[:13] != _MAGIC:
            raise ValueError(
                f"not a post-3.0 GADEN result file (bad magic {raw[:13]!r}). "
                "Pre-3.0 files are unsupported by this loader."
            )
        off = 13
        comp_mode = raw[off]; off += 1
        (uncompressed_size,) = struct.unpack_from("<Q", raw, off); off += 8
        blob = raw[off:]
        if comp_mode == 0:          # UNCOMPRESSED
            return blob[:uncompressed_size]
        if comp_mode == 1:          # ZLIB
            out = zlib.decompress(blob)
            if len(out) != uncompressed_size:
                raise ValueError(
                    f"zlib output {len(out)} != declared {uncompressed_size}"
                )
            return out
        if comp_mode == 2:          # LIBBSC
            raise NotImplementedError(
                "Snapshot uses LIBBSC compression (file > 5 MB). stdlib can't "
                "decode it. Regenerate with smaller per-snapshot size, or decode "
                "via the gaden player. (ZLIB is used automatically below 5 MB.)"
            )
        raise ValueError(f"unknown compression mode {comp_mode}")

    def _parse_body(self, body: bytes):
        # Front: version (2x int32) + Description (40 B POD).
        off = 0
        ver_major, ver_minor = struct.unpack_from("<ii", body, off); off += 8
        if ver_major < 3:
            raise NotImplementedError(
                f"file version {ver_major}.{ver_minor} < 3.0 — body layout differs; "
                "use the gaden player for legacy files."
            )
        dx, dy, dz = struct.unpack_from("<iii", body, off); off += 12
        min_x, min_y, min_z = struct.unpack_from("<fff", body, off); off += 12
        _max = struct.unpack_from("<fff", body, off); off += 12
        (cell_size,) = struct.unpack_from("<f", body, off); off += 4

        self.dims = (dx, dy, dz)
        self.min_coord = (min_x, min_y, min_z)
        self.cell_size = float(cell_size)
        n_cells = dx * dy * dz
        if n_cells <= 0:
            raise ValueError(f"bad grid dims {self.dims}")

        # Tail: concentration vector is the last write — [uint64 count][count f32].
        # Robustly grab the final n_cells floats and verify the preceding count.
        grid_bytes = n_cells * 4
        if len(body) < grid_bytes + 8:
            raise ValueError(
                f"body too short ({len(body)} B) for {n_cells} cells + count"
            )
        count_off = len(body) - grid_bytes - 8
        (count,) = struct.unpack_from("<Q", body, count_off)
        if count != n_cells:
            raise ValueError(
                f"tail vector count {count} != numCells {n_cells}. File may be "
                "'filaments' mode (regenerate with preCalculateConcentrations: true) "
                "or layout mismatch."
            )
        flat = np.frombuffer(body, dtype="<f4", count=n_cells, offset=count_off + 8)
        # indexFrom3D = x + y*nx + z*nx*ny  → reshape (nz, ny, nx)
        self.grid = flat.reshape(dz, dy, dx).astype(np.float32)

    # ------------------------------------------------------------------
    def concentration_at(self, x: float, y: float, z: float = 0.5) -> float:
        """Gas concentration [ppm] at world coords (env frame). Nearest cell.

        Matches gaden Environment::coordsToIndices: floor((coord - min)/cellSize).
        Out-of-bounds queries return 0.0.
        """
        if self.grid is None:
            raise RuntimeError("call load_iteration() first")
        nx, ny, nz = self.dims
        ix = int((x - self.min_coord[0]) / self.cell_size)
        iy = int((y - self.min_coord[1]) / self.cell_size)
        iz = int((z - self.min_coord[2]) / self.cell_size)
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            return 0.0
        return float(self.grid[iz, iy, ix])

    def slice_z(self, z: float = 0.5) -> np.ndarray:
        """Return the (ny, nx) horizontal concentration slice at height z."""
        if self.grid is None:
            raise RuntimeError("call load_iteration() first")
        iz = int((z - self.min_coord[2]) / self.cell_size)
        iz = max(0, min(self.dims[2] - 1, iz))
        return self.grid[iz]


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Inspect a GADEN concentration snapshot.")
    p.add_argument("--result-dir", required=True, help="path to <sim>/result")
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--at", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="query concentration at world coords")
    args = p.parse_args()

    seq = GadenResultSequence(args.result_dir).load_iteration(args.iteration)
    nx, ny, nz = seq.dims
    sl = seq.slice_z(args.at[2] if args.at else 0.5)
    print(f"iteration {args.iteration}: dims={seq.dims} cell={seq.cell_size} "
          f"min={seq.min_coord}")
    print(f"  z-slice nonzero cells: {(sl > 0).sum()}/{sl.size}  "
          f"max ppm: {sl.max():.4f}  mean(nonzero): "
          f"{sl[sl > 0].mean() if (sl > 0).any() else 0:.4f}")
    if args.at:
        print(f"  concentration_at{tuple(args.at)} = "
              f"{seq.concentration_at(*args.at):.4f} ppm")


if __name__ == "__main__":
    _cli()
