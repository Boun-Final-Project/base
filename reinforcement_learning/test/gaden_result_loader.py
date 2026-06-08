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

    # per-type extra fields (bytes) BEFORE the trailing Vector3 pos + int gasType,
    # per GasSource::DeserializeBinary (src/GasSource.cpp).
    _GASSOURCE_EXTRA = {
        "point": 0,
        "box": 12,        # Vector3 size
        "line": 12,       # Vector3 lineEnd
        "sphere": 4,      # float radius
        "cylinder": 8,    # float radius + float height
    }

    def _parse_body(self, body: bytes):
        # Body layout (BufferWriter order, see RunningSimulation::SaveResults):
        #   i32 verMajor, i32 verMinor
        #   Description (40 B): Vec3i dims, Vec3 min, Vec3 max, f32 cellSize
        #   GasSource (var): string type + per-type fields + Vec3 pos + i32 gasType
        #   Constants (8 B): f32 totalMolesInFilament, f32 numMolesAllGasesIncm3
        #   i32 windIndex
        #   string mode  ("concentrations" | "filaments")
        #   vector payload (u64 count + count * {f32 grid | Filament})
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

        # GasSource (variable) — must be parsed to reach mode + payload.
        (slen,) = struct.unpack_from("<Q", body, off); off += 8
        stype = body[off:off + slen].decode("ascii", "replace"); off += slen
        off += self._GASSOURCE_EXTRA.get(stype, 0)
        self.source_pos = struct.unpack_from("<fff", body, off); off += 12
        (self.gas_type,) = struct.unpack_from("<i", body, off); off += 4

        # Constants (used for filament-mode concentration).
        total_moles, moles_all = struct.unpack_from("<ff", body, off); off += 8
        self._total_moles = float(total_moles)
        self._moles_all = float(moles_all)

        (self._wind_index,) = struct.unpack_from("<i", body, off); off += 4

        (mlen,) = struct.unpack_from("<Q", body, off); off += 8
        self.mode = body[off:off + mlen].decode("ascii", "replace"); off += mlen

        (count,) = struct.unpack_from("<Q", body, off); off += 8

        if self.mode == "concentrations":
            if count != n_cells:
                raise ValueError(f"concentration count {count} != numCells {n_cells}")
            flat = np.frombuffer(body, dtype="<f4", count=n_cells, offset=off)
            # indexFrom3D = x + y*nx + z*nx*ny  → reshape (nz, ny, nx)
            self.grid = flat.reshape(dz, dy, dx).astype(np.float32)
            self.filaments = None
        elif self.mode == "filaments":
            # Filament POD = Vector3 position (3 f32) + float sigma = 16 B.
            fil = np.frombuffer(body, dtype="<f4", count=count * 4, offset=off)
            fil = fil.reshape(count, 4)
            self.filaments = {
                "pos": fil[:, :3].astype(np.float64),     # (N,3) metres
                "sigma": fil[:, 3].astype(np.float64),    # (N,)  GADEN sigma (cm)
            }
            self.grid = None
        else:
            raise ValueError(f"unknown payload mode '{self.mode}'")

    # ------------------------------------------------------------------
    def _conc_at_center(self, sigma):
        """ppm at a filament's center — GADEN Simulation::ConcentrationAtCenter.

        numMolesTarget_cm3 = totalMolesInFilament / (sqrt(8 pi^3) * sigma^3)
        ppm = 1e6 * numMolesTarget_cm3 / numMolesAllGasesIncm3
        (sigma is GADEN's filament sigma, in cm.)
        """
        denom = np.sqrt(8.0 * np.pi ** 3) * sigma ** 3
        num_moles = self._total_moles / denom
        return 1e6 * num_moles / self._moles_all

    def concentration_at(self, x: float, y: float, z: float = 0.5) -> float:
        """Gas concentration [ppm] at world coords (env frame).

        concentrations mode: nearest-cell lookup (matches coordsToIndices).
        filaments mode: sum of Gaussian filament kernels exactly as GADEN's
        Simulation::CalculateConcentration — per filament within 3*sigma/100 m,
            ppm = ConcentrationAtCenter(sigma) * exp(-dist_cm^2 / (2 sigma^2))
        with dist_cm = 100 * |fil.pos - sample|. Line-of-sight wall occlusion
        (GADEN's CheckLineOfSight) is NOT applied here — the loader has no
        occupancy grid; pass an occupancy to ``concentration_field_slice`` if
        you need it. For point queries the LoS effect is usually small.
        Out-of-bounds returns 0.0.
        """
        nx, ny, nz = self.dims
        ix = int((x - self.min_coord[0]) / self.cell_size)
        iy = int((y - self.min_coord[1]) / self.cell_size)
        iz = int((z - self.min_coord[2]) / self.cell_size)
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            return 0.0
        if self.mode == "concentrations":
            return float(self.grid[iz, iy, ix])
        # filaments mode
        f = self.filaments
        if f is None or len(f["sigma"]) == 0:
            return 0.0
        sample = np.array([x, y, z], dtype=np.float64)
        d = f["pos"] - sample                              # (N,3)
        dist2_m = np.einsum("ij,ij->i", d, d)              # squared metres
        sigma = f["sigma"]                                 # cm
        limit_m = 3.0 * sigma / 100.0                      # GADEN's 3-sigma cutoff
        near = dist2_m < limit_m * limit_m
        if not near.any():
            return 0.0
        dist_cm2 = dist2_m[near] * 1e4                     # (100*m)^2
        s = sigma[near]
        ppm = self._conc_at_center(s) * np.exp(-dist_cm2 / (2.0 * s * s))
        return float(ppm.sum())

    def slice_z(self, z: float = 0.5) -> np.ndarray:
        """Return the (ny, nx) horizontal concentration slice at height z.

        concentrations mode: direct grid slice. filaments mode: evaluate the
        kernel sum at every cell center on that z-plane (vectorized).
        """
        nx, ny, nz = self.dims
        iz = int((z - self.min_coord[2]) / self.cell_size)
        iz = max(0, min(nz - 1, iz))
        if self.mode == "concentrations":
            return self.grid[iz]
        # filaments: evaluate at each cell center on the plane.
        zc = self.min_coord[2] + (iz + 0.5) * self.cell_size
        xs = self.min_coord[0] + (np.arange(nx) + 0.5) * self.cell_size
        ys = self.min_coord[1] + (np.arange(ny) + 0.5) * self.cell_size
        out = np.zeros((ny, nx), dtype=np.float32)
        f = self.filaments
        if f is None or len(f["sigma"]) == 0:
            return out
        # Loop over filaments (typically a few thousand) accumulating onto grid.
        for p, sg in zip(f["pos"], f["sigma"]):
            limit_m = 3.0 * sg / 100.0
            if abs(p[2] - zc) > limit_m:
                continue
            dx2 = (xs - p[0]) ** 2
            dy2 = (ys - p[1]) ** 2
            dz2 = (zc - p[2]) ** 2
            dist2 = dx2[None, :] + dy2[:, None] + dz2
            mask = dist2 < limit_m * limit_m
            if not mask.any():
                continue
            cc = self._conc_at_center(sg)
            out[mask] += (cc * np.exp(-(dist2[mask] * 1e4) / (2.0 * sg * sg))).astype(np.float32)
        return out


class ReplayGasSource:
    """Drop-in gas source backed by real GADEN result/iteration_<N> snapshots.

    Matches the subset of the FilamentPlume interface that GasSourceEnv uses
    (``update()``, ``concentration_at(pos)``, ``get_all_filaments()``), so the
    env can query the *real* GADEN concentration field instead of the surrogate
    plume — eliminating the largest eval fidelity gap (the labyrinth gas field
    is near-homogeneous in real GADEN but the surrogate produces a different,
    more followable field).

    Coordinate handling: the env works in env-frame (origin-shifted) positions;
    GADEN snapshots are in absolute world coords. ``origin_xy`` (the map's
    env_min) converts env→absolute. Time: saveDeltaTime = 0.5 s per snapshot, so
    one env step advances one snapshot. Starts at ``start_iteration`` (=
    start_time / saveDeltaTime) and clamps to the last available snapshot.
    """

    def __init__(self, result_dir, origin_xy, start_iteration: int,
                 z: float = 0.5):
        self._seq = GadenResultSequence(result_dir)
        self._avail = self._seq.available_iterations()
        self._max_iter = self._avail[-1] if self._avail else 0
        self._ox, self._oy = float(origin_xy[0]), float(origin_xy[1])
        self._z = float(z)
        self._iter = max(0, min(int(start_iteration), self._max_iter))
        self._seq.load_iteration(self._iter)

    def update(self):
        # One env step == one saved snapshot (saveDeltaTime). Hold the last
        # snapshot once the recording runs out (short result dirs, e.g. u_left).
        if self._iter < self._max_iter:
            self._iter += 1
            self._seq.load_iteration(self._iter)

    def concentration_at(self, pos):
        return self._seq.concentration_at(pos[0] + self._ox,
                                          pos[1] + self._oy, self._z)

    def get_all_filaments(self):
        # For viz compatibility; return absolute filament positions if present.
        if self._seq.filaments is not None:
            return {"positions": self._seq.filaments["pos"][:, :2].copy()}
        return {"positions": np.zeros((0, 2))}


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
