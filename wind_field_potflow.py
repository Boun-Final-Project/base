"""
Spatially-varying wind field for the fast-bundle training pipeline.

Replaces the uniform `WindModel`. Per episode:
  1. Sample mean direction θ ∈ [0, 2π) and speed s ∈ speed_range.
  2. Solve ∇²φ = 0 on a *box-with-open-edges* version of the room. The
     four array-edge cells are treated as Dirichlet inflow/outflow
     (φ = -s · (cosθ · x + sinθ · y)); interior obstacle walls are
     Neumann (∂φ/∂n = 0). This gives a non-trivial steady flow for
     closed training rooms whose outer ring is solid wall.
  3. Velocity (Ux, Uy) = -∇φ.
  4. Optionally overlay a divergence-free curl-noise field for stationary
     eddies (closes part of the irrotational-vs-CFD gap).
  5. Output velocity is zeroed at all wall cells (interior and outer).

The plume queries `local_batch(positions)` per filament. The CNN policy
still consumes the *spatial mean* of the field as a 4-D ctx vector
(`[speed/MAX, cosθ, sinθ, t]`), so existing 5-channel checkpoints can be
fine-tuned without architecture surgery.

Implementation notes
--------------------
* Sparse Laplace matrix is built fully vectorised (no per-cell Python
  loop) and solved with `scipy.sparse.linalg.spsolve`. ~30 ms on
  100×60 grids.
* Curl noise: ψ = smoothed Gaussian random field, overlay = (∂ψ/∂y, -∂ψ/∂x).
  Divergence-free, so it doesn't violate mass conservation.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter, label


class WindField:
    """Spatially-varying steady wind from 2D potential flow + curl noise."""

    def __init__(self,
                 speed_range=(0.1, 1.5),
                 max_speed: float = 2.0,
                 curl_noise_amplitude: float = 0.0,
                 curl_noise_scale: float = 4.0,
                 ctx_uses_sampled_wind: bool = True):
        """
        Parameters
        ----------
        speed_range : (float, float)
            (min, max) sampled mean wind speed per episode (m/s).
        max_speed : float
            Normalisation constant for ctx vector (matches training-time
            WIND_MAX_SPEED).
        curl_noise_amplitude : float
            Std-dev of the underlying scalar field whose curl is added as
            divergence-free turbulence. 0 = pure potential flow.
        curl_noise_scale : float
            Smoothing length (in cells) of the noise scalar field.
        ctx_uses_sampled_wind : bool
            If True (default), `get_observation_spatial()` returns the
            *sampled* inflow speed/direction (the boundary-condition
            parameters). This matches how the original uniform `WindModel`
            populated ctx and therefore preserves ctx semantics for
            fine-tuning existing checkpoints.
            If False, returns the spatial mean of the actual velocity field
            (which is ≈ 0 for closed rooms by mass conservation — only
            useful when the room has explicit openings, e.g. when matching
            deployment's GADEN-csv_mean ctx exactly).
        """
        self.speed_range = speed_range
        self.max_speed = float(max_speed)
        self.curl_amp = float(curl_noise_amplitude)
        self.curl_scale = float(curl_noise_scale)
        self.ctx_uses_sampled_wind = bool(ctx_uses_sampled_wind)

        # Per-episode state
        self.speed: float = 0.0
        self.direction: float = 0.0
        self.Ux: np.ndarray | None = None       # (H, W)
        self.Uy: np.ndarray | None = None
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0
        self._dx: float = 0.0
        self._H: int = 0
        self._W: int = 0
        self._mean_speed_field: float = 0.0
        self._mean_dir_field: float = 0.0

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------

    def randomize(self, occ_grid, rng=None):
        """Sample (speed, direction) and build the field for this episode."""
        if rng is None:
            rng = np.random.default_rng()
        self.speed = float(rng.uniform(*self.speed_range))
        self.direction = float(rng.uniform(0.0, 2.0 * np.pi))
        self._build_field(occ_grid, rng)

    def _build_field(self, occ_grid, rng):
        walls = (occ_grid.grid != 0).astype(bool)
        self._H, self._W = walls.shape
        self._dx = float(occ_grid.resolution)
        self._origin_x = float(getattr(occ_grid, 'origin_x', 0.0) or 0.0)
        self._origin_y = float(getattr(occ_grid, 'origin_y', 0.0) or 0.0)

        Ux, Uy = _solve_potential_flow(
            walls, self._dx, self.direction, self.speed,
            ox=self._origin_x, oy=self._origin_y,
        )

        if self.curl_amp > 0:
            psi = rng.normal(0.0, self.curl_amp, size=walls.shape)
            psi = gaussian_filter(psi, sigma=self.curl_scale)
            # curl(ψ ẑ) = (∂ψ/∂y, -∂ψ/∂x), divergence-free in 2D.
            curl_x = np.gradient(psi, self._dx, axis=0)
            curl_y = -np.gradient(psi, self._dx, axis=1)
            Ux = Ux + curl_x
            Uy = Uy + curl_y

        Ux[walls] = 0.0
        Uy[walls] = 0.0
        self.Ux = Ux.astype(np.float32)
        self.Uy = Uy.astype(np.float32)

        free = ~walls
        if free.any():
            mu_x = float(self.Ux[free].mean())
            mu_y = float(self.Uy[free].mean())
        else:
            mu_x = self.speed * np.cos(self.direction)
            mu_y = self.speed * np.sin(self.direction)
        self._mean_speed_field = float(np.hypot(mu_x, mu_y))
        self._mean_dir_field = float(np.arctan2(mu_y, mu_x)) % (2.0 * np.pi)

    # ------------------------------------------------------------------
    # Plume / observation queries
    # ------------------------------------------------------------------

    def local_batch(self, positions: np.ndarray) -> np.ndarray:
        """Wind vectors at world positions. positions: (N, 2) → (N, 2)."""
        if self.Ux is None:
            raise RuntimeError("WindField.randomize() must be called first.")
        if positions.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        col = np.floor((positions[:, 0] - self._origin_x) / self._dx).astype(np.int32)
        row = np.floor((positions[:, 1] - self._origin_y) / self._dx).astype(np.int32)
        col = np.clip(col, 0, self._W - 1)
        row = np.clip(row, 0, self._H - 1)
        out = np.empty((positions.shape[0], 2), dtype=np.float32)
        out[:, 0] = self.Ux[row, col]
        out[:, 1] = self.Uy[row, col]
        return out

    # Drop-in API matching the old WindModel
    def _ctx_speed_dir(self):
        if self.ctx_uses_sampled_wind:
            return self.speed, self.direction
        return self._mean_speed_field, self._mean_dir_field

    def get_observation(self):
        s, d = self._ctx_speed_dir()
        return (s / self.max_speed, d / (2.0 * np.pi))

    def get_observation_spatial(self):
        s, d = self._ctx_speed_dir()
        return (s / self.max_speed, float(np.cos(d)), float(np.sin(d)))

    def get_observation_spatial_at(self, position):
        """Local-cell ctx: return (speed/max, cos, sin) at a world position.

        Used by the spatial obs wrapper so the policy ctx reflects the wind
        the agent actually experiences at its current cell, not a global
        summary. This couples ctx semantics between training (local) and
        deployment (also local, queried from the GADEN field), eliminating
        the ctx-vs-local mismatch we observed on real CFD wind fields.
        """
        local = self.local_batch(np.asarray([position], dtype=np.float64))[0]
        spd = float(np.hypot(local[0], local[1]))
        if spd > 1e-8:
            return (spd / self.max_speed,
                    float(local[0]) / spd,
                    float(local[1]) / spd)
        # Degenerate cell (wall or zero wind): return zero-magnitude ctx.
        return (0.0, 1.0, 0.0)

    def get_dispersion_offset(self, dispersion_factor: float):
        s, d = self._ctx_speed_dir()
        dx = s * dispersion_factor * np.cos(d)
        dy = s * dispersion_factor * np.sin(d)
        return np.array([dx, dy], dtype=np.float64)


# ----------------------------------------------------------------------
# Sparse Laplace solver — vectorised matrix construction
# ----------------------------------------------------------------------
def _solve_potential_flow(walls: np.ndarray, dx: float,
                          mean_dir: float, mean_speed: float,
                          ox: float = 0.0, oy: float = 0.0):
    """Potential-flow velocity for a closed/bounded grid.

    Treats array-edge cells as open-boundary Dirichlet (φ = -s(cosθ·x + sinθ·y))
    so that closed training rooms with solid outer walls still produce a
    non-trivial steady flow. Interior obstacle walls are Neumann.

    Returns (Ux, Uy) with zero velocity at original wall cells.
    """
    H, W = walls.shape
    n = H * W

    # Cell-centre world coordinates (origin-shifted).
    xs = ox + (np.arange(W) + 0.5) * dx
    ys = oy + (np.arange(H) + 0.5) * dx
    XX, YY = np.meshgrid(xs, ys)

    cosT = float(np.cos(mean_dir))
    sinT = float(np.sin(mean_dir))
    bdy_phi = -mean_speed * (cosT * XX + sinT * YY)

    # Cell type masks. The "outer wall" of the room may be several cells
    # thick (training maps draw walls at the array edge with thickness 1-3
    # cells); we need to treat that entire outer ring as Dirichlet inflow/
    # outflow so flow can penetrate to the interior. We find the outer ring
    # by flood-filling walls 4-connected to the array edge.
    is_edge_seed = np.zeros_like(walls, dtype=bool)
    is_edge_seed[0, :]  = walls[0, :]
    is_edge_seed[-1, :] = walls[-1, :]
    is_edge_seed[:, 0]  = walls[:, 0]
    is_edge_seed[:, -1] = walls[:, -1]

    if is_edge_seed.any():
        struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        labels, _ = label(walls.astype(np.int8), structure=struct4)
        edge_labels = np.unique(labels[is_edge_seed])
        outer_wall = np.isin(labels, edge_labels) & walls
    else:
        # No walls touch the array edge — treat array edges themselves as Dirichlet.
        outer_wall = np.zeros_like(walls)
        outer_wall[0, :] = True; outer_wall[-1, :] = True
        outer_wall[:, 0] = True; outer_wall[:, -1] = True

    is_int_wall = walls & ~outer_wall       # interior obstacles only (Neumann)
    is_dir = outer_wall                     # Dirichlet rows = outer-ring cells
    is_int = ~walls & ~is_dir               # interior free cells (solve here)

    # Diagonal: -count of free/edge neighbours for interior cells; 1 for pinned rows.
    # A free or edge neighbour contributes (we read its φ); an interior-wall neighbour
    # is mirrored (Neumann), so it doesn't appear at all in the row.
    n_free_n = np.zeros(walls.shape, dtype=np.int32)
    n_free_n[1:, :]  += (~is_int_wall[:-1, :]).astype(np.int32)   # north
    n_free_n[:-1, :] += (~is_int_wall[1:, :]).astype(np.int32)    # south
    n_free_n[:, 1:]  += (~is_int_wall[:, :-1]).astype(np.int32)   # west
    n_free_n[:, :-1] += (~is_int_wall[:, 1:]).astype(np.int32)    # east

    flat = lambda M: M.reshape(-1)
    diag = np.empty(n, dtype=np.float64)
    # Pinned rows: outer-wall cells (Dirichlet) + interior obstacles (decoupled with φ=0)
    diag[flat(is_dir | is_int_wall)] = 1.0
    diag[flat(is_int)] = -np.maximum(flat(n_free_n)[flat(is_int)], 1).astype(np.float64)

    rows = [np.arange(n)]
    cols = [np.arange(n)]
    vals = [diag]

    idx_grid = np.arange(n).reshape(H, W)

    def _add_offdiag(di, dj):
        # interior cell (i, j) has a contributing neighbour at (i+di, j+dj)
        # only if neighbour is in-array AND not an interior wall.
        mask_int = is_int.copy()
        if di == -1: mask_int[0, :]  = False
        if di == +1: mask_int[-1, :] = False
        if dj == -1: mask_int[:, 0]  = False
        if dj == +1: mask_int[:, -1] = False

        # neighbour mask: shifted is_int_wall
        nb_int_wall = np.zeros_like(walls)
        if di == -1: nb_int_wall[1:, :]  = is_int_wall[:-1, :]; nb_int_wall[0, :]  = False
        if di == +1: nb_int_wall[:-1, :] = is_int_wall[1:, :];  nb_int_wall[-1, :] = False
        if dj == -1: nb_int_wall[:, 1:]  = is_int_wall[:, :-1]; nb_int_wall[:, 0]  = False
        if dj == +1: nb_int_wall[:, :-1] = is_int_wall[:, 1:];  nb_int_wall[:, -1] = False

        use = mask_int & ~nb_int_wall
        if not use.any():
            return
        i_idx, j_idx = np.where(use)
        target = (i_idx + di) * W + (j_idx + dj)
        rows.append(idx_grid[use])
        cols.append(target)
        vals.append(np.ones(use.sum(), dtype=np.float64))

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        _add_offdiag(di, dj)

    A = csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n)
    )

    b = np.zeros(n, dtype=np.float64)
    b[flat(outer_wall)] = flat(bdy_phi)[flat(outer_wall)]
    # is_int_wall rows: b stays 0 → φ=0 inside obstacles (their φ never
    # enters any interior free cell's laplacian thanks to Neumann mirroring).

    phi = spsolve(A, b)
    phi_2d = phi.reshape(H, W)

    # Custom gradient that ignores wall neighbours.
    # `np.gradient` would use phi=0 inside walls, creating huge spurious
    # velocities at the wall-free interface. Instead, use central diff where
    # both opposite neighbours are non-wall, otherwise one-sided diff toward
    # the available free neighbour. This is consistent with the Neumann BC.
    Ux = -_wall_aware_gradient(phi_2d, walls, dx, axis=1)
    Uy = -_wall_aware_gradient(phi_2d, walls, dx, axis=0)
    return Ux, Uy


def _wall_aware_gradient(phi: np.ndarray, walls: np.ndarray,
                         dx: float, axis: int) -> np.ndarray:
    """∂φ/∂axis using only non-wall neighbours. Vectorised."""
    H, W = phi.shape
    grad = np.zeros_like(phi)
    if axis == 1:
        # gradient in x (column direction): central uses (j-1, j+1), forward (j, j+1), backward (j-1, j)
        west_free = np.zeros_like(walls)
        west_free[:, 1:] = ~walls[:, :-1]
        east_free = np.zeros_like(walls)
        east_free[:, :-1] = ~walls[:, 1:]

        central = np.zeros_like(phi)
        central[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx)

        forward = np.zeros_like(phi)
        forward[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / dx

        backward = np.zeros_like(phi)
        backward[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / dx
    else:
        north_free = np.zeros_like(walls)
        north_free[1:, :] = ~walls[:-1, :]
        south_free = np.zeros_like(walls)
        south_free[:-1, :] = ~walls[1:, :]

        central = np.zeros_like(phi)
        central[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)

        forward = np.zeros_like(phi)
        forward[:-1, :] = (phi[1:, :] - phi[:-1, :]) / dx

        backward = np.zeros_like(phi)
        backward[1:, :] = (phi[1:, :] - phi[:-1, :]) / dx

        west_free, east_free = north_free, south_free  # reuse names below

    use_central  = west_free & east_free
    use_forward  = east_free & ~west_free   # only east/south available
    use_backward = west_free & ~east_free   # only west/north available

    grad = np.where(use_central, central,
           np.where(use_forward, forward,
           np.where(use_backward, backward, 0.0)))
    grad[walls] = 0.0
    return grad
