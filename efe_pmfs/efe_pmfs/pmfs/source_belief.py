"""
Discrete source belief over a fixed candidate lattice.

No particle filter. Candidate sources sit on a stride-downsampled lattice over
the (known) map bounding box. Each tick:
  - Invalidate candidates that fall into walls in the current SLAM grid.
  - If refresh is requested: run filament simulations from every valid candidate,
    cache predicted hit maps f^{S_k}, and recompute p(S|H) by comparing against
    the measured hit map weighted by per-cell confidence α_i.

Mutual information at a candidate planner cell v is then cheap:
   p(hit|v)   = Σ_k f_S[k, v] · p_S[k]
   p(miss|v)  = 1 − p(hit|v)
   p(S_k | hit at v)  ∝ f_S[k, v]       · p_S[k]
   p(S_k | miss at v) ∝ (1 − f_S[k, v]) · p_S[k]
   I(v) = H(S) − p(hit)·H(S|hit) − p(miss)·H(S|miss)

Source declaration: weighted position variance of candidates under p_S.
"""
from __future__ import annotations
import numpy as np


def _log_safe(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


def _entropy(p, eps=1e-12):
    p = np.asarray(p)
    return float(-np.sum(p * _log_safe(p, eps)))


class SourceBelief:
    def __init__(self, grid_shape, stride, source_discrimination_power=0.2,
                 min_confidence=1e-3):
        """
        Parameters
        ----------
        grid_shape : (height, width)
        stride : int
            Lattice spacing in cells between candidate sources.
        source_discrimination_power : float
            PMFS parameter (paper §VI); higher → sharper likelihood ratio.
        min_confidence : float
            Cells with α below this are ignored in the hit-map comparison.
        """
        h, w = grid_shape
        self.height = h
        self.width = w
        self.stride = int(stride)
        self.sdp = float(source_discrimination_power)
        self.min_confidence = float(min_confidence)

        # Build fixed lattice of candidate (gx, gy).
        ys = np.arange(self.stride // 2, h, self.stride, dtype=np.int32)
        xs = np.arange(self.stride // 2, w, self.stride, dtype=np.int32)
        gy, gx = np.meshgrid(ys, xs, indexing='ij')
        self.candidate_cells = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (N, 2)
        self.n_candidates = len(self.candidate_cells)

        self.p_S = np.full(self.n_candidates, 1.0 / self.n_candidates, dtype=np.float64)
        self.candidate_valid = np.ones(self.n_candidates, dtype=bool)

        # f_S cache; allocated lazily on first refresh to save memory if unused.
        self.f_S = None  # shape (N, H, W) float32

    def update_validity(self, occupancy):
        """Mark candidates in non-free cells as invalid."""
        gx = self.candidate_cells[:, 0]
        gy = self.candidate_cells[:, 1]
        self.candidate_valid = (occupancy[gy, gx] == 0)

    def refresh(self, occupancy, filament_sim, hit_prob, confidence, log=None):
        """
        Re-run filament sims for all valid candidates and update p(S|H).

        Parameters
        ----------
        occupancy : np.int8[H, W]
        filament_sim : FilamentSimulator
        hit_prob : np.float64[H, W]    measured hit probability (from HitMap)
        confidence : np.float64[H, W]  α_i per cell
        """
        if self.f_S is None:
            self.f_S = np.zeros((self.n_candidates, self.height, self.width), dtype=np.float32)

        self.update_validity(occupancy)
        valid_idx = np.where(self.candidate_valid)[0]
        if log is not None:
            log(f"[SourceBelief] refreshing {len(valid_idx)} candidates "
                f"of {self.n_candidates}")

        for k in valid_idx:
            gx, gy = self.candidate_cells[k]
            self.f_S[k] = filament_sim.simulate_source((gx, gy), occupancy)

        # Compute p(S_k | H) via hit-map comparison (log-sum for stability).
        obs_mask = confidence > self.min_confidence
        if not np.any(obs_mask):
            # No confident observations yet — keep uniform over valid candidates.
            self.p_S = np.zeros(self.n_candidates)
            self.p_S[valid_idx] = 1.0 / max(len(valid_idx), 1)
            return

        # Likelihood per cell per candidate:
        #   L_ik = α_i * (1 − |f_i^z − f_i^{S_k}|·sdp) + (1 − α_i) * 1
        # We only accumulate over cells with α > min_confidence.
        measured = hit_prob[obs_mask]                    # (M,)
        alpha = confidence[obs_mask]                     # (M,)
        one_minus_alpha = 1.0 - alpha
        predicted = self.f_S[:, obs_mask]                # (N, M)

        diff = np.abs(measured[None, :] - predicted) * self.sdp
        # Clamp to avoid negatives under pathological settings.
        prob_single = np.clip(1.0 - diff, 1e-6, 1.0)
        # L_ik, shape (N, M)
        L = alpha[None, :] * prob_single + one_minus_alpha[None, :]

        log_p = np.sum(np.log(L), axis=1)  # (N,)
        log_p[~self.candidate_valid] = -np.inf

        # Normalize (softmax-safe).
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        s = np.sum(p)
        if s <= 0.0 or not np.isfinite(s):
            # fallback: uniform over valid
            self.p_S = np.zeros(self.n_candidates)
            self.p_S[valid_idx] = 1.0 / max(len(valid_idx), 1)
        else:
            self.p_S = p / s

    # -----------------------------------------------------------------------
    # Planner interface — cheap, uses cached f_S
    # -----------------------------------------------------------------------

    def entropy(self):
        if self.p_S is None:
            return 0.0
        return _entropy(self.p_S)

    def mutual_information_at(self, cell_xy):
        """
        I(v) at a single grid cell v = (gx, gy).

        Binary observation: hit or miss, with
           p(hit | v) = Σ_k f_S[k, v] · p_S[k].
        """
        if self.f_S is None:
            return 0.0
        gx, gy = int(cell_xy[0]), int(cell_xy[1])
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return 0.0
        f_v = self.f_S[:, gy, gx].astype(np.float64)   # (N,)
        p_hit = float(np.sum(f_v * self.p_S))
        p_hit = min(max(p_hit, 1e-9), 1.0 - 1e-9)
        p_miss = 1.0 - p_hit

        # Posteriors under each outcome
        p_S_hit = f_v * self.p_S
        z_h = np.sum(p_S_hit)
        if z_h > 0:
            p_S_hit /= z_h
        p_S_miss = (1.0 - f_v) * self.p_S
        z_m = np.sum(p_S_miss)
        if z_m > 0:
            p_S_miss /= z_m

        H_S = _entropy(self.p_S)
        H_hit = _entropy(p_S_hit)
        H_miss = _entropy(p_S_miss)
        return H_S - (p_hit * H_hit + p_miss * H_miss)

    def mutual_information_batch(self, cells):
        """Vectorized MI over many cells. `cells` shape (M, 2) in (gx, gy)."""
        if self.f_S is None or len(cells) == 0:
            return np.zeros(len(cells))
        gx = np.asarray(cells[:, 0], dtype=np.int32)
        gy = np.asarray(cells[:, 1], dtype=np.int32)
        in_bounds = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        out = np.zeros(len(cells), dtype=np.float64)
        if not np.any(in_bounds):
            return out

        gx = gx[in_bounds]
        gy = gy[in_bounds]
        f = self.f_S[:, gy, gx].astype(np.float64)      # (N, M)
        p = self.p_S[:, None]                           # (N, 1)

        p_hit = np.sum(f * p, axis=0)                   # (M,)
        p_hit = np.clip(p_hit, 1e-9, 1.0 - 1e-9)
        p_miss = 1.0 - p_hit

        # Posteriors
        w_hit = f * p
        z_h = np.sum(w_hit, axis=0, keepdims=True) + 1e-12
        p_S_hit = w_hit / z_h                           # (N, M)

        w_miss = (1.0 - f) * p
        z_m = np.sum(w_miss, axis=0, keepdims=True) + 1e-12
        p_S_miss = w_miss / z_m

        H_S = _entropy(self.p_S)
        H_hit = -np.sum(p_S_hit * _log_safe(p_S_hit), axis=0)
        H_miss = -np.sum(p_S_miss * _log_safe(p_S_miss), axis=0)
        mi = H_S - (p_hit * H_hit + p_miss * H_miss)
        out[in_bounds] = mi
        return out

    # -----------------------------------------------------------------------
    # Source declaration
    # -----------------------------------------------------------------------

    def estimate_world(self, grid_to_world):
        """Weighted mean of candidate positions in world coordinates."""
        cells = self.candidate_cells
        xs = np.empty(self.n_candidates)
        ys = np.empty(self.n_candidates)
        for i, (gx, gy) in enumerate(cells):
            xs[i], ys[i] = grid_to_world(gx, gy)
        mx = float(np.sum(xs * self.p_S))
        my = float(np.sum(ys * self.p_S))
        return mx, my

    def std_world(self, grid_to_world):
        """Weighted standard deviation of candidate positions (per axis, world)."""
        mx, my = self.estimate_world(grid_to_world)
        cells = self.candidate_cells
        xs = np.empty(self.n_candidates)
        ys = np.empty(self.n_candidates)
        for i, (gx, gy) in enumerate(cells):
            xs[i], ys[i] = grid_to_world(gx, gy)
        var_x = float(np.sum((xs - mx) ** 2 * self.p_S))
        var_y = float(np.sum((ys - my) ** 2 * self.p_S))
        return np.sqrt(var_x), np.sqrt(var_y)

    def argmax_world(self, grid_to_world):
        k = int(np.argmax(self.p_S))
        gx, gy = self.candidate_cells[k]
        return grid_to_world(gx, gy)
