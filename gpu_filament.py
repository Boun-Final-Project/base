"""Batched GPU filament-plume kernels (PyTorch) — proof of concept.

The CPU env runs 96 independent FilamentPlume processes; each step does a NumPy
Gaussian-sum concentration query + advection over ~240 filaments. The GPU win is
to batch ALL envs into single tensor ops: state shaped [E, F, ...] on device, so
one kernel computes all E envs at once.

This module mirrors the CPU physics in filament_plume.py:
  concentration:  C = Σ_i  m_i / (2π σ_i²) · exp(-r_i² / (2σ_i²)),  σ² clamped to min_σ²
  advect+diffuse: x += (wind + turb)·dt ;  σ = sqrt(σ² + 2K·dt)

SCOPE (honest): this covers the free-space plume core — the dominant per-step cost.
NOT yet ported: wall occlusion (line-of-sight) and obstacle reflection. Occlusion is
sketched (batched line-sampling, same pattern as the lidar grid-sampler) and validated
separately as APPROXIMATE (sampled DDA vs the CPU's exact Bresenham).
"""
import math
import torch

_2PI = 2.0 * math.pi


def concentration_batch(positions, sigmas, masses, active_mask, query, min_sigma=0.01):
    """Free-space concentration for E envs at once. Matches CPU concentration_at
    with FILAMENT_WALL_OCCLUSION=False, exactly (same float math).

    positions   : [E, F, 2]   filament xy
    sigmas      : [E, F]
    masses      : [E, F]
    active_mask : [E, F]       1.0 for live filaments, 0.0 for padding
    query       : [E, 2]       per-env query point
    returns     : [E]          concentration
    """
    d = positions - query[:, None, :]                      # [E,F,2]
    r2 = (d * d).sum(-1)                                   # [E,F]
    sigma2 = torch.clamp(sigmas * sigmas, min=min_sigma * min_sigma)
    contrib = masses / (_2PI * sigma2) * torch.exp(-r2 / (2.0 * sigma2))
    return (contrib * active_mask).sum(-1)                 # [E]


def advect_diffuse_batch(positions, sigmas, active_mask, wind, turbulence, dt, K):
    """One advection+diffusion step for all envs (uniform-wind path).

    positions   : [E, F, 2]  (modified in place semantics — returns new tensors)
    wind        : [E, 2]     per-env uniform wind vector
    turbulence  : [E, F, 2]  pre-sampled N(0,1)*turb_sigma (shared with CPU for test)
    Returns (positions', sigmas').
    """
    disp = (wind[:, None, :] + turbulence) * dt            # [E,F,2]
    positions = positions + disp * active_mask[..., None]
    sigmas = torch.sqrt(sigmas * sigmas + 2.0 * K * dt)
    return positions, sigmas


def _gather_occ(grid_flat, gx, gy, W, H):
    """grid_flat [E, H*W]; gx,gy [E, ...] -> occupancy bool, OOB counts as occupied."""
    oob = (gx < 0) | (gx >= W) | (gy < 0) | (gy >= H)
    idx = (gy.clamp(0, H - 1) * W + gx.clamp(0, W - 1))
    E = grid_flat.shape[0]
    occ = torch.gather(grid_flat, 1, idx.reshape(E, -1)).reshape(idx.shape) != 0
    return occ | oob


def reflect_batch(positions, pre_positions, velocities, active, grid, res,
                  reflection_energy, dt, tunnel_steps):
    """Batched port of FilamentPlume._handle_obstacles (no RNG escape: fully-enclosed
    filaments snap back to pre-position instead of a random kick — rare, deterministic).
    positions/pre_positions/velocities: [E,F,2]; grid: [E,H,W]; returns corrected positions."""
    E, F, _ = positions.shape
    H, W = grid.shape[1], grid.shape[2]
    gf = grid.reshape(E, -1).float()

    post_g = torch.floor(positions / res).long()                 # [E,F,2]
    pre_g = torch.floor(pre_positions / res).long()
    cell_changed = (pre_g[..., 0] != post_g[..., 0]) | (pre_g[..., 1] != post_g[..., 1])
    in_wall_post = _gather_occ(gf, post_g[..., 0], post_g[..., 1], W, H)
    cand = active.bool() & (cell_changed | in_wall_post)         # [E,F]

    # sample path pre->post at n_s points (t in (0,1])
    n_s = tunnel_steps
    t = torch.linspace(0.0, 1.0, n_s + 1, device=positions.device, dtype=positions.dtype)[1:]
    d = positions - pre_positions                                # [E,F,2]
    sx = pre_positions[..., 0:1] + t * d[..., 0:1]               # [E,F,n_s]
    sy = pre_positions[..., 1:2] + t * d[..., 1:2]
    sgx = torch.floor(sx / res).long(); sgy = torch.floor(sy / res).long()
    s_wall = _gather_occ(gf, sgx, sgy, W, H)                     # [E,F,n_s]
    any_blocked = s_wall.any(-1) & cand                         # [E,F]
    first_hit = s_wall.float().argmax(-1)                       # [E,F]

    # wall cell at first hit
    fh = first_hit.unsqueeze(-1)
    wall_gx = torch.gather(sgx, 2, fh).squeeze(-1)              # [E,F]
    wall_gy = torch.gather(sgy, 2, fh).squeeze(-1)

    # normal = mean direction toward free cardinal neighbors
    offs = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=positions.device)
    dirs = offs.to(positions.dtype)                             # [4,2]
    ngx = wall_gx.unsqueeze(-1) + offs[:, 0]                    # [E,F,4]
    ngy = wall_gy.unsqueeze(-1) + offs[:, 1]
    nbr_occ = _gather_occ(gf, ngx, ngy, W, H)                   # [E,F,4] (OOB=occupied)
    is_free = (~nbr_occ).to(positions.dtype)                   # [E,F,4]
    normal = (is_free.unsqueeze(-1) * dirs).sum(2)             # [E,F,2]
    nn = torch.linalg.norm(normal, dim=-1, keepdim=True)
    has_n = (nn.squeeze(-1) > 0)
    normal = torch.where(nn > 0, normal / nn.clamp_min(1e-9), normal)

    # reflect: v_ref = v - 2(v·n)n ; ref_pos = pre + v_ref*energy*dt
    vdotn = (velocities * normal).sum(-1, keepdim=True)
    v_ref = velocities - 2.0 * vdotn * normal
    ref_pos = pre_positions + v_ref * reflection_energy * dt    # [E,F,2]

    # validate reflected pos in free + reflected path clear
    rg = torch.floor(ref_pos / res).long()
    ref_in_wall = _gather_occ(gf, rg[..., 0], rg[..., 1], W, H)
    rsx = pre_positions[..., 0:1] + t * (ref_pos[..., 0:1] - pre_positions[..., 0:1])
    rsy = pre_positions[..., 1:2] + t * (ref_pos[..., 1:2] - pre_positions[..., 1:2])
    ref_path_blocked = _gather_occ(gf, torch.floor(rsx / res).long(),
                                   torch.floor(rsy / res).long(), W, H).any(-1)
    use_ref = any_blocked & has_n & (~ref_in_wall) & (~ref_path_blocked)
    snap_pre = any_blocked & ~use_ref                           # blocked but can't reflect -> pre

    out = positions.clone()
    out = torch.where(use_ref.unsqueeze(-1), ref_pos, out)
    out = torch.where(snap_pre.unsqueeze(-1), pre_positions, out)
    return out


def concentration_batch_occluded(positions, sigmas, masses, active_mask, query,
                                 grid, res, n_steps, min_sigma=0.01):
    """APPROXIMATE occlusion via batched line-sampling (sampled DDA, not Bresenham).

    grid : [E, H, W] uint8 occupancy (0=free). Samples each filament->query segment
    at n_steps points; a filament is occluded if any sample lands in a wall cell.
    This is the SAME grid-gather pattern the lidar uses; expect small boundary
    differences vs the CPU's exact Bresenham (validated as 'close', not exact).
    """
    E, F, _ = positions.shape
    d = positions - query[:, None, :]
    r2 = (d * d).sum(-1)
    sigma2 = torch.clamp(sigmas * sigmas, min=min_sigma * min_sigma)
    # 3-sigma prefilter (matches CPU)
    near = r2 < 9.0 * sigma2
    # sample the segment query->filament at t in (0,1]
    t = torch.linspace(0.0, 1.0, n_steps + 1, device=positions.device)[1:]   # [S]
    sx = query[:, None, None, 0] + t * d[..., 0:1]          # [E,F,S]
    sy = query[:, None, None, 1] + t * d[..., 1:2]
    gx = torch.clamp((sx / res).long(), 0, grid.shape[2] - 1)
    gy = torch.clamp((sy / res).long(), 0, grid.shape[1] - 1)
    # gather occupancy per env
    flat = (gy * grid.shape[2] + gx).reshape(E, -1)         # [E, F*S]
    occ = torch.gather(grid.reshape(E, -1).float(), 1, flat).reshape(E, F, n_steps)
    blocked = (occ > 0).any(-1)                             # [E,F]
    visible = active_mask.bool() & near & (~blocked)
    contrib = masses / (_2PI * sigma2) * torch.exp(-r2 / (2.0 * sigma2))
    return (contrib * visible.float()).sum(-1)
