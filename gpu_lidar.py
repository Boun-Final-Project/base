"""Batched GPU LiDAR — mirrors lidar_sim.LidarSim.scan (batch grid-sampling + 6-iter
sub-cell bisection), vectorized across E envs.

grid : [E, H, W] occupancy (0 = free), pos : [E, 2], heading : [E].
Returns normalized distances [E, R] in [0, 1] = dist / max_range.
"""
import torch


def make_lidar_consts(num_rays, max_range, res, device, dtype=torch.float32):
    ray_angles = torch.linspace(0, 2 * torch.pi, num_rays + 1, device=device, dtype=dtype)[:-1]
    n_steps = int((max_range + res - 1e-9) // res)            # ceil(max_range/res)
    t = (torch.arange(1, n_steps + 1, device=device, dtype=dtype)) * res
    return ray_angles, t


def lidar_scan_batch(grid, res, pos, heading, ray_angles, t, max_range, n_bisect=6):
    E = pos.shape[0]
    H, W = grid.shape[1], grid.shape[2]
    ang = ray_angles[None, :] + heading[:, None]              # [E,R]
    cos = torch.cos(ang); sin = torch.sin(ang)                # [E,R]

    wx = pos[:, 0, None, None] + cos[..., None] * t           # [E,R,S]
    wy = pos[:, 1, None, None] + sin[..., None] * t
    gx = torch.floor(wx / res).long()
    gy = torch.floor(wy / res).long()
    oob = (gx < 0) | (gx >= W) | (gy < 0) | (gy >= H)
    gxc = gx.clamp(0, W - 1); gyc = gy.clamp(0, H - 1)
    flat = (gyc * W + gxc).reshape(E, -1)                     # [E, R*S]
    occ = torch.gather(grid.reshape(E, -1).float(), 1, flat).reshape(wx.shape) != 0
    hit = occ | oob                                           # [E,R,S]

    any_hit = hit.any(-1)                                     # [E,R]
    first = hit.float().argmax(-1)                            # [E,R] first-True step idx
    coarse = torch.where(any_hit, t[first], torch.full_like(any_hit, max_range, dtype=t.dtype))

    # sub-cell bisection on the [coarse-res, coarse] interval (mirrors CPU 6 iters)
    lo = torch.clamp(coarse - res, min=0.0)
    hi = coarse.clone()
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        mgx = torch.floor((pos[:, 0, None] + cos * mid) / res).long()
        mgy = torch.floor((pos[:, 1, None] + sin * mid) / res).long()
        m_oob = (mgx < 0) | (mgx >= W) | (mgy < 0) | (mgy >= H)
        mflat = (mgy.clamp(0, H - 1) * W + mgx.clamp(0, W - 1))
        m_occ = m_oob | (torch.gather(grid.reshape(E, -1).float(), 1, mflat) != 0)
        hi = torch.where(m_occ, mid, hi)
        lo = torch.where(m_occ, lo, mid)
    distances = torch.where(any_hit, hi, coarse)
    return distances / max_range
