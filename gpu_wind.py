"""Batched GPU spatially-varying wind query — mirrors WindModel.query (wall-aware
bilinear, training/advection path) and the GADEN-faithful nearest-cell obs path.

field : [E,H,W,2] per-cell wind ; occ : [E,H,W] (1=wall) ; pos : [E,F,2] world.
"""
import torch


def wind_query_bilinear(field, occ, pos, res):
    """Wall-aware bilinear interpolation (WindModel.query default path), batched.
    Returns wind [E,F,2]. Cells that are walls/OOB are dropped from the blend;
    if all 4 corners invalid -> zero wind (matches CPU w_sum==0 branch)."""
    E, Hh, Ww, _ = field.shape
    F = pos.shape[1]
    field_flat = field.reshape(E, Hh * Ww, 2)
    occ_flat = occ.reshape(E, Hh * Ww).to(field.dtype)

    cf = pos[..., 0] / res - 0.5
    rf = pos[..., 1] / res - 0.5
    c0 = torch.floor(cf).long(); r0 = torch.floor(rf).long()
    c1 = c0 + 1; r1 = r0 + 1
    tx = cf - c0; ty = rf - r0

    vc0 = (c0 >= 0) & (c0 <= Ww - 1); vc1 = (c1 >= 0) & (c1 <= Ww - 1)
    vr0 = (r0 >= 0) & (r0 <= Hh - 1); vr1 = (r1 >= 0) & (r1 <= Hh - 1)
    c0c = c0.clamp(0, Ww - 1); c1c = c1.clamp(0, Ww - 1)
    r0c = r0.clamp(0, Hh - 1); r1c = r1.clamp(0, Hh - 1)

    def gocc(rc, cc):
        idx = (rc * Ww + cc)
        return torch.gather(occ_flat, 1, idx.reshape(E, -1)).reshape(rc.shape) != 0

    wall00 = gocc(r0c, c0c) & vr0 & vc0
    wall01 = gocc(r0c, c1c) & vr0 & vc1
    wall10 = gocc(r1c, c0c) & vr1 & vc0
    wall11 = gocc(r1c, c1c) & vr1 & vc1
    v00 = vr0 & vc0 & ~wall00; v01 = vr0 & vc1 & ~wall01
    v10 = vr1 & vc0 & ~wall10; v11 = vr1 & vc1 & ~wall11

    z = torch.zeros_like(tx)
    w00 = torch.where(v00, (1 - tx) * (1 - ty), z)
    w01 = torch.where(v01, tx * (1 - ty), z)
    w10 = torch.where(v10, (1 - tx) * ty, z)
    w11 = torch.where(v11, tx * ty, z)
    wsum = w00 + w01 + w10 + w11
    inv = 1.0 / torch.where(wsum > 0, wsum, torch.ones_like(wsum))
    w00 *= inv; w01 *= inv; w10 *= inv; w11 *= inv

    def gfield(rc, cc):
        idx = (rc * Ww + cc).unsqueeze(-1).expand(-1, -1, 2)
        return torch.gather(field_flat, 1, idx)
    f00 = gfield(r0c, c0c); f01 = gfield(r0c, c1c)
    f10 = gfield(r1c, c0c); f11 = gfield(r1c, c1c)
    return (w00[..., None] * f00 + w01[..., None] * f01
            + w10[..., None] * f10 + w11[..., None] * f11)


def wind_query_bilinear_pool(field_pool, occ_pool, map_idx, pos, res):
    """Like wind_query_bilinear but field is a shared POOL [K,H,W,2] indexed per-env by
    map_idx [E] (memory-light: avoids materializing [E,H,W,2]). pos [E,F,2] -> [E,F,2]."""
    K, Hh, Ww, _ = field_pool.shape
    pool = field_pool.reshape(K * Hh * Ww, 2)
    occ = occ_pool.reshape(K * Hh * Ww).to(field_pool.dtype)
    base = (map_idx * (Hh * Ww)).view(-1, 1)            # [E,1] global offset per env

    cf = pos[..., 0] / res - 0.5; rf = pos[..., 1] / res - 0.5
    c0 = torch.floor(cf).long(); r0 = torch.floor(rf).long(); c1 = c0 + 1; r1 = r0 + 1
    tx = cf - c0; ty = rf - r0
    vc0 = (c0 >= 0) & (c0 <= Ww - 1); vc1 = (c1 >= 0) & (c1 <= Ww - 1)
    vr0 = (r0 >= 0) & (r0 <= Hh - 1); vr1 = (r1 >= 0) & (r1 <= Hh - 1)
    c0c = c0.clamp(0, Ww - 1); c1c = c1.clamp(0, Ww - 1)
    r0c = r0.clamp(0, Hh - 1); r1c = r1.clamp(0, Hh - 1)

    def gid(rc, cc): return base + rc * Ww + cc          # [E,F] global flat index
    def gocc(rc, cc): return occ[gid(rc, cc)] != 0
    def gfield(rc, cc): return pool[gid(rc, cc)]         # [E,F,2]

    wall00 = gocc(r0c, c0c) & vr0 & vc0; wall01 = gocc(r0c, c1c) & vr0 & vc1
    wall10 = gocc(r1c, c0c) & vr1 & vc0; wall11 = gocc(r1c, c1c) & vr1 & vc1
    v00 = vr0 & vc0 & ~wall00; v01 = vr0 & vc1 & ~wall01
    v10 = vr1 & vc0 & ~wall10; v11 = vr1 & vc1 & ~wall11
    z = torch.zeros_like(tx)
    w00 = torch.where(v00, (1 - tx) * (1 - ty), z); w01 = torch.where(v01, tx * (1 - ty), z)
    w10 = torch.where(v10, (1 - tx) * ty, z); w11 = torch.where(v11, tx * ty, z)
    wsum = w00 + w01 + w10 + w11
    inv = 1.0 / torch.where(wsum > 0, wsum, torch.ones_like(wsum))
    w00 *= inv; w01 *= inv; w10 *= inv; w11 *= inv
    return (w00[..., None] * gfield(r0c, c0c) + w01[..., None] * gfield(r0c, c1c)
            + w10[..., None] * gfield(r1c, c0c) + w11[..., None] * gfield(r1c, c1c))


def faithful_obs_wind_pool(field_pool, map_idx, robot, res, max_speed, flip=True):
    """GADEN-faithful obs wind (nearest-cell + flip + encode), pool-indexed. robot [E,2] -> [E,2].
    (Noise omitted — deterministic; add the speed^2-space anemometer noise at train time if desired.)"""
    K, Hh, Ww, _ = field_pool.shape
    pool = field_pool.reshape(K * Hh * Ww, 2)
    col = torch.clamp(torch.floor(robot[:, 0] / res).long(), 0, Ww - 1)
    row = torch.clamp(torch.floor(robot[:, 1] / res).long(), 0, Hh - 1)
    uv = pool[map_idx * (Hh * Ww) + row * Ww + col]      # [E,2]
    if flip:
        uv = torch.stack([-uv[:, 0], uv[:, 1]], 1)
    ms = max_speed if max_speed > 0 else 1.0
    return torch.clamp((uv / ms + 1.0) / 2.0, 0.0, 1.0)
