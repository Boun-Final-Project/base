"""Batched on-device GPU vectorized env — components + GpuVecEnv (open-map scope).

Mirrors GasSourceEnv for the uniform-wind, free-space (no obstacle reflection) regime.
Component functions are validated exactly against the CPU env; GpuVecEnv ties plume
(gpu_filament) + lidar (gpu_lidar) + these into an on-device step(). Obstacle reflection
and per-env map generation are the remaining production pieces (see GPU_PORT_PLAN.md).
"""
import math
import numpy as np
import torch
import gpu_filament as gpuf
import gpu_lidar as gl
import gpu_wind as gw

_2PI = 2.0 * math.pi


# --------------------------------------------------------------- components
def encode_wind_batch(uv, max_speed, flip=True):
    """uv: [E,2] map-frame wind. Mirror get_local_wind: optional (u,v)->(-u,+v) flip,
    then encode (uv/ms+1)/2 clipped. (Noise omitted — deterministic path.)"""
    if flip:
        uv = torch.stack([-uv[:, 0], uv[:, 1]], dim=1)
    ms = max_speed if max_speed > 0 else 1.0
    return torch.clamp((uv / ms + 1.0) / 2.0, 0.0, 1.0)


def sensor_step_batch(threshold, measurement, weight=0.5):
    """Ratchet-up adaptive threshold + binary. Mirrors BinarySensorModel
    (update_threshold then get_binary). binary = meas > pre-update threshold;
    threshold ratchets up via EMA where meas>threshold. Returns (binary[E], new_threshold[E])."""
    binary = (measurement > threshold).to(measurement.dtype)
    up = measurement > threshold
    new_thr = torch.where(up, weight * measurement + (1 - weight) * threshold, threshold)
    return binary, new_thr


def assemble_obs_batch(gas_xyb, gas_valid, lidar, robot_pos, wind_enc, step_count,
                       map_w, map_h, max_steps):
    """gas_xyb: [E,10,3] (ax,ay,b); gas_valid: [E,10] bool; lidar: [E,R];
    robot_pos: [E,2]; wind_enc: [E,2]; step_count: [E]. Returns obs [E,107] clipped [0,1]."""
    E = robot_pos.shape[0]
    # map_w/map_h may be scalar (shared map) or [E] tensor (per-env, heterogeneous CFD maps)
    mw_g = map_w.view(-1, 1) if torch.is_tensor(map_w) else map_w
    mh_g = map_h.view(-1, 1) if torch.is_tensor(map_h) else map_h
    ax, ay, b = gas_xyb[..., 0], gas_xyb[..., 1], gas_xyb[..., 2]
    rel_x = 0.5 + (ax - robot_pos[:, 0:1]) / (2.0 * mw_g)
    rel_y = 0.5 + (ay - robot_pos[:, 1:2]) / (2.0 * mh_g)
    rel_x = torch.clamp(rel_x, 0.0, 1.0); rel_y = torch.clamp(rel_y, 0.0, 1.0)
    # sentinel slots -> (0.5, 0.5, 0)
    rel_x = torch.where(gas_valid, rel_x, torch.full_like(rel_x, 0.5))
    rel_y = torch.where(gas_valid, rel_y, torch.full_like(rel_y, 0.5))
    bb = torch.where(gas_valid, b, torch.zeros_like(b))
    gas = torch.stack([rel_x, rel_y, bb], dim=-1).reshape(E, -1)      # [E,30]
    pos = torch.stack([robot_pos[:, 0] / map_w, robot_pos[:, 1] / map_h], dim=1)
    time_frac = (step_count.to(robot_pos.dtype) / max_steps).unsqueeze(1)
    obs = torch.cat([gas, lidar, pos, wind_enc, time_frac], dim=1)     # [E,107]
    return torch.clamp(obs, 0.0, 1.0)


def reward_batch(dist, collision, binary, d_success, r_step, r_coll, r_det, r_success):
    r = torch.full_like(dist, r_step)
    r = r + collision.to(dist.dtype) * r_coll
    r = r + binary.to(dist.dtype) * r_det
    terminated = dist < d_success
    r = r + terminated.to(dist.dtype) * r_success
    return r, terminated


# --------------------------------------------------------------- GpuVecEnv (open maps)
class GpuVecEnv:
    """Minimal on-device batched env for OPEN maps (border walls only, no interior
    reflection). Demonstrates the full on-device step loop + throughput. Production
    needs obstacle reflection + map generation (see plan)."""

    def __init__(self, E, grid, source_pos, wind_vec, cfg, device, F_cap=None, dtype=torch.float32):
        self.E = E; self.dev = device; self.dtype = dtype; self.cfg = cfg
        self.map_w = grid.shape[1] * cfg.res  # grid: [H,W]; map_w = W*res
        self.map_h = grid.shape[0] * cfg.res
        self.res = cfg.res
        self.grid = torch.as_tensor(grid, device=device).unsqueeze(0).expand(E, -1, -1).contiguous()
        self.source = torch.as_tensor(source_pos, device=device, dtype=dtype).expand(E, 2).contiguous()
        self.wind_vec = torch.as_tensor(wind_vec, device=device, dtype=dtype).expand(E, 2).contiguous()
        self.per_step = cfg.filaments_per_step
        self.max_age = cfg.max_age
        self.F = F_cap or (self.max_age * self.per_step * 2)
        self.ray_angles, self.t = gl.make_lidar_consts(cfg.lidar_rays, cfg.lidar_range, self.res, device, dtype)
        self._alloc()

    def _alloc(self):
        E, F = self.E, self.F
        self.pos = torch.zeros((E, F, 2), device=self.dev, dtype=self.dtype)
        self.sig = torch.zeros((E, F), device=self.dev, dtype=self.dtype)
        self.mass = torch.full((E, F), self.cfg.mass, device=self.dev, dtype=self.dtype)
        self.age = torch.full((E, F), self.max_age + 1, device=self.dev, dtype=torch.int64)  # all dead
        self.wptr = torch.zeros(E, device=self.dev, dtype=torch.int64)
        self.robot = self.source.clone()                          # placeholder; set in reset
        self.heading = torch.zeros(E, device=self.dev, dtype=self.dtype)
        self.threshold = torch.zeros(E, device=self.dev, dtype=self.dtype)
        self.step_count = torch.zeros(E, device=self.dev, dtype=torch.int64)
        self.gas_xyb = torch.zeros((E, self.cfg.gas_hist, 3), device=self.dev, dtype=self.dtype)
        self.gas_valid = torch.zeros((E, self.cfg.gas_hist), device=self.dev, dtype=torch.bool)

    def _release(self):
        # ring-buffer write of per_step new filaments at source (F >= 2*max_age*per_step
        # guarantees the slot is long-dead before reuse)
        for k in range(self.per_step):
            idx = (self.wptr + k) % self.F                       # [E]
            ar = torch.arange(self.E, device=self.dev)
            self.pos[ar, idx] = self.source
            self.sig[ar, idx] = self.cfg.initial_sigma
            self.age[ar, idx] = 0
        self.wptr = (self.wptr + self.per_step) % self.F

    def reset(self, robot_pos):
        self.robot = torch.as_tensor(robot_pos, device=self.dev, dtype=self.dtype).expand(self.E, 2).contiguous()
        self.heading.zero_(); self.step_count.zero_()
        self.age.fill_(self.max_age + 1)
        self.gas_valid.zero_()
        # warm up the plume a bit
        for _ in range(self.cfg.warmup):
            self._plume_step()
        # init threshold from first reading
        conc = self._conc()
        self.threshold = conc.clone()
        return self._obs(torch.zeros(self.E, device=self.dev, dtype=self.dtype))

    def _plume_step(self):
        self._release()
        active = (self.age < self.max_age)
        turb = torch.randn((self.E, self.F, 2), device=self.dev, dtype=self.dtype) * self.cfg.turb_sigma
        self.pos, self.sig = gpuf.advect_diffuse_batch(
            self.pos, self.sig, active.to(self.dtype), self.wind_vec, turb, self.cfg.dt, self.cfg.K)
        self.age = self.age + active.to(torch.int64)

    def _conc(self):
        active = (self.age < self.max_age).to(self.dtype)
        return gpuf.concentration_batch(self.pos, self.sig, self.mass, active, self.robot,
                                        min_sigma=self.cfg.min_sigma)

    def _obs(self, wind_dummy):
        lidar = gl.lidar_scan_batch(self.grid, self.res, self.robot, self.heading,
                                    self.ray_angles, self.t, self.cfg.lidar_range)
        wind_enc = encode_wind_batch(self.wind_vec, self.cfg.max_speed, flip=self.cfg.flip)
        return assemble_obs_batch(self.gas_xyb, self.gas_valid, lidar, self.robot, wind_enc,
                                  self.step_count, self.map_w, self.map_h, self.cfg.max_steps)

    def step(self, actions):
        # action (cos,sin)->heading; move STEP_SIZE; clamp to border (open map)
        theta = torch.atan2(actions[:, 1], actions[:, 0])
        self.heading = theta
        nx = self.robot[:, 0] + self.cfg.step_size * torch.cos(theta)
        ny = self.robot[:, 1] + self.cfg.step_size * torch.sin(theta)
        margin = self.res
        cx = torch.clamp(nx, margin, self.map_w - margin)
        cy = torch.clamp(ny, margin, self.map_h - margin)
        collision = (cx != nx) | (cy != ny)
        self.robot = torch.stack([cx, cy], dim=1)

        self._plume_step()
        conc = self._conc()
        binary, self.threshold = sensor_step_batch(self.threshold, conc, self.cfg.thr_weight)

        # push (robot_x, robot_y, binary) into gas history ring (newest last)
        self.gas_xyb = torch.roll(self.gas_xyb, -1, dims=1)
        self.gas_valid = torch.roll(self.gas_valid, -1, dims=1)
        self.gas_xyb[:, -1, 0] = self.robot[:, 0]
        self.gas_xyb[:, -1, 1] = self.robot[:, 1]
        self.gas_xyb[:, -1, 2] = binary
        self.gas_valid[:, -1] = True

        self.step_count = self.step_count + 1
        dist = torch.linalg.norm(self.robot - self.source, dim=1)
        reward, term = reward_batch(dist, collision, binary, self.cfg.d_success,
                                    self.cfg.r_step, self.cfg.r_coll, self.cfg.r_det, self.cfg.r_success)
        trunc = self.step_count >= self.cfg.max_steps
        obs = self._obs(None)
        return obs, reward, term, trunc, {"dist": dist, "binary": binary}


# ===================================================================== full env
def is_valid_batch(grid_flat, pos, res, W, H, radius_cells):
    """Batched GasSourceEnv collision check (mirrors OccupancyGrid.is_valid):
    invalid if center cell OOB, or any in-bounds cell in the (2r+1)^2 box is occupied
    (OOB box cells are treated as free). grid_flat:[E,H*W]; pos:[E,2]. Returns valid[E]."""
    E = pos.shape[0]
    gx = torch.floor(pos[:, 0] / res).long()
    gy = torch.floor(pos[:, 1] / res).long()
    center_oob = (gx < 0) | (gx >= W) | (gy < 0) | (gy >= H)
    r = radius_cells
    offs = torch.arange(-r, r + 1, device=pos.device)
    ox, oy = torch.meshgrid(offs, offs, indexing="xy")
    ox = ox.reshape(-1); oy = oy.reshape(-1)                      # [(2r+1)^2]
    bx = gx[:, None] + ox[None, :]                                # [E,B]
    by = gy[:, None] + oy[None, :]
    inb = (bx >= 0) & (bx < W) & (by >= 0) & (by < H)
    idx = (by.clamp(0, H - 1) * W + bx.clamp(0, W - 1))
    occ = torch.gather(grid_flat, 1, idx) != 0
    box_block = (occ & inb).any(1)
    return ~(center_oob | box_block)


class GpuVecEnvMulti:
    """Complete batched env: per-env map pool, reflection-in-loop, collision, batched
    auto-reset. Plumes warmed once per map (template snapshot) and copied on reset."""

    def __init__(self, grids, sources, winds, free_cells, res, cfg, device, E,
                 dtype=torch.float32, seed=0, wind_fields=None, map_dims=None, free_dists=None):
        """grids: [K,H,W] np uint8; sources: [K,2]; winds: [K,2] (vx,vy uniform fallback);
        free_cells: list of K arrays [n_k,2] world coords of free cells (robot starts).
        wind_fields: optional [K,H,W,2] CFD spatially-varying wind per map (else uniform).
        map_dims: optional [K,2] (true map_w,map_h) per map for heterogeneous (padded) CFD maps;
                  if None all maps share the padded W*res / H*res extent.
        free_dists: optional list of K arrays [n_k] dist-to-source per free cell. When given,
                  free cells are stored sorted by distance ascending so set_start_radius() can
                  cap robot starts to a radius band (the opt-in reverse curriculum). Default
                  (no set_start_radius call) samples uniformly over ALL free cells, unchanged."""
        self.dev = device; self.dtype = dtype; self.cfg = cfg; self.E = E
        self.res = res
        self._wf_np = wind_fields
        self.spatial_wind = wind_fields is not None
        self.per_env_dims = map_dims is not None
        K, H, W = grids.shape
        self.K, self.H, self.W = K, H, W
        self.map_w = W * res; self.map_h = H * res
        self.grids = torch.as_tensor(grids, device=device).reshape(K, -1).contiguous()  # [K,H*W]
        self.sources_pool = torch.as_tensor(sources, device=device, dtype=dtype)        # [K,2]
        self.winds_pool = torch.as_tensor(winds, device=device, dtype=dtype)            # [K,2]
        if self.spatial_wind:
            self.wind_pool = torch.as_tensor(np.stack(wind_fields), device=device, dtype=dtype)  # [K,H,W,2]
            self.occ_pool = self.grids.reshape(K, H, W)                                  # [K,H,W]
        if self.per_env_dims:
            md = torch.as_tensor(map_dims, device=device, dtype=dtype)                   # [K,2]
            self.map_w_pool = md[:, 0].contiguous(); self.map_h_pool = md[:, 1].contiguous()
            self.map_w_e = torch.zeros(E, device=device, dtype=dtype)
            self.map_h_e = torch.zeros(E, device=device, dtype=dtype)
        # padded free-cell pool for VECTORIZED robot sampling (no python loop / host sync).
        # If free_dists given, sort each map's cells by distance-to-source ascending so a
        # start-radius cap = "sample from the first start_cnt cells" (reverse curriculum).
        Fmax = max(len(fc) for fc in free_cells)
        self.free_pad = torch.zeros((K, Fmax, 2), device=device, dtype=dtype)
        self.free_cnt = torch.zeros(K, device=device, dtype=torch.long)
        self.free_dist_pad = torch.full((K, Fmax), float("inf"), device=device, dtype=dtype)
        for k, fc in enumerate(free_cells):
            fc = np.asarray(fc)
            if free_dists is not None:
                order = np.argsort(np.asarray(free_dists[k]), kind="stable")
                fc = fc[order]; fd = np.asarray(free_dists[k])[order]
                self.free_dist_pad[k, :len(fc)] = torch.as_tensor(fd, device=device, dtype=dtype)
            self.free_pad[k, :len(fc)] = torch.as_tensor(fc, device=device, dtype=dtype)
            self.free_cnt[k] = len(fc)
        self.has_dists = free_dists is not None
        # start_cnt: how many (nearest) free cells each map may start in. Default = all of them
        # (== current uniform-over-free-cells behavior). set_start_radius() shrinks it.
        self.start_cnt = self.free_cnt.clone()
        self.radius_cells = int(np.ceil(cfg.robot_radius / res))
        self.per_step = cfg.filaments_per_step; self.max_age = cfg.max_age
        self.F = self.max_age * self.per_step * 2
        self.ray_angles, self.t = gl.make_lidar_consts(cfg.lidar_rays, cfg.lidar_range, res, device, dtype)
        self.tunnel = 16
        self.g = torch.Generator(device='cpu').manual_seed(seed)

        # per-env state
        # curriculum: per-map difficulty + sampling weights (set_curriculum gates by tier)
        self.map_difficulty = torch.zeros(K, device=device, dtype=torch.long)
        self.map_weight = torch.ones(K, device=device, dtype=dtype)
        self.map_idx = torch.zeros(E, device=device, dtype=torch.long)
        self.grid = torch.zeros((E, H * W), device=device, dtype=self.grids.dtype)
        self.source = torch.zeros((E, 2), device=device, dtype=dtype)
        self.wind = torch.zeros((E, 2), device=device, dtype=dtype)
        self.robot = torch.zeros((E, 2), device=device, dtype=dtype)
        self.heading = torch.zeros(E, device=device, dtype=dtype)
        self.pos = torch.zeros((E, self.F, 2), device=device, dtype=dtype)
        self.sig = torch.zeros((E, self.F), device=device, dtype=dtype)
        self.mass = torch.full((E, self.F), cfg.mass, device=device, dtype=dtype)
        self.age = torch.full((E, self.F), self.max_age + 1, device=device, dtype=torch.long)
        self.wptr = torch.zeros(E, device=device, dtype=torch.long)
        self.threshold = torch.zeros(E, device=device, dtype=dtype)
        self.step_count = torch.zeros(E, device=device, dtype=torch.long)
        self.gas_xyb = torch.zeros((E, cfg.gas_hist, 3), device=device, dtype=dtype)
        self.gas_valid = torch.zeros((E, cfg.gas_hist), device=device, dtype=torch.bool)
        # detection-reward mode: "continuous" (default, = control/s2: +r_det every in-gas
        # step), "edge" (pay on each 0->1 re-acquisition), "once" (pay only first contact
        # of the episode). edge/once need per-env history, reset in _reset_envs.
        self.r_det_mode = getattr(cfg, "r_det_mode", "continuous")
        self.prev_binary = torch.zeros(E, device=device, dtype=dtype)
        self.ever_detected = torch.zeros(E, device=device, dtype=torch.bool)

        self._build_templates(grids, sources, winds, cfg)

    def _build_templates(self, grids, sources, winds, cfg):
        """Warm a CPU FilamentPlume per map; snapshot filaments -> padded device tensors."""
        from reinforcement_learning.envs.occupancy_grid import OccupancyGrid
        from reinforcement_learning.envs.filament_plume import FilamentPlume
        K = self.K
        self.tpl_pos = torch.zeros((K, self.F, 2), device=self.dev, dtype=self.dtype)
        self.tpl_sig = torch.zeros((K, self.F), device=self.dev, dtype=self.dtype)
        self.tpl_age = torch.full((K, self.F), self.max_age + 1, device=self.dev, dtype=torch.long)
        self.tpl_n = torch.zeros(K, device=self.dev, dtype=torch.long)
        for k in range(K):
            og = OccupancyGrid(width=self.map_w, height=self.map_h, resolution=self.res)
            og.grid = grids[k].astype(np.int8)
            spd = float(np.hypot(*winds[k])); ang = float(np.arctan2(winds[k][1], winds[k][0]))
            wmod = None
            if self.spatial_wind:
                from reinforcement_learning.envs.wind_model import WindModel
                wmod = WindModel(field=self._wf_np[k], resolution=self.res,
                                 occupancy=(grids[k] != 0), max_speed=cfg.max_speed)
            pl = FilamentPlume(source_pos=tuple(sources[k]), wind_speed=spd, wind_angle=ang,
                               occupancy_grid=og, rng=np.random.default_rng(100 + k),
                               max_age=cfg.max_age, filaments_per_step=cfg.filaments_per_step,
                               mass=cfg.mass, initial_sigma=cfg.initial_sigma, wind_field=wmod)
            for _ in range(cfg.warmup):
                pl.update()
            f = pl.get_all_filaments(); n = min(f["positions"].shape[0], self.F)
            self.tpl_pos[k, :n] = torch.as_tensor(f["positions"][:n], device=self.dev, dtype=self.dtype)
            self.tpl_sig[k, :n] = torch.as_tensor(f["sigmas"][:n], device=self.dev, dtype=self.dtype)
            self.tpl_age[k, :n] = torch.as_tensor(f["ages"][:n], device=self.dev)
            self.tpl_n[k] = n

    def _reset_envs(self, mask):
        """Reset the envs where mask is True: new random map, robot, template plume."""
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        n = idx.numel()
        m = torch.multinomial(self.map_weight, n, replacement=True)   # curriculum-gated
        self.map_idx[idx] = m
        self.grid[idx] = self.grids[m]
        self.source[idx] = self.sources_pool[m]
        self.wind[idx] = self.winds_pool[m]
        # robot: random free cell per chosen map — fully vectorized (no python loop).
        # start_cnt == free_cnt by default (uniform over all free cells); under the reverse
        # curriculum it caps to the nearest cells within the current start radius.
        cnt = self.start_cnt[m].to(self.dtype)
        ridx = (torch.rand(n, device=self.dev) * cnt).long()
        ridx = torch.minimum(ridx, self.start_cnt[m] - 1)
        self.robot[idx] = self.free_pad[m, ridx]
        if self.per_env_dims:
            self.map_w_e[idx] = self.map_w_pool[m]; self.map_h_e[idx] = self.map_h_pool[m]
        self.heading[idx] = 0
        self.step_count[idx] = 0
        self.pos[idx] = self.tpl_pos[m]
        self.sig[idx] = self.tpl_sig[m]
        self.age[idx] = self.tpl_age[m]
        self.mass[idx] = self.cfg.mass
        self.wptr[idx] = self.tpl_n[m] % self.F
        self.gas_xyb[idx] = 0; self.gas_valid[idx] = False
        self.prev_binary[idx] = 0; self.ever_detected[idx] = False
        # init threshold from first reading
        c = self._conc()
        self.threshold[idx] = c[idx]

    def reset_all(self):
        self._reset_envs(torch.ones(self.E, device=self.dev, dtype=torch.bool))
        return self._obs()

    def set_difficulty(self, difficulties):
        """difficulties: list/array len K of integer tiers."""
        self.map_difficulty = torch.as_tensor(difficulties, device=self.dev, dtype=torch.long)

    def set_curriculum(self, cap):
        """Allow sampling only maps with difficulty <= cap (curriculum gate)."""
        self.map_weight = (self.map_difficulty <= cap).to(self.dtype)

    def set_start_radius(self, radius):
        """Reverse curriculum (opt-in): cap robot starts to free cells within `radius` metres
        of the source (always >=1 cell so every map stays usable). radius=inf restores the
        default uniform-over-all-free-cells sampling. Requires free_dists at construction."""
        if not self.has_dists:
            raise RuntimeError("set_start_radius needs free_dists passed to GpuVecEnvMulti")
        if radius == float("inf"):
            self.start_cnt = self.free_cnt.clone()
            return
        within = (self.free_dist_pad <= radius).sum(dim=1)            # [K] nearest cells in band
        self.start_cnt = torch.clamp(within, min=1)
        self.start_cnt = torch.minimum(self.start_cnt, self.free_cnt)

    def _conc(self):
        active = (self.age < self.max_age).to(self.dtype)
        return gpuf.concentration_batch(self.pos, self.sig, self.mass, active, self.robot,
                                        min_sigma=self.cfg.min_sigma)

    def _plume_step(self):
        # release per_step at per-env source (ring buffer)
        ar = torch.arange(self.E, device=self.dev)
        for k in range(self.per_step):
            slot = (self.wptr + k) % self.F
            self.pos[ar, slot] = self.source
            self.sig[ar, slot] = self.cfg.initial_sigma
            self.age[ar, slot] = 0
        self.wptr = (self.wptr + self.per_step) % self.F
        active = (self.age < self.max_age)
        pre = self.pos.clone()
        randn = torch.randn((self.E, self.F, 2), device=self.dev, dtype=self.dtype)
        if self.spatial_wind:
            # per-filament wind from the CFD field; per-filament turbulence scaled by LOCAL speed
            wind_pf = gw.wind_query_bilinear_pool(self.wind_pool, self.occ_pool, self.map_idx,
                                                  self.pos, self.res)               # [E,F,2]
            sp = torch.linalg.norm(wind_pf, dim=-1, keepdim=True)
            vel = wind_pf + randn * (self.cfg.turb_scale * sp)
        else:
            vel = self.wind[:, None, :] + randn * self.cfg.turb_sigma
        self.pos = self.pos + vel * self.cfg.dt * active[..., None]
        self.sig = torch.sqrt(self.sig * self.sig + 2.0 * self.cfg.K * self.cfg.dt)
        # reflection (per-env grids)
        self.pos = gpuf.reflect_batch(self.pos, pre, vel, active.to(self.dtype),
                                      self.grid.reshape(self.E, self.H, self.W), self.res,
                                      self.cfg.reflection_energy, self.cfg.dt, self.tunnel)
        self.age = self.age + active.to(torch.long)

    def _obs(self):
        lidar = gl.lidar_scan_batch(self.grid.reshape(self.E, self.H, self.W), self.res,
                                    self.robot, self.heading, self.ray_angles, self.t, self.cfg.lidar_range)
        if self.spatial_wind:
            wind_enc = gw.faithful_obs_wind_pool(self.wind_pool, self.map_idx, self.robot,
                                                 self.res, self.cfg.max_speed, flip=self.cfg.flip)
        else:
            wind_enc = encode_wind_batch(self.wind, self.cfg.max_speed, flip=self.cfg.flip)
        mw = self.map_w_e if self.per_env_dims else self.map_w
        mh = self.map_h_e if self.per_env_dims else self.map_h
        return assemble_obs_batch(self.gas_xyb, self.gas_valid, lidar, self.robot, wind_enc,
                                  self.step_count, mw, mh, self.cfg.max_steps)

    def step(self, actions):
        theta = torch.atan2(actions[:, 1], actions[:, 0]); self.heading = theta
        new = self.robot + self.cfg.step_size * torch.stack([torch.cos(theta), torch.sin(theta)], 1)
        valid = is_valid_batch(self.grid.float(), new, self.res, self.W, self.H, self.radius_cells)
        collision = ~valid
        self.robot = torch.where(valid[:, None], new, self.robot)

        self._plume_step()
        conc = self._conc()
        binary, self.threshold = sensor_step_batch(self.threshold, conc, self.cfg.thr_weight)

        self.gas_xyb = torch.roll(self.gas_xyb, -1, 1); self.gas_valid = torch.roll(self.gas_valid, -1, 1)
        self.gas_xyb[:, -1, 0] = self.robot[:, 0]; self.gas_xyb[:, -1, 1] = self.robot[:, 1]
        self.gas_xyb[:, -1, 2] = binary; self.gas_valid[:, -1] = True

        self.step_count = self.step_count + 1
        dist = torch.linalg.norm(self.robot - self.source, dim=1)
        # detection-reward trigger: shape the *reward* term only; the real `binary`
        # still feeds the obs (gas_xyb) above, unchanged.
        if self.r_det_mode == "edge":
            det = (binary > self.prev_binary).to(self.dtype)             # 0->1 re-acquisition
        elif self.r_det_mode == "once":
            det = ((binary > 0.5) & ~self.ever_detected).to(self.dtype)  # first contact only
        else:
            det = binary                                                # continuous (default)
        self.prev_binary = binary
        self.ever_detected = self.ever_detected | (binary > 0.5)
        reward, term = reward_batch(dist, collision, det, self.cfg.d_success,
                                    self.cfg.r_step, self.cfg.r_coll, self.cfg.r_det, self.cfg.r_success)
        trunc = self.step_count >= self.cfg.max_steps
        done = term | trunc
        info = {"dist": dist, "binary": binary, "success": term, "done": done}
        # auto-reset done envs, THEN return the fresh obs for them
        self._reset_envs(done)
        return self._obs(), reward, term, trunc, info
