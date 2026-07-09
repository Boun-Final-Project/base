# Benchmark map dimensions

Top-down floorplans of every scenario in `benchmark_env/scenarios`, for choosing
gas-source and robot-start placements. Dimensions are computed from the `walls.stl`
CAD bounding box (STL units = meters). Cell size is `0.1 m`.

- **Bounding box** includes the outer boundary-wall thickness.
- **Usable interior** is the free floor area (bbox minus the perimeter wall).
- **Source** = current `sim1/sim.yaml` `source.position` (green ★ in the drawings).
  Robot start is **not** baked into the scenario — it is set at launch time.

All `_1.._4` (and `_a.._d`, `_1.._5`) variants of a family share the **same outer
size and gas source**; they differ only in internal wall/door geometry, so check the
per-variant PNG for the interior layout.

| Family | Bounding box X | Bounding box Y | Interior W×H (m) | Wall thick | Source (x,y,z) |
|---|---|---|---|---|---|
| `10x6_u_left_*`         | −0.2 … 10.2 | −0.2 … 6.2  | 10 × 6   | 0.2 | (1.45, 3.0, 0.5) |
| `10x6_u_right_*`        | −0.2 … 10.2 | −0.2 … 6.2  | 10 × 6   | 0.2 | (1.45, 3.0, 0.5) |
| `curved_labrinth_left_*`  | −0.2 … 10.2 | −0.2 … 6.2  | 10 × 6   | 0.2 | (0.5, 3.0, 0.5) |
| `curved_labrinth_right_*` | −0.2 … 10.2 | −0.2 … 6.2  | 10 × 6   | 0.2 | (0.5, 3.0, 0.5) |
| `4_rooms_start_*`       | −0.4 … 13.4 | −0.4 … 13.4 | 13 × 13  | 0.4 | (2.5, 3.0, 0.5) |
| `many_rooms_*`          | −0.4 … 17.4 | −0.4 … 10.4 | 17 × 10  | 0.4 | (2.8, 4.2, 0.5) |
| `ultimate_*`            | −0.4 … 20.4 | −0.4 … 12.4 | 20 × 12  | 0.4 | (1.45, 3.0, 0.5) |

Height (Z) of every map is **3.0 m**.

## Placement guidance
- Keep source/robot within the interior box, clear of walls by at least the robot
  clamp radius. Free XY ranges (interior):
  - 10×6 maps: `x ∈ [0, 10]`, `y ∈ [0, 6]`
  - 4_rooms:   `x ∈ [0, 13]`, `y ∈ [0, 13]`
  - many_rooms:`x ∈ [0, 17]`, `y ∈ [0, 10]`
  - ultimate:  `x ∈ [0, 20]`, `y ∈ [0, 12]`
- Source z is typically `0.5 m` (mid-height release).

## Files
- `<scenario>.png` — one floorplan per scenario (walls dark, doors red, source ★).
- `_family_<name>.png` — all variants of a family side by side.
