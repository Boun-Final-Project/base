# cfd_wind_pipeline

Generate a library of CFD wind fields on procedural maps using OpenFOAM, for
use as a training distribution that matches GADEN's spatial-wind characteristics.

## Why this exists

GADEN evaluation maps use spatially-varying wind fields computed by OpenFOAM
(via SimScale CFD). Our procedural training previously used uniform-per-episode
wind, which is the only metric where *every* GADEN map lies outside the
training distribution. This pipeline closes that gap by running real CFD on
procedural maps offline, then sampling `(map, wind_field)` pairs at training
time.

Reproduces GADEN's wind statistics within ~10% on the same geometry (validated
on `many_rooms`):

|                  | mean &#124;U&#124; | speed std | direction std |
|------------------|-------|-----------|---------------|
| our CFD          | 0.715 | 0.541     | 1.536 rad     |
| GADEN actual     | 0.733 | 0.583     | 1.805 rad     |

## Requirements

- SLURM cluster with **Pyxis** + **Enroot** (most HPC clusters)
- OpenFOAM container as a `.sqsh` file. Build with:
  ```bash
  enroot import 'docker://opencfd/openfoam-default:latest'
  ```
- Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib` (no other deps)
- A `reinforcement_learning` package providing `envs.map_generator.MapGenerator`
  and `test.gaden_loader` (for procedural maps + GADEN refs)

Edit `config.py` (or set env vars `CFD_OPENFOAM_SQSH`, `CFD_RL_PACKAGE_PATH`,
`CFD_GADEN_MAPS_ROOT`, `CFD_DATA_ROOT`) to match your environment.

## Quickstart

```bash
cd cfd_wind_pipeline
DATA=${CFD_DATA_ROOT:-/tmp/cfd_data}; mkdir -p $DATA

# 0. Smoke test (verify OpenFOAM + Pyxis + cluster work end-to-end)
#    Requires a copy of the windAroundBuildings tutorial at $DATA/test_unmodified
sbatch sbatch/run_smoke.sh $DATA/test_unmodified

# 1. Generate one case (procedural T5 multi_room)
python gen_case.py --out-dir $DATA/case_t5_demo \
    --template-id 5 --seed 7 --inlet-speed 0.5 \
    --openings-west "1.4-2.9,6.1-7.6" --openings-east "1.4-2.9,6.1-7.6"

# 2. Run CFD (~1-5 min once it starts)
sbatch sbatch/run_case.sh $DATA/case_t5_demo

# 3. Extract wind field as numpy
python extract_wind.py --case-dir $DATA/case_t5_demo --write-dict-only
sbatch sbatch/run_case.sh $DATA/case_t5_demo  # if you also need postProcess re-run
python extract_wind.py --case-dir $DATA/case_t5_demo --parse-only

# 4. Visualize
python viz/viz_wind.py --case-dir $DATA/case_t5_demo
```

## File map

| File | Purpose |
|---|---|
| `config.py` | Centralized paths + defaults (env-var overridable) |
| `gen_case.py` | Generate OpenFOAM case from a procedural map |
| `gen_case_from_gaden.py` | Same but for a GADEN map (validation only) |
| `placement.py` | Sample inlet/outlet placements with constraints |
| `extract_wind.py` | Postprocess OpenFOAM output → (H,W,2) numpy |
| `parse_doors.py` | Utility to read GADEN's `doors.stl` and find inlet/outlet positions |
| `viz/viz_wind.py` | Quiver plot of one case's wind field |
| `viz/viz_placements.py` | Contact-sheet of map + opening placements (pre-CFD review) |
| `viz/viz_gaden_wind.py` | Visualize a GADEN reference wind field (for comparison) |
| `sbatch/run_case.sh` | SLURM wrapper: blockMesh + snappyHexMesh + simpleFoam |
| `sbatch/run_smoke.sh` | Smoke test using OpenFOAM's stock tutorial |

## Pipeline

```
MapGenerator → 2D grid → STL (extruded walls, with openings + boundary
                              blockers) →
  blockMeshDict + snappyHexMeshDict + k-epsilon BCs →
  blockMesh → snappyHexMesh → simpleFoam (steady, k-epsilon) →
  postProcess (cutting-plane sample at z=H/2) →
  Python: interpolate to (H,W,2) → wind_field.npz
```

## Notes on the opening placement strategy

GADEN scenarios have **4 doors** total: 2 inlets + 2 outlets, placed at
specific positions on opposite walls. We mirror this with a heuristic sampler
(`placement.py`) that:

- Picks 1–3 inlets on one wall (default W or S), 1–3 outlets on opposite wall
- Requires opening width ∈ [1.0, 2.5] m
- Requires openings ≥ 1 m apart on the same wall
- Requires each opening's clearance zone (immediate cells inside) to be 100%
  free (no interior walls right behind the opening) and far zone ≥ 85% free
- Requires the opening to connect to the map's main interior via flood-fill
- Requires inlet-to-outlet distance ≥ 0.5 × map diagonal

Use `viz/viz_placements.py` to render a contact sheet of N sampled placements
for review *before* spending CFD compute.

## What stays outside the repo (in `$CFD_DATA_ROOT`)

- Generated case directories (`case_*/`) — these contain GBs of mesh data
- `*.sqsh` container files (huge)
- `wind_field.npz` library outputs
- `.png` visualizations

The repo only contains source.

## Validation

On GADEN `many_rooms` geometry with the same 4-door layout GADEN uses:
- See `case_gaden_4doors/` (if you generated one) and compare to
  `gaden_many_rooms_wind.png`
- Stats match within 5-15% on mean, speed std, direction std

On procedural T5 multi_room with random 2-inlet/2-outlet placement:
- Direction circular_std ≈ 1.1–1.5 rad (vs GADEN range 1.02-2.00)
- Speed std ≈ 0.1–0.6 m/s (vs GADEN range 0.29-0.65)
- Channels through interior gaps + recirculation zones in side rooms appear
  naturally (the "trap room" failure-mode geometry)
