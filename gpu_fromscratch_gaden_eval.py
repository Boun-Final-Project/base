"""Eval a GPU-from-scratch checkpoint on the real GADEN maps (flip-ON real gas)."""
import os, sys
for k in ("OSL_LOCAL_WIND_OBS","OSL_DEPLOY_ANEMO","GADEN_ANEMO_FRAME","GADEN_FAITHFUL_WIND"):
    os.environ.setdefault(k, "1")
from pathlib import Path
import numpy as np, torch
_PKG = Path("champ_far02_python_eval").resolve(); sys.path.insert(0, str(_PKG))
from cf_eval.envs.gas_source_env import GasSourceEnv
from cf_eval.models.actor_critic import ActorCriticDualBackbone
from cf_eval.test.gaden_loader import load_full_map, GadenGasField
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("ckpt", nargs="?", default="reinforcement_learning/runs/gpu_fromscratch_flipON/checkpoints/agent_1966080.pt")
_ap.add_argument("--maps", default="4rooms,uleft,uright,labyrinth_left,labyrinth_right,many_rooms,ultimate")
_ap.add_argument("--eps", type=int, default=8)
_a = _ap.parse_args()
DEV = torch.device("cpu")
CK = _a.ckpt
MAPS = _a.maps.split(",")
N_EPS = _a.eps

a = ActorCriticDualBackbone()
ck = torch.load(CK, map_location=DEV, weights_only=False)
a.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck, strict=False)
a.eval()
def greedy(o):
    with torch.no_grad(): return a._actor_dist(a._encode_shared(o)).mean.cpu().numpy()[0]
def fresh(fm):
    gf=fm.get("gas_field")
    if gf is None: return None
    g=GadenGasField(gf._dir,gf._start_iteration,gf._max_iteration,tuple(gf._origin),z_query=gf._z,save_dt=gf.save_dt)
    g.set_occupancy(fm["grid"].grid,fm["grid"].resolution); g.set_iters_per_step(5); return g
print(f"GADEN eval (flip-ON, real gas) of from-scratch ckpt: {CK}\n")
print(f"  {'map':16s} {'success':>8} {'min_dist':>9}")
tot=[]
for mp in MAPS:
    try:
        fm=load_full_map(Path("gaden_scenarios"),Path("base/gaden_maps/recommended_configs.yaml"),mp,replay_gas=True)
    except Exception as e:
        print(f"  {mp:16s}  skip ({e})"); continue
    src=np.array(fm["source_pos"],float); md={k:fm[k] for k in ("grid","source_pos","robot_pos","width","height")}; md["start_time"]=fm.get("start_time",0)
    succ=[]; mind=[]
    for ep in range(N_EPS):
        env=GasSourceEnv(); obs,_=env.reset(seed=7000+ep,options={"map_data":md,"wind_field":fm["wind_field"],"gas_field":fresh(fm),"deploy_motion":True,"sensor_noise":False})
        s=0.0; m=99.0
        while True:
            o=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
            obs,r,term,trunc,info=env.step(greedy(o)); m=min(m,float(np.linalg.norm(env._robot_pos-src)))
            if term: s=1.0
            if term or trunc: break
        succ.append(s); mind.append(m)
    sr=100*np.mean(succ); tot.append(sr)
    print(f"  {mp:16s} {sr:7.0f}% {np.mean(mind):8.2f}m")
print(f"\n  OVERALL success = {np.mean(tot):.0f}%  (from-scratch flip-ON, simple-map training)")
