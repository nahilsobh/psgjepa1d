"""World-model vs ground truth: per-step per-channel error for each arm.

Free-run rollout: encode(s_0) once, then z_{t+1} = predict(z_t, u_t) for t=0..K-1,
decode z_t each step. Ground truth: (x, v, a)_{t+1} = W.exact_step(x, v, a, u_t)
starting from x=0. Compare decoded (gap_dec, v_dec, a_dec) vs (gap0 - x, v, a).
"""
import os, sys, numpy as np, torch
sys.path.insert(0, '/u/sobh/psgjepa1d')
from psgjepa1d import JEPA1D
from psgjepa1d import world as W
from psgjepa1d.data import normalisers
from eval.evaluate import fit_decoder

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={DEV}", flush=True)
assert DEV == 'cuda', "this diagnostic must run on GPU (login node has no GPU)"
K = 10
N = 5000
SEED = 17

rng = np.random.default_rng(SEED)
gap0 = rng.uniform(0.3, 22, N)
v0   = rng.uniform(0.0, 14, N)
a0   = rng.uniform(-8.0, 8, N)
u    = rng.uniform(-20.0, 20, (N, K))

print(f"rolling exact ground truth: N={N} K={K}", flush=True)
gt = np.empty((N, K, 3), np.float64)
for i in range(N):
    x, v, a = 0.0, float(v0[i]), float(a0[i])
    for t in range(K):
        x, v, a = W.exact_step(x, v, a, u[i, t])
        gt[i, t] = (gap0[i] - x, v, a)

cache = '/u/sobh/psgjepa1d/window_cache.npz'
d = np.load(cache); S_pool, U_pool = d['S'], d['U']
n_pool = min(500_000, len(S_pool))
S_pool = S_pool[:n_pool]; U_pool = U_pool[:n_pool]
sm, ss, um, us = normalisers(S_pool, U_pool, 'phys')
t = lambda a: torch.tensor(np.asarray(a, np.float32), device=DEV)
D = dict(states=t((S_pool - sm) / ss), actions=t((U_pool - um) / us),
         sm=sm, ss=ss, um=um, us=us)

sm_t = t(sm); ss_t = t(ss); um_t = t(um); us_t = t(us)
s0_arr = np.stack([gap0, v0, a0], axis=1).astype(np.float32)
u_t = t(u[..., None])

def rollout(model, decoder):
    with torch.no_grad():
        s0 = t(s0_arr)
        z = model.encode((s0 - sm_t) / ss_t)
        out = np.empty((N, K, 3), np.float64)
        for k in range(K):
            z = model.predict(z, (u_t[:, k] - um_t) / us_t)
            d_ = decoder(z) * ss_t + sm_t
            out[:, k] = d_.cpu().numpy()
    return out

def report(name, gt, pred):
    err = np.abs(pred - gt)
    print(f"\n=== {name}: free-run rollout error vs W.exact_step (physical units) ===")
    print(f"{'step':>4} | {'gap MAE(m)':>10} {'gap p95':>9} | {'v MAE(m/s)':>11} {'v p95':>9} | {'a MAE(m/s²)':>12} {'a p95':>9}")
    print('-' * 88)
    for k in range(K):
        e = err[:, k]
        p = np.percentile(e, 95, axis=0)
        m = e.mean(axis=0)
        print(f"{k+1:>4} | {m[0]:10.4f} {p[0]:9.4f} | {m[1]:11.4f} {p[1]:9.4f} | {m[2]:12.4f} {p[2]:9.4f}")

for arm in ['baseline', 'static', 'transition', 'full']:
    path = f'/u/sobh/psgjepa1d/ckpt/{arm}.pt'
    if not os.path.exists(path):
        print(f"skip {arm} (no ckpt)"); continue
    print(f"\nloading {path}", flush=True)
    ck = torch.load(path, map_location=DEV, weights_only=False)
    cfg = ck['cfg']
    m = JEPA1D(int(cfg['wm']['embed_dim']), int(cfg['wm']['hidden'])).to(DEV)
    m.load_state_dict(ck['model']); m.eval()
    print(f"fitting decoder for {arm} ...", flush=True)
    dec = fit_decoder(m, D, DEV, verbose=True)
    pred = rollout(m, dec)
    report(arm, gt, pred)

print("\ndone")
