"""World-model vs ground truth: per-step per-channel error for each arm.

Free-run rollout: encode(s_0) once, then z_{t+1} = predict(z_t, u_t) for t=0..K-1,
decode z_t each step. Ground truth: (x, v, a)_{t+1} = W.exact_step(x, v, a, u_t)
starting from x=0. Compare decoded (gap_dec, v_dec, a_dec) vs (gap0 - x, v, a).
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, '/u/sobh/psgjepa1d')
from psgjepa1d import JEPA1D
from psgjepa1d import world as W
from psgjepa1d.data import normalisers
from eval.evaluate import fit_decoder

ap = argparse.ArgumentParser()
ap.add_argument('--arms', nargs='+', default=['baseline', 'static', 'transition', 'full'],
                help='ckpt names under --ckpt-dir (without .pt suffix)')
ap.add_argument('--ckpt-dir', default='/u/sobh/psgjepa1d/ckpt')
ap.add_argument('--jerk-mode', choices=['indep', 'trainstyle'], default='indep',
                help='indep: fresh jerk each step (default). trainstyle: 35%% resample rate (matches training).')
ap.add_argument('--report-regions', action='store_true',
                help='bucket errors by (v-tercile, gap-tercile) and print per-region k=1/5/10 gap MAE')
args = ap.parse_args()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={DEV}  arms={args.arms}", flush=True)
assert DEV == 'cuda', "this diagnostic must run on GPU (login node has no GPU)"
K = 10
N = 5000
SEED = 17

rng = np.random.default_rng(SEED)
gap0 = rng.uniform(0.3, 22, N)
v0   = rng.uniform(0.0, 14, N)
a0   = rng.uniform(-8.0, 8, N)
if args.jerk_mode == 'indep':
    u = rng.uniform(-20.0, 20, (N, K))
else:  # trainstyle: piecewise-constant, 35% resample per step (matches gen_windows)
    u = np.empty((N, K))
    u[:, 0] = rng.uniform(-20.0, 20, N)
    for k in range(1, K):
        resample = rng.random(N) < 0.35
        fresh = rng.uniform(-20.0, 20, N)
        u[:, k] = np.where(resample, fresh, u[:, k-1])
print(f"jerk_mode={args.jerk_mode}  changed_frac={float(np.mean(u[:,1:] != u[:,:-1])):.3f}", flush=True)

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
    if args.report_regions:
        # tercile bucket by initial v and gap
        v_edges = np.percentile(v0, [33.33, 66.67])
        g_edges = np.percentile(gap0, [33.33, 66.67])
        v_bin = np.digitize(v0, v_edges)          # 0=low,1=mid,2=hi
        g_bin = np.digitize(gap0, g_edges)
        print(f"\n  === {name}: gap MAE (mm) by (v-tercile, gap-tercile), k in {{1,5,10}} ===")
        print(f"  v\\gap     |  gap_lo    gap_mid   gap_hi   |  n_lo n_mid n_hi")
        for vi, vlab in enumerate(['v_lo ', 'v_mid', 'v_hi ']):
            row_k1 = []; row_k5 = []; row_k10 = []; ns = []
            for gi in range(3):
                mask = (v_bin == vi) & (g_bin == gi)
                if mask.sum() == 0:
                    row_k1.append('  n/a'); row_k5.append('  n/a'); row_k10.append('  n/a'); ns.append(0)
                    continue
                e_k1  = err[mask, 0, 0].mean() * 1000
                e_k5  = err[mask, 4, 0].mean() * 1000
                e_k10 = err[mask, 9, 0].mean() * 1000
                row_k1.append(f'{e_k1:6.1f}'); row_k5.append(f'{e_k5:6.1f}'); row_k10.append(f'{e_k10:6.1f}')
                ns.append(int(mask.sum()))
            print(f"  {vlab} k=1  | {' '.join(row_k1)} | {' '.join(f'{n:4d}' for n in ns)}")
            print(f"  {vlab} k=5  | {' '.join(row_k5)}")
            print(f"  {vlab} k=10 | {' '.join(row_k10)}")

for arm in args.arms:
    path = os.path.join(args.ckpt_dir, f'{arm}.pt')
    if not os.path.exists(path):
        print(f"skip {arm} (no ckpt)"); continue
    print(f"\nloading {path}", flush=True)
    ck = torch.load(path, map_location=DEV, weights_only=False)
    cfg = ck['cfg']
    m = JEPA1D(int(cfg['wm']['embed_dim']), int(cfg['wm']['hidden'])).to(DEV)
    m.load_state_dict(ck['model'], strict=False); m.eval()  # decoder is optional
    print(f"fitting decoder for {arm} ...", flush=True)
    dec = fit_decoder(m, D, DEV, verbose=True)
    pred = rollout(m, dec)
    report(arm, gt, pred)

print("\ndone")
