"""Trajectory-window synthesis for PSG-JEPA-1D.

Upstream loads (video, proprioceptive state) windows of T frames. Here we SYNTHESISE windows of
T states from the exact 1-D dynamics, with a piecewise-constant jerk sequence per window, so the
data has the same shape: states (T,3) + actions (T-1,1). Futures are exact (machine zero).
"""
import numpy as np
from . import world as W


def gen_windows(n=200_000, T=7, seed=0, verbose=True):
    """Returns states (n,T,3) = [gap,v,a], actions (n,T-1,1) = jerk."""
    rng = np.random.default_rng(seed)
    S = np.empty((n, T, 3)); U = np.empty((n, T-1, 1))
    def sample():
        r = rng.random()
        if   r < .32: return rng.uniform(0.3,22), rng.uniform(0,14),  rng.uniform(-8,8)
        elif r < .42: return rng.uniform(1,7),    rng.uniform(11,14), rng.uniform(-8,8)
        elif r < .50: return rng.uniform(0.15,3), rng.uniform(0,4),   rng.uniform(-6,6)
        elif r < .59: return rng.uniform(0.5,16), rng.uniform(2,14),  rng.uniform(-8,-2)
        elif r < .65: return rng.uniform(2,20),   rng.uniform(0,12),  rng.uniform(2,8)
        elif r < .78:
            v = rng.uniform(3,14); d = W.mbd_va(v,0.)
            return d*rng.uniform(0.75,1.35), v, rng.uniform(-8,2)
        else:
            v = rng.uniform(13.,14.); d = W.mbd_va(v,0.)
            return d*rng.uniform(0.90,1.20), v, rng.uniform(-8,4)
    for i in range(n):
        g, v, a = sample()
        x = 0.0; S[i,0] = (g, v, a)
        for t in range(T-1):
            # piecewise-constant jerk: mostly held, sometimes resampled (covers ramps AND holds)
            j = rng.uniform(-20,20) if (t == 0 or rng.random() < 0.35) else U[i,t-1,0]
            U[i,t,0] = j
            x, v, a = W.exact_step(x, v, a, j)
            S[i,t+1] = (g-x, v, a)
        if verbose and (i+1) % 200_000 == 0: print(f"    {i+1}/{n}", flush=True)
    return S, U


def normalisers(S, U, mode='phys'):
    if mode == 'phys':
        return np.zeros(3), W.SCALE_PHYS.copy(), np.zeros(1), np.array([W.SCALE_J])
    s = S.reshape(-1,3); u = U.reshape(-1,1)
    return s.mean(0), s.std(0)+1e-6, u.mean(0), u.std(0)+1e-6
