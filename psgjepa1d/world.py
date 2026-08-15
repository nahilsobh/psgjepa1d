"""
JEPA-1D core: exact world, data synthesis, non-dimensional scales.
Pure NumPy. Imported by the GPU training script.
"""
import numpy as np, os, json
DT, A_MAX, J_MAX, VMAX, K_DEFAULT = 0.1, 8.0, 20.0, 14.0, 6
# --- physics-derived non-dimensional scales (see README section 2) ---
T0 = A_MAX/J_MAX                 # 0.40 s
L0 = A_MAX**3/J_MAX**2           # 1.28 m
V0 = A_MAX**2/J_MAX              # 3.20 m/s
PI = VMAX*J_MAX/A_MAX**2         # 4.375  (the ONLY dimensionless parameter)
SCALE_PHYS = np.array([L0, V0, A_MAX]);  SCALE_J = J_MAX

def exact_step(x, v, a, j, dt=DT, A=A_MAX):
    """EXACT event-driven integration of constant-jerk dynamics. Verified to 7e-7 vs fine RK."""
    t = 0.0; guard = 0
    while t < dt - 1e-12 and guard < 60:
        guard += 1; rem = dt - t
        if v <= 1e-12:
            if a <= 1e-12 and j <= 1e-12:
                a = min(max(a + j*rem, -A), A); t = dt; break
            if a < -1e-12 and j > 1e-12:
                d = min(rem, (0.0-a)/j); a = min(max(a + j*d, -A), A); v = 0.0; t += d; continue
        je = j
        if (a >= A-1e-12 and j > 0) or (a <= -A+1e-12 and j < 0): je = 0.0
        d = rem
        if je > 1e-12 and a < A:  d = min(d, (A-a)/je)
        if je < -1e-12 and a > -A: d = min(d, (-A-a)/je)
        if v > 1e-12:
            if abs(je) > 1e-12:
                disc = a*a - 2*je*v
                if disc >= 0:
                    sq = disc**0.5
                    for r in ((-a+sq)/je, (-a-sq)/je):
                        if 1e-12 < r < d: d = r
            elif a < -1e-12:
                r = -v/a
                if 1e-12 < r < d: d = r
        if d <= 1e-12: d = rem
        x = x + v*d + 0.5*a*d*d + je*d**3/6.0
        v = v + a*d + 0.5*je*d*d
        a = min(max(a + je*d, -A), A)
        if v < 1e-9: v = 0.0
        t += d
    return x, v, a

# ---------- corrected analytical solution (lands v=0 AND a=0 exactly) ----------
def plan_terminal(v, a):
    for n in range(1, 12):
        b = -2*v/(DT*(n+1))
        if b < -A_MAX-1e-9 or b > 1e-9: continue
        jt = (b-a)/DT
        if abs(jt) > J_MAX+1e-9: continue
        jterm = -b/(n*DT)
        if abs(jterm) > J_MAX+1e-9: continue
        return jt, n, jterm
    return None
def brake_from(x, v, a):
    for _ in range(600):
        p = plan_terminal(v, a)
        if p is not None:
            jt, n, jt2 = p
            x, v, a = exact_step(x, v, a, jt)
            for _ in range(n): x, v, a = exact_step(x, v, a, jt2)
            return x, True
        x, v, a = exact_step(x, v, a, -J_MAX if a > -A_MAX else 0.0)
        if v <= 1e-12: return x, True
    return x, False
def min_brake_dist(v0, a0=0.0): return brake_from(0.0, float(v0), a0)[0]
def sc_jerk(v, a):
    p = plan_terminal(v, a)
    return p[0] if p is not None else (-J_MAX if a > -A_MAX else 0.0)

_G = None
def mbd_va(v, a):
    """exact discrete braking distance from (v,a), bilinear on a cached grid."""
    global _G
    if _G is None:
        vg = np.linspace(0., 14., 71); ag = np.linspace(-8., 8., 33)
        _G = (vg, ag, np.array([[min_brake_dist(x, y) for y in ag] for x in vg]))
    vg, ag, M = _G
    v = min(max(v, vg[0]), vg[-1]); a = min(max(a, ag[0]), ag[-1])
    i = int(np.clip(np.searchsorted(vg, v)-1, 0, len(vg)-2)); k = int(np.clip(np.searchsorted(ag, a)-1, 0, len(ag)-2))
    tv = (v-vg[i])/(vg[i+1]-vg[i]); ta = (a-ag[k])/(ag[k+1]-ag[k])
    return float((1-tv)*((1-ta)*M[i,k]+ta*M[i,k+1]) + tv*((1-ta)*M[i+1,k]+ta*M[i+1,k+1]))

def optimal_stop(v0, obs, NJ=81):
    """TRUE discrete optimum (the ruler): best coast length + trim jerk, then exact brake."""
    x, v, a = 0., float(v0), 0.; st = [(x, v, a)]
    for _ in range(90):
        x, v, a = exact_step(x, v, a, max(min(-a/DT, J_MAX), -J_MAX))
        if obs-x <= 0 or v <= 1e-9: break
        st.append((x, v, a))
    for k in range(len(st)-1, -1, -1):
        xs, vs, as_ = st[k]; best = None
        xe, ok = brake_from(xs, vs, as_)
        if ok and obs-xe >= -1e-9: best = obs-xe
        for jf in np.linspace(-J_MAX, J_MAX, NJ):
            x2, v2, a2 = exact_step(xs, vs, as_, jf)
            if obs-x2 <= 0: continue
            xe2, ok2 = brake_from(x2, v2, a2)
            if ok2 and obs-xe2 >= -1e-9 and (best is None or obs-xe2 < best): best = obs-xe2
        if best is not None: return best
    return None

# ---------- data synthesis ----------
def gen_dataset(n_total=2_000_000, K=K_DEFAULT, seed=0, verbose=True):
    """Eight coverage streams. Futures are the EXACT closed-form solution (machine zero)."""
    rng = np.random.default_rng(seed)
    def s_bulk():    return rng.uniform(0.3,22), rng.uniform(0,14),  rng.uniform(-8,8), rng.uniform(-20,20)
    def s_fast():    return rng.uniform(1,7),    rng.uniform(11,14), rng.uniform(-8,8), rng.uniform(-20,20)
    def s_final():   return rng.uniform(0.15,3), rng.uniform(0,4),   rng.uniform(-6,6), rng.uniform(-20,20)
    def s_brake():   return rng.uniform(0.5,16), rng.uniform(2,14),  rng.uniform(-8,-2),rng.uniform(-20,20)
    def s_accel():   return rng.uniform(2,20),   rng.uniform(0,12),  rng.uniform(2,8),  rng.uniform(-20,20)
    def s_bound():
        v = rng.uniform(3,14); d = mbd_va(v,0.); return d*rng.uniform(0.75,1.35), v, rng.uniform(-8,2), rng.uniform(-20,20)
    def s_jerkext(): return rng.uniform(0.5,20), rng.uniform(0,14),  rng.uniform(-8,8), rng.choice([-20.,-20.,20.,20.,rng.uniform(-20,20)])
    def s_edge():
        v = rng.uniform(13.,14.); d = mbd_va(v,0.); return d*rng.uniform(0.90,1.20), v, rng.uniform(-8,4), rng.uniform(-20,20)
    W = [(s_bulk,.315),(s_fast,.100),(s_final,.080),(s_brake,.086),
         (s_accel,.063),(s_bound,.129),(s_jerkext,.057),(s_edge,.170)]
    S0 = np.empty((n_total,3)); AA = np.empty((n_total,1)); FUT = np.empty((n_total,K,3))
    i = 0
    for fn, frac in W:
        n = int(n_total*frac) if fn is not W[-1][0] else n_total-i
        for _ in range(min(n, n_total-i)):
            g, v, a, j = fn()
            x, vv, aa = 0., v, a
            S0[i] = (g, v, a); AA[i] = j
            for k in range(K):
                x, vv, aa = exact_step(x, vv, aa, j); FUT[i,k] = (g-x, vv, aa)
            i += 1
            if verbose and i % 250_000 == 0: print(f"    {i}/{n_total}", flush=True)
        if i >= n_total: break
    safe = np.array([1.0 if S0[i,0] >= mbd_va(S0[i,1], S0[i,2]) else 0.0 for i in range(n_total)], np.float32)
    return S0[:n_total], AA[:n_total], FUT[:n_total], safe

def normalisers(S0, AA, mode='phys'):
    """'phys' = physics non-dimensional (recommended); 'sd' = sample standard deviation."""
    if mode == 'phys':
        return np.zeros(3), SCALE_PHYS.copy(), np.zeros(1), np.array([SCALE_J])
    return S0.mean(0), S0.std(0)+1e-6, AA.mean(0), AA.std(0)+1e-6
