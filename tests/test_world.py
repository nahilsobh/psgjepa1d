"""Physics and data-synthesis integrity. Requires only NumPy."""
import numpy as np
from psgjepa1d import world as W
from psgjepa1d.data import gen_windows, normalisers

def test_exact_dynamics():
    S, U = gen_windows(2000, T=7, seed=0, verbose=False)
    err = 0.0
    for i in np.random.default_rng(0).integers(0, len(S), 300):
        g = S[i,0,0]; x = 0.0; v = S[i,0,1]; a = S[i,0,2]
        for t in range(S.shape[1]-1):
            x, v, a = W.exact_step(x, v, a, U[i,t,0])
            err = max(err, abs((g-x)-S[i,t+1,0]) + abs(v-S[i,t+1,1]) + abs(a-S[i,t+1,2]))
    assert err == 0.0, f"windows disagree with dynamics: {err}"
    print(f"  exact dynamics: max err {err:.1e}  OK")

def test_scales():
    S, U = gen_windows(500, T=7, seed=0, verbose=False)
    _, ss, _, us = normalisers(S, U, 'phys')
    assert np.allclose(ss, [W.L0, W.V0, W.A_MAX]), ss
    assert np.allclose(us, [W.J_MAX]), us
    print(f"  physics scales L0={W.L0} V0={W.V0} A={W.A_MAX} J={W.J_MAX}  OK")

def test_analytical_optimum():
    for v0, obs, lo, hi in [(9.,14.,0.0,0.05), (14.,18.,0.0,0.06)]:
        g = W.optimal_stop(v0, obs)
        assert g is not None and lo <= g <= hi, (v0, obs, g)
    print("  analytical optimum in range  OK")

def test_terminal_lands_at_zero():
    for v0 in (3., 6., 9., 12., 14.):
        x, v, a = 0., v0, 0.
        for _ in range(600):
            p = W.plan_terminal(v, a)
            if p is not None:
                jt, n, j2 = p
                x, v, a = W.exact_step(x, v, a, jt)
                for _ in range(n): x, v, a = W.exact_step(x, v, a, j2)
                break
            x, v, a = W.exact_step(x, v, a, -W.J_MAX if a > -W.A_MAX else 0.0)
            if v <= 1e-12: break
        assert abs(v) < 1e-6 and abs(a) < 1e-6, (v0, v, a)
    print("  terminal solve lands v=0 AND a=0  OK")

if __name__ == '__main__':
    test_exact_dynamics(); test_scales(); test_analytical_optimum(); test_terminal_lands_at_zero()
    print("test_world: ALL PASS")
