"""Bake world-model trajectories for every (v0, wall) slider grid combination.

For each (v0, wall) that admits a valid stopping plan under the analytical
optimum:
  1. Extract the jerk sequence that the analytical controller commits to.
  2. Encode the initial state (wall, v0, 0), then step the JEPA1D predictor
     through that same jerk sequence.
  3. Decode each step to physical units and save the model's decoded
     (gap, v, a) trajectory.

Output: demo_data.json in the repo root, consumed by demo.html for the
second (world-model) track.
"""
import os, sys, json, argparse, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psgjepa1d import JEPA1D
from psgjepa1d import world as W


def brake_jerks_from(x, v, a):
    """Reproduces W.brake_from and returns the sequence of jerks applied."""
    jerks = []
    for _ in range(600):
        p = W.plan_terminal(v, a)
        if p is not None:
            jt, n, jt2 = p
            jerks.append(jt)
            x, v, a = W.exact_step(x, v, a, jt)
            for _ in range(n):
                jerks.append(jt2)
                x, v, a = W.exact_step(x, v, a, jt2)
            return jerks
        j = -W.J_MAX if a > -W.A_MAX else 0.0
        jerks.append(j)
        x, v, a = W.exact_step(x, v, a, j)
        if v <= 1e-12:
            return jerks
    return jerks


def optimal_stop_plan(v0, wall, NJ=81):
    """Returns (jerks_list, excess_m) or (None, None) if infeasible."""
    x, v, a = 0.0, float(v0), 0.0
    coast_states = [(x, v, a)]
    coast_jerks = []
    for _ in range(90):
        j_coast = max(min(-a / W.DT, W.J_MAX), -W.J_MAX)
        x, v, a = W.exact_step(x, v, a, j_coast)
        coast_jerks.append(j_coast)
        if wall - x <= 0 or v <= 1e-9:
            break
        coast_states.append((x, v, a))
    for k in range(len(coast_states) - 1, -1, -1):
        xs, vs, as_ = coast_states[k]
        best = None  # (trim_jerk_or_None, excess, brake_jerks)
        # no-trim option: brake immediately from coast state
        xe, ok = W.brake_from(xs, vs, as_)
        if ok and wall - xe >= -1e-9:
            bj = brake_jerks_from(xs, vs, as_)
            best = (None, wall - xe, bj)
        for jf in np.linspace(-W.J_MAX, W.J_MAX, NJ):
            x2, v2, a2 = W.exact_step(xs, vs, as_, jf)
            if wall - x2 <= 0:
                continue
            xe2, ok2 = W.brake_from(x2, v2, a2)
            if ok2 and wall - xe2 >= -1e-9:
                if best is None or (wall - xe2) < best[1]:
                    bj = brake_jerks_from(x2, v2, a2)
                    best = (float(jf), wall - xe2, bj)
        if best is not None:
            trim, excess, brake = best
            jerks = list(coast_jerks[:k])
            if trim is not None:
                jerks.append(trim)
            jerks.extend(brake)
            return jerks, float(excess)
    return None, None


@torch.no_grad()
def model_rollout(m, sm, ss, um, us, s0, jerks):
    """Encode s0, step through jerks, decode each step. Returns list of
    [gap, v, a] in physical units (one entry per applied jerk)."""
    t = lambda a: torch.tensor(np.asarray(a, np.float32))
    s0_t = t(np.asarray(s0).reshape(1, 3))
    z = m.encode((s0_t - t(sm.reshape(1, 3))) / t(ss.reshape(1, 3)))
    out = []
    ss_t = t(ss.reshape(1, 3)); sm_t = t(sm.reshape(1, 3))
    um_t = t(um.reshape(1, 1)); us_t = t(us.reshape(1, 1))
    for j in jerks:
        u = t(np.array([[j]], np.float32))
        z = m.predict(z, (u - um_t) / us_t)
        d = (m.decode(z) * ss_t + sm_t).cpu().numpy()[0]
        out.append([float(d[0]), float(d[1]), float(d[2])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpt/full_recon_two.pt')
    ap.add_argument('--out', default='demo_data.js',
                    help='output path; a .js is written that assigns window.bakedData')
    ap.add_argument('--v0-step', type=float, default=0.5)
    ap.add_argument('--wall-step', type=float, default=0.5)
    ap.add_argument('--v0-min', type=float, default=3.0)
    ap.add_argument('--v0-max', type=float, default=14.0)
    ap.add_argument('--wall-min', type=float, default=6.0)
    ap.add_argument('--wall-max', type=float, default=22.0)
    ap.add_argument('--arm-label', default='full_recon_two')
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck['cfg']; norm = ck['norm']
    m = JEPA1D(int(cfg['wm']['embed_dim']), int(cfg['wm']['hidden']),
               decoder_hidden=cfg['wm'].get('decoder_hidden'),
               two_decoders=cfg['wm'].get('two_decoders', False))
    m.load_state_dict(ck['model'], strict=False)
    m.eval()
    sm = np.asarray(norm['sm'], np.float32)
    ss = np.asarray(norm['ss'], np.float32)
    um = np.asarray(norm['um'], np.float32)
    us = np.asarray(norm['us'], np.float32)

    v0s = np.round(np.arange(args.v0_min, args.v0_max + 1e-6, args.v0_step), 1)
    walls = np.round(np.arange(args.wall_min, args.wall_max + 1e-6, args.wall_step), 1)

    n_feasible = 0; n_total = 0
    out = {
        'meta': {
            'arm': args.arm_label,
            'ckpt': args.ckpt,
            'v0_range': [float(v0s[0]), float(v0s[-1]), float(args.v0_step)],
            'wall_range': [float(walls[0]), float(walls[-1]), float(args.wall_step)],
            'dt': W.DT, 'A_max': W.A_MAX, 'J_max': W.J_MAX,
        },
        'traj': {},
    }
    for v0 in v0s:
        for wall in walls:
            n_total += 1
            jerks, excess = optimal_stop_plan(float(v0), float(wall))
            if jerks is None:
                continue
            n_feasible += 1
            model_traj = model_rollout(m, sm, ss, um, us, [wall, v0, 0.0], jerks)
            # Round to save space; 4 decimal places is plenty
            traj_rounded = [[round(g, 4), round(v, 4), round(a, 4)] for g, v, a in model_traj]
            key = f"{v0:.1f}_{wall:.1f}"
            out['traj'][key] = {
                'excess': round(excess, 6),
                'model': traj_rounded,
            }
        print(f"  v0={v0:.1f}  cumulative feasible={n_feasible}/{n_total}", flush=True)

    with open(args.out, 'w') as f:
        f.write('window.bakedData = ')
        json.dump(out, f, separators=(',', ':'))
        f.write(';\n')
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nwrote {args.out}  {n_feasible}/{n_total} feasible  ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
