"""PSG-JEPA-1D full evaluation harness.

Runs, for each checkpoint in ckpt/:
  - level 1 (identifiability):     probes()             -- linear + MLP, state + transition
  - level 2 (control):             closed-loop drive on the recoverable scenario grid
  - level 3 (diagnostics):         E1..E4 per component, compounding, RankMe
  - gauge test on one of the checkpoints (invariance sanity check)

Writes a single eval_results.json.
"""
import os, sys, json, time, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psgjepa1d import JEPA1D
from psgjepa1d import world as W
from psgjepa1d.data import gen_windows, normalisers
from eval.evaluate import probes, fit_decoder, taxonomy, gauge


# ------------------------------ RankMe ------------------------------
@torch.no_grad()
def rankme(m, dev, sm, ss, n=8000, seed=11):
    rng = np.random.default_rng(seed)
    S = np.stack([rng.uniform(0.3, 22, n), rng.uniform(0, 14, n), rng.uniform(-8, 8, n)], 1)
    z = m.encode(torch.tensor(((S - sm) / ss).astype(np.float32), device=dev)).cpu().numpy()
    zc = z - z.mean(0)
    sv = np.linalg.svd(zc, compute_uv=False)
    p = sv / (sv.sum() + 1e-7) + 1e-7
    return float(np.exp(-(p * np.log(p)).sum()))


# --------------------- Model brake-distance grid ---------------------
@torch.no_grad()
def build_model_brake_grid(m, dec, dev, sm, ss, um, us,
                           v_grid=None, a_grid=None, gap0=15.0, max_steps=80):
    """For each (v,a) on the grid: encode (gap0,v,a), roll predictor forward under the
    max-brake template (jerk=-J_MAX until a<=-A_MAX then 0) until the DECODED v <= 0 or
    max_steps. Record model's decoded displacement gap0 - final_decoded_gap.

    This is entirely a model-in-the-loop quantity: the controller uses it to decide when
    it has run out of room to defer the terminal brake.
    """
    if v_grid is None: v_grid = np.linspace(0.0, 14.0, 15)
    if a_grid is None: a_grid = np.linspace(-8.0, 8.0, 9)
    sm_t = torch.tensor(sm.astype(np.float32), device=dev)
    ss_t = torch.tensor(ss.astype(np.float32), device=dev)
    um_t = torch.tensor(um.astype(np.float32), device=dev)
    us_t = torch.tensor(us.astype(np.float32), device=dev)

    VV, AA = np.meshgrid(v_grid, a_grid, indexing='ij')
    G = VV.shape[0] * VV.shape[1]
    v_flat = VV.reshape(-1)
    a_flat = AA.reshape(-1)
    s0 = np.stack([np.full(G, gap0, np.float32), v_flat.astype(np.float32),
                   a_flat.astype(np.float32)], 1)
    s0_t = torch.tensor(s0, device=dev)
    z = m.encode((s0_t - sm_t) / ss_t)                           # (G, D)
    dec0 = dec(z) * ss_t + sm_t                                  # (G, 3) initial decoded
    gap_dec0 = dec0[:, 0].cpu().numpy().copy()

    # rollout under braking template; track decoded (v, a) to pick jerk each step
    a_cur = a_flat.copy().astype(np.float32)
    active = np.ones(G, dtype=bool)
    displacement = np.zeros(G, np.float32)
    for step in range(max_steps):
        # braking template: -J_MAX unless already at -A_MAX
        jerk = np.where(a_cur > -W.A_MAX + 1e-3, -W.J_MAX, 0.0).astype(np.float32)
        u = torch.tensor(jerk.reshape(-1, 1), device=dev)
        z = m.predict(z, (u - um_t) / us_t)
        d = (dec(z) * ss_t + sm_t).cpu().numpy()                 # (G, 3) [gap, v, a]
        gap_cur, v_cur, a_cur = d[:, 0], d[:, 1], d[:, 2]
        # displacement at this step = gap_dec0 - current gap
        newly_stopped = active & (v_cur <= 1e-3)
        displacement[newly_stopped] = gap_dec0[newly_stopped] - gap_cur[newly_stopped]
        active = active & (v_cur > 1e-3)
        if not active.any():
            break
    # anything still active: take current decoded displacement as its brake distance
    if active.any():
        displacement[active] = gap_dec0[active] - gap_cur[active]

    return v_grid, a_grid, displacement.reshape(len(v_grid), len(a_grid))


def _bilerp(vg, ag, M, v, a):
    """Bilinear interpolation on the (v, a) grid, with clamping to edges."""
    v = max(vg[0], min(vg[-1], float(v)))
    a = max(ag[0], min(ag[-1], float(a)))
    i = int(np.clip(np.searchsorted(vg, v) - 1, 0, len(vg) - 2))
    k = int(np.clip(np.searchsorted(ag, a) - 1, 0, len(ag) - 2))
    tv = (v - vg[i]) / (vg[i+1] - vg[i])
    ta = (a - ag[k]) / (ag[k+1] - ag[k])
    return float((1-tv) * ((1-ta)*M[i,k] + ta*M[i,k+1])
                 +  tv    * ((1-ta)*M[i+1,k] + ta*M[i+1,k+1]))


# ------------------------------ Controller ------------------------------
@torch.no_grad()
def drive_scenario(m, dec, dev, sm, ss, um, us, brake_grid, v0, wall,
                   nj=41, safety=0.25, target_excess=None, T_max=250):
    # aim at the safety threshold => pick the tightest feasible plan
    if target_excess is None: target_excess = safety
    """Closed-loop stop-in-front-of-wall.

    Strategy (mirrors world.optimal_stop but with the MODEL as the world model):
      1. From the current state, roll a coast trajectory forward in the MODEL, one tick
         at a time, using jerk that zeroes out the acceleration (clipped to J_MAX).
      2. At each coast index k, and for each trim jerk in an nj-point linspace, predict
         one more step with that jerk and query the MODEL's brake-distance grid at the
         result. Predicted resting gap = predicted_gap_after_trim - brake_dist.
      3. Pick the (k, trim_jerk) with the smallest non-negative predicted excess.
      4. Execute in the EXACT world: k coast ticks + one trim tick + brake_from().

    brake_from() internally applies max-brake ticks until plan_terminal returns a plan,
    then commits to (one trim tick + n ramp ticks) -- no re-planning inside the terminal
    sequence. (Re-solving each tick gives ~0.20 m instead of ~0.013 m.)
    """
    vg, ag, M = brake_grid
    sm_t = torch.tensor(sm.astype(np.float32), device=dev)
    ss_t = torch.tensor(ss.astype(np.float32), device=dev)
    um_t = torch.tensor(um.astype(np.float32), device=dev)
    us_t = torch.tensor(us.astype(np.float32), device=dev)
    trims = np.linspace(-W.J_MAX, W.J_MAX, nj).astype(np.float32)
    trims_t = torch.tensor(trims.reshape(-1, 1), device=dev)

    K_max = 90
    x, v, a = 0.0, float(v0), 0.0

    # ---- MODEL rollout of the coast trajectory (k = 0 .. K_max) ----
    s0 = torch.tensor([[wall - x, v, a]], dtype=torch.float32, device=dev)
    z = m.encode((s0 - sm_t) / ss_t)                                        # (1, D)
    coast_z = [z.clone()]
    v_coast = [float(v)]; a_coast = [float(a)]; gap_coast = [float(wall - x)]
    for k in range(1, K_max + 1):
        # coast jerk = zero-out accel (clipped)
        j_coast = max(-W.J_MAX, min(W.J_MAX, -a_coast[-1] / W.DT))
        u = torch.tensor([[j_coast]], dtype=torch.float32, device=dev)
        z = m.predict(z, (u - um_t) / us_t)
        dec_step = (dec(z) * ss_t + sm_t).cpu().numpy()[0]                  # [gap, v, a]
        coast_z.append(z.clone())
        gap_coast.append(float(dec_step[0]))
        v_coast.append(float(dec_step[1]))
        a_coast.append(float(dec_step[2]))
        if dec_step[1] <= 1e-3 or dec_step[0] <= safety:
            break

    # ---- for each k, fan out trim jerks in batch, evaluate predicted excess ----
    # Bias selection toward candidates whose predicted excess is closest to `target_excess`
    # from above. `safety` is set to > the model's own decode error (~E1_gap) so we don't
    # commit to plans the model claims are safe by less than its own noise level.
    best = None  # (k, j_trim, |excess - target|)
    for k in range(len(coast_z)):
        z_k = coast_z[k].expand(nj, -1)
        zp = m.predict(z_k, (trims_t - um_t) / us_t)
        d = (dec(zp) * ss_t + sm_t).cpu().numpy()                            # (nj, 3)
        for i in range(nj):
            gp, vp, ap = float(d[i, 0]), float(d[i, 1]), float(d[i, 2])
            if gp <= safety: continue                                        # would crash
            bd = _bilerp(vg, ag, M, vp, ap)
            excess = gp - bd
            if excess < safety: continue                                     # not enough margin
            score = abs(excess - target_excess)
            if best is None or score < best[2]:
                best = (k, float(trims[i]), score)

    # ---- fall through: brake now if nothing feasible ----
    if best is None:
        xf, ok = W.brake_from(x, v, a)
        excess = wall - xf
        return excess, (excess < -1e-6) or (not ok)

    k_star, j_trim, _ = best
    # execute k_star coast ticks in the EXACT world (matches the coast policy the model saw)
    for _ in range(k_star):
        j_coast_exact = max(-W.J_MAX, min(W.J_MAX, -a / W.DT))
        x, v, a = W.exact_step(x, v, a, j_coast_exact)
        if v <= 1e-9 or (wall - x) <= 0:
            excess = wall - x
            return excess, excess < -1e-6
    # one trim tick, then commit to brake_from (which does plan_terminal + ramp)
    x, v, a = W.exact_step(x, v, a, j_trim)
    if (wall - x) < -1e-6:
        return wall - x, True
    xf, ok = W.brake_from(x, v, a)
    excess = wall - xf
    return excess, (excess < -1e-6) or (not ok)


def scenario_grid():
    v0s = np.arange(3.0, 14.5, 1.0)
    walls = np.arange(6.0, 22.5, 1.0)
    scen = []
    for v0 in v0s:
        for wall in walls:
            if wall >= W.mbd_va(float(v0), 0.0) - 1e-9:
                scen.append((float(v0), float(wall)))
    return scen


@torch.no_grad()
def control_eval_at_margin(m, dec, dev, sm, ss, um, us, brake_grid, scenarios, safety):
    """Drive every scenario at a specific safety margin. Returns per-scenario detail so we
    can later intersect success masks across arms."""
    details = []
    for (v0, wall) in scenarios:
        excess, crashed = drive_scenario(
            m, dec, dev, sm, ss, um, us, brake_grid, v0, wall, safety=safety
        )
        details.append(dict(v0=v0, wall=wall, excess=excess, crashed=bool(crashed)))
    excesses = [d['excess'] for d in details if not d['crashed']]
    crashes = sum(1 for d in details if d['crashed'])
    return dict(
        safety=safety,
        n_scenarios=len(scenarios),
        n_valid=len(excesses),
        crashes=crashes,
        excess_mean=float(np.mean(excesses)) if excesses else None,
        excess_median=float(np.median(excesses)) if excesses else None,
        excess_p90=float(np.percentile(excesses, 90)) if excesses else None,
        per_scenario=details,
    )


def sweep_margins(m, dec, dev, sm, ss, um, us, brake_grid, scenarios, margins):
    """For each safety margin in `margins`, run the full scenario grid.  Returns a dict
    keyed by margin; each entry has per_scenario / crashes / excess_mean."""
    out = {}
    for s in margins:
        t0 = time.time()
        r = control_eval_at_margin(m, dec, dev, sm, ss, um, us, brake_grid, scenarios, s)
        out[s] = r
        print(f"    margin={s:.3f}m  crashes={r['crashes']:3d}/{r['n_scenarios']}  "
              f"excess_mean={('%.4f'%r['excess_mean']) if r['excess_mean'] is not None else '  n/a  '}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    return out


def pick_tightest_safe(sweep, threshold=2):
    """Return the smallest margin whose crash count is <= threshold, else None."""
    for s in sorted(sweep.keys()):
        if sweep[s]['crashes'] <= threshold:
            return s
    return None


# ------------------------------ optimum ------------------------------
def optimum_mean_excess(scenarios):
    xs = []
    for (v0, wall) in scenarios:
        e = W.optimal_stop(float(v0), float(wall))
        if e is not None: xs.append(e)
    return float(np.mean(xs)), len(xs)


# ------------------------------ Data prep ------------------------------
def load_windows(dev, n=500000, T=7, seed=3072, cache='window_cache.npz'):
    if os.path.exists(cache):
        d = np.load(cache)
        S, U = d['S'], d['U']
        if len(S) > n:
            S = S[:n]; U = U[:n]
    else:
        S, U = gen_windows(n, T, seed=seed)
    sm, ss, um, us = normalisers(S, U, 'phys')
    t = lambda a: torch.tensor(a.astype(np.float32), device=dev)
    return dict(
        states=t((S - sm) / ss), actions=t((U - um) / us),
        sm=sm, ss=ss, um=um, us=us,
    )


# ------------------------------ Main ------------------------------
def load_checkpoint(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = ck['cfg']; norm = ck['norm']
    m = JEPA1D(int(cfg['wm']['embed_dim']), int(cfg['wm']['hidden'])).to(dev)
    m.load_state_dict(ck['model'], strict=False); m.eval()  # decoder is optional
    return m, cfg, norm


MARGINS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75,
           1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CRASH_THRESHOLD = 2  # tightest-safe = smallest margin with crashes <= threshold


def evaluate_checkpoint(path, dev, D, scenarios, do_gauge=False, margins=MARGINS,
                        crash_threshold=CRASH_THRESHOLD):
    print(f"\n--- {path} ---", flush=True)
    m, cfg, norm = load_checkpoint(path, dev)
    sm, ss, um, us = norm['sm'], norm['ss'], norm['um'], norm['us']

    r_probes = probes(m, dev, sm, ss)
    print(f"  probes: {r_probes}", flush=True)

    t0 = time.time()
    dec = fit_decoder(m, D, dev)
    print(f"  fit_decoder: {time.time()-t0:.1f}s", flush=True)

    r_tax = taxonomy(m, dec, dev, sm, ss, um, us)
    print(f"  taxonomy E1={r_tax['E1']}  E4={r_tax['E4']}  compound={r_tax['compound']:.3f}", flush=True)

    rme = rankme(m, dev, sm, ss)
    print(f"  RankMe: {rme:.2f}", flush=True)

    t0 = time.time()
    bgrid = build_model_brake_grid(m, dec, dev, sm, ss, um, us)
    print(f"  brake grid: {time.time()-t0:.1f}s", flush=True)

    print(f"  margin sweep ({len(margins)} points, threshold<= {crash_threshold} crashes):",
          flush=True)
    sweep = sweep_margins(m, dec, dev, sm, ss, um, us, bgrid, scenarios, margins)
    m_star = pick_tightest_safe(sweep, threshold=crash_threshold)
    if m_star is None:
        print(f"  UNSAFE: no margin in {margins} achieved <= {crash_threshold} crashes",
              flush=True)
    else:
        r = sweep[m_star]
        print(f"  tightest-safe margin = {m_star:.3f}m  crashes={r['crashes']}  "
              f"excess_mean={r['excess_mean']:.4f}m  n_valid={r['n_valid']}", flush=True)

    # Trim per_scenario details out of sweep before json — keep only summary + chosen margin
    sweep_summary = {
        str(s): {k: v for k, v in sweep[s].items() if k != 'per_scenario'}
        for s in sweep
    }
    per_scenario_at_chosen = sweep[m_star]['per_scenario'] if m_star is not None else None

    out = dict(
        probes=r_probes, taxonomy=r_tax, rankme=rme,
        margin_sweep=sweep_summary,
        chosen_margin=m_star,
        crash_threshold=crash_threshold,
        control_at_chosen=(
            {k: v for k, v in sweep[m_star].items() if k != 'per_scenario'}
            if m_star is not None else None
        ),
        per_scenario_at_chosen=per_scenario_at_chosen,
    )
    if do_gauge:
        g = gauge(m, dec, dev, sm, ss, um, us)
        out['gauge'] = g
        print(f"  gauge: {g}", flush=True)
    return out


def paired_common_analysis(arms_results):
    """Compute the excess on the set of scenarios where every arm (with a valid
    chosen margin) succeeded.  Immune to survivorship bias."""
    arms_with_choice = {a: r for a, r in arms_results.items()
                        if r['chosen_margin'] is not None
                        and r['per_scenario_at_chosen'] is not None}
    if len(arms_with_choice) < 2:
        return dict(n_common=0, per_arm={}, note='fewer than 2 arms with a valid margin')
    # per_scenario lists are aligned by index (same scenarios in same order)
    n = len(next(iter(arms_with_choice.values()))['per_scenario_at_chosen'])
    common_mask = [
        all(not arms_with_choice[a]['per_scenario_at_chosen'][i]['crashed']
            for a in arms_with_choice)
        for i in range(n)
    ]
    n_common = sum(common_mask)
    per_arm = {}
    for a in arms_with_choice:
        per_scen = arms_with_choice[a]['per_scenario_at_chosen']
        excesses = [per_scen[i]['excess'] for i in range(n) if common_mask[i]]
        per_arm[a] = dict(
            n=n_common,
            excess_mean=float(np.mean(excesses)) if excesses else None,
            excess_median=float(np.median(excesses)) if excesses else None,
            excess_p90=float(np.percentile(excesses, 90)) if excesses else None,
        )
    return dict(n_common=n_common, per_arm=per_arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='ckpt')
    ap.add_argument('--out', default='eval_results.json')
    ap.add_argument('--cache', default='window_cache.npz')
    ap.add_argument('--n-decoder', type=int, default=150000)
    ap.add_argument('--gauge-on', default='full',
                    help='which arm name to run the gauge test on')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  ckpt-dir={args.ckpt_dir}", flush=True)

    arms = ['baseline', 'static', 'transition', 'full']
    ckpts = {a: os.path.join(args.ckpt_dir, f"{a}.pt") for a in arms}
    missing = [a for a, p in ckpts.items() if not os.path.exists(p)]
    if missing:
        print(f"WARNING: missing checkpoints: {missing}", flush=True)
        arms = [a for a in arms if a not in missing]

    scenarios = scenario_grid()
    opt_mean, opt_n = optimum_mean_excess(scenarios)
    print(f"scenarios={len(scenarios)}  optimum mean excess={opt_mean:.5f} m (n={opt_n})",
          flush=True)

    D = load_windows(dev, cache=args.cache)

    results = dict(
        meta=dict(device=str(dev), scenarios=len(scenarios),
                  optimum_mean_excess=opt_mean, optimum_n=opt_n,
                  margins=list(MARGINS), crash_threshold=CRASH_THRESHOLD),
        arms={},
    )
    for a in arms:
        results['arms'][a] = evaluate_checkpoint(
            ckpts[a], dev, D, scenarios, do_gauge=(a == args.gauge_on)
        )

    # Paired common-scenario analysis across arms with a valid chosen margin.
    results['paired_common'] = paired_common_analysis(results['arms'])

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    # ---- summary tables ----
    print("\n=== per-arm at tightest-safe margin (<= {} crashes/{}) ==="
          .format(CRASH_THRESHOLD, len(scenarios)))
    print("arm        | E1_gap  | E4_gap  | compound | margin*  | crashes | n_valid | excess_mean (m)")
    print("-" * 100)
    for a in arms:
        r = results['arms'][a]
        m_star = r['chosen_margin']
        if m_star is None:
            tag = "UNSAFE"
            print(f"{a:10s} | {r['taxonomy']['E1'][0]:.4f}  | {r['taxonomy']['E4'][0]:.4f}  | "
                  f"{r['taxonomy']['compound']:7.3f}  | {tag:8s} | (no margin achieved threshold)")
        else:
            c = r['control_at_chosen']
            print(f"{a:10s} | {r['taxonomy']['E1'][0]:.4f}  | {r['taxonomy']['E4'][0]:.4f}  | "
                  f"{r['taxonomy']['compound']:7.3f}  | {m_star:6.3f}m  | "
                  f"{c['crashes']:7d} | {c['n_valid']:7d} | {c['excess_mean']:9.4f}")

    # crash-rate curve
    print("\n=== crashes vs safety margin ===")
    header = "margin (m) | " + " | ".join(f"{a:10s}" for a in arms)
    print(header)
    print("-" * len(header))
    for s in MARGINS:
        cells = []
        for a in arms:
            crashes = results['arms'][a]['margin_sweep'][str(s)]['crashes']
            cells.append(f"{crashes:10d}")
        print(f"{s:9.3f}  | " + " | ".join(cells))

    # excess vs margin (only where the arm survived enough to have a mean)
    print("\n=== excess_mean (m) vs safety margin (blank = all crashed or no survivors) ===")
    print(header)
    print("-" * len(header))
    for s in MARGINS:
        cells = []
        for a in arms:
            e = results['arms'][a]['margin_sweep'][str(s)]['excess_mean']
            cells.append(f"{e:10.4f}" if e is not None else "         ")
        print(f"{s:9.3f}  | " + " | ".join(cells))

    # paired intersection
    pc = results['paired_common']
    print(f"\n=== paired common-scenario excess (n_common = {pc['n_common']}) ===")
    if pc['n_common'] == 0:
        print("(no arms with a valid chosen margin, or empty intersection)")
    else:
        print("arm        | excess_mean (m) | excess_median | excess_p90")
        for a in arms:
            if a in pc['per_arm']:
                r = pc['per_arm'][a]
                em = f"{r['excess_mean']:.4f}" if r['excess_mean'] is not None else "  n/a "
                md = f"{r['excess_median']:.4f}" if r['excess_median'] is not None else "  n/a "
                p9 = f"{r['excess_p90']:.4f}" if r['excess_p90'] is not None else "  n/a "
                print(f"{a:10s} | {em:>15} | {md:>13} | {p9:>10}")
            else:
                print(f"{a:10s} | UNSAFE (no chosen margin)")


if __name__ == '__main__':
    main()
