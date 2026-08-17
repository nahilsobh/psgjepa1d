"""PSG-JEPA-1D training. Mirrors PSG-JEPA/train.py.

  python train.py                                  # defaults from configs/psgjepa1d.yaml
  python train.py loss.grounding.weight=0.0        # ablate grounding (= LeWM-1D baseline)
  python train.py loss.grounding.use_velocity=false
"""
import argparse, json, os, sys, time, numpy as np, torch, yaml
from psgjepa1d import JEPA1D, PSGGroundingHeads, training_step, LAM0
from psgjepa1d.data import gen_windows, normalisers

def deep_set(d, dotted, val):
    ks = dotted.split('.'); cur = d
    for k in ks[:-1]: cur = cur.setdefault(k, {})
    try: val = yaml.safe_load(val)
    except Exception: pass
    cur[ks[-1]] = val

def load_cfg(overrides):
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__),'configs/psgjepa1d.yaml')))
    for o in overrides:
        if '=' in o: deep_set(cfg, *o.split('=',1))
    return cfg

def build(cfg, dev, cache='window_cache.npz'):
    n, T = cfg['data']['n_windows'], cfg['data']['window_T']
    if os.path.exists(cache):
        d = np.load(cache)
        if len(d['S'])==n and d['S'].shape[1]==T: S,U = d['S'],d['U']
        else: S=None
    else: S=None
    if S is None:
        t0=time.time(); print(f"synthesising {n} windows (T={T}) ...", flush=True)
        S,U = gen_windows(n, T, seed=cfg['seed'])
        np.savez_compressed(cache, S=S, U=U); print(f"  {time.time()-t0:.0f}s")
    sm,ss,um,us = normalisers(S,U,cfg['data']['normaliser'])
    t = lambda a: torch.tensor(np.asarray(a,np.float32), device=dev)
    return dict(states=t((S-sm)/ss), actions=t((U-um)/us), sm=sm, ss=ss, um=um, us=us)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('overrides', nargs='*')
    ap.add_argument('--out', default='psgjepa1d.pt')
    ap.add_argument('--cache', default='window_cache.npz')
    A = ap.parse_args()
    cfg = load_cfg(A.overrides)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    print(f"device={dev}  reg={cfg['loss']['reg_type']}  lambda_g={cfg['loss']['grounding']['weight']}"
          f"  use_velocity={cfg['loss']['grounding']['use_velocity']}")
    D = build(cfg, dev, A.cache)
    m = JEPA1D(cfg['wm']['embed_dim'], cfg['wm']['hidden']).to(dev)
    g = cfg['loss']['grounding']
    heads = (PSGGroundingHeads(cfg['wm']['embed_dim'], len(g['state_idx']), len(g['joint_idx']),
                               max(len(g['vel_idx']),1), g['use_velocity'], g['hidden_dim'],
                               use_static=g.get('use_static', True),
                               use_transition=g.get('use_transition', True)).to(dev)
             if g['weight'] > 0 else None)
    if heads is not None:
        print(f"[grounding] weight={g['weight']} params={sum(p.numel() for p in heads.parameters())/1e6:.2f}M")
    params = list(m.parameters()) + (list(heads.parameters()) if heads else [])
    opt = torch.optim.AdamW(params, lr=float(cfg['optimizer']['lr']),
                        weight_decay=float(cfg['optimizer']['weight_decay']))
    E = cfg['trainer']['max_epochs']; bs = cfg['trainer']['batch_size']
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
    lc = dict(reg_type=cfg['loss']['reg_type'], reg_weight=cfg['loss']['reg_weight'],
              grounding_weight=g['weight'], state_idx=g['state_idx'],
              joint_idx=g['joint_idx'], vel_idx=g['vel_idx'],
              recon_weight=cfg['loss'].get('recon_weight', 0.0),
              recon_channel_weights=cfg['loss'].get('recon_channel_weights', None))
    N = len(D['states'])
    for ep in range(E):
        perm = torch.randperm(N, device=dev); tot=0.; nb=0
        for st in range(0,N,bs):
            b = perm[st:st+bs]
            out = training_step(m, heads, dict(states=D['states'][b], actions=D['actions'][b]), lc)
            opt.zero_grad(set_to_none=True); out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += float(out['loss']); nb += 1
        sch.step()
        if (ep+1) % 5 == 0 or ep == 0: print(f"  ep {ep+1:>3}/{E}  loss {tot/nb:.5f}", flush=True)
    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    torch.save(dict(model=m.state_dict(), cfg=cfg,
                    norm=dict(sm=D['sm'],ss=D['ss'],um=D['um'],us=D['us'])), A.out)
    print(f"saved {A.out}  (grounding heads intentionally discarded)")

if __name__ == '__main__': main()
