"""Grounding shapes, ablation arms, and the two upstream details that are easy to get wrong.
   Requires torch."""
import torch, numpy as np
from psgjepa1d.grounding import PSGGroundingHeads, grounding_loss

B, T, D = 8, 7, 64

def _mk(**kw):
    d = dict(state_dim=3, joint_dim=1, vel_dim=1, use_velocity=True,
             hidden=32, use_static=True, use_transition=True); d.update(kw)
    return PSGGroundingHeads(D, **d)

def test_all_terms_finite():
    emb = torch.randn(B,T,D); st = torch.randn(B,T,3)
    L = grounding_loss(_mk(), emb, st, (0,1,2), (0,), (1,))
    assert set(L) == {'static','djoint','velocity','loss'}, sorted(L)
    assert torch.isfinite(L['loss']), L['loss']
    print("  full grounding: all terms finite  OK")

def test_ablation_arms():
    emb = torch.randn(B,T,D); st = torch.randn(B,T,3)
    arms = {
        'static only':     dict(use_static=True,  use_transition=False, use_velocity=False),
        'transition only': dict(use_static=False, use_transition=True,  use_velocity=True),
        'full':            dict(use_static=True,  use_transition=True,  use_velocity=True)}
    for nm, kw in arms.items():
        L = grounding_loss(_mk(**kw), emb, st, (0,1,2), (0,), (1,))
        assert torch.isfinite(L['loss']), f"{nm} produced non-finite loss"
        assert ('static' in L) == kw['use_static']
        assert ('djoint' in L) == kw['use_transition']
        assert ('velocity' in L) == kw['use_velocity']
        print(f"  ablation '{nm}': finite, correct terms  OK")

def test_horizons_weighted_equally():
    """The transition loss must weight HORIZONS equally, not PAIRS. Short horizons yield more
       pairs, so per-pair averaging would let k=1 dominate."""
    emb = torch.randn(B,T,D); st = torch.zeros(B,T,3)
    st[:,:,0] = torch.arange(T, dtype=torch.float)          # gap grows linearly -> dq = k
    h = _mk(use_velocity=False, use_static=False)
    L = grounding_loss(h, emb, st, (0,1,2), (0,), (1,))
    # recompute per-horizon and per-pair means; they must differ, and ours must match per-horizon
    with torch.no_grad():
        per_h = []; all_sq = []
        for k in range(1, T):
            za = emb[:, :T-k].reshape(B*(T-k), D); zb = emb[:, k:].reshape(B*(T-k), D)
            p = h.djoint_head(torch.cat([za, zb], -1)).reshape(B, T-k, -1)
            tgt = st[:,k:,0:1] - st[:,:T-k,0:1]
            per_h.append(((p-tgt)**2).mean()); all_sq.append(((p-tgt)**2).reshape(-1))
        per_horizon = torch.stack(per_h).mean()
        per_pair = torch.cat(all_sq).mean()
    assert torch.allclose(L['djoint'], per_horizon, atol=1e-6), (L['djoint'], per_horizon)
    assert not torch.allclose(per_horizon, per_pair, atol=1e-6), "test degenerate"
    print(f"  per-horizon {float(per_horizon):.4f} != per-pair {float(per_pair):.4f}; ours matches per-horizon  OK")

def test_velocity_head_default_on():
    import yaml, os
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'psgjepa1d.yaml')))
    assert cfg['loss']['grounding']['use_velocity'] is True, "upstream default is use_velocity: true"
    print("  velocity head ON by default (matches upstream)  OK")

def test_regularisers_do_not_oom():
    """sliced regularisers must subsample; a full 4096x7 batch would allocate ~4 GB."""
    from psgjepa1d.model import sigreg, visreg, vicreg_reg, barlow_reg
    z = torch.randn(28672, 64)
    for fn in (sigreg, visreg, vicreg_reg, barlow_reg):
        v = fn(z); assert torch.isfinite(v), fn.__name__
    print("  all regularisers finite on a 28672-row batch (subsampled)  OK")

if __name__ == '__main__':
    test_all_terms_finite(); test_ablation_arms(); test_horizons_weighted_equally()
    test_velocity_head_default_on(); test_regularisers_do_not_oom()
    print("test_grounding: ALL PASS")
