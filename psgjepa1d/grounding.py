"""Physical state grounding for PSG-JEPA-1D (training-only; heads discarded at inference).

FAITHFUL PORT of PSG-JEPA/psgjepa/grounding.py, adapted to the 1-D car.

    L_static  : per-latent state grounding           H_s : z_t            -> s_t
    L_dynamic : latent-pair transition grounding
                  - multi-horizon POSITION change    H_d : (z_t, z_{t+k}) -> dgap_{t,k}   (always)
                  - instantaneous velocity           H_v : (z_t, z_{t+1}) -> v_t          (optional)

    L_PSG = L_JEPA + lambda_g * (L_static + L_dynamic)

MAPPING robot arm -> 1-D car:
    robot s_t (joint angles + effector + gripper)   ->  car s_t = [gap, v, a]
    robot q_t (joint ANGLES, position-like dof)     ->  car q_t = [gap]
    robot joint velocity                            ->  car v_t = [v]
Delta-q is the change of the POSITION-like coordinate over a horizon; its derivative is logged
separately as the velocity target. In the car, gap plays the role of q and v of qdot.

TWO DETAILS THAT MATTER (both wrong in our earlier ad-hoc test):
  1. horizons weighted EQUALLY, not pairs -- short horizons yield more pairs, so a per-pair mean
     lets k=1 dominate. Upstream: torch.stack(per_horizon_means).mean().
  2. the velocity head is ON by default upstream (use_velocity: true).
"""
import torch
from torch import nn


def _masked_mse(pred, tgt):
    finite = torch.isfinite(tgt)
    if bool(finite.all()):
        return ((pred - tgt) ** 2).mean()
    sq = (pred - torch.nan_to_num(tgt, 0.0, 0.0, 0.0)) ** 2
    return (sq * finite).sum() / finite.sum().clamp(min=1)


def _mlp(d_in, d_out, hidden=256):
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, d_out),
    )


class PSGGroundingHeads(nn.Module):
    def __init__(self, embed_dim, state_dim=3, joint_dim=1, vel_dim=1,
                 use_velocity=True, hidden=256, use_static=True, use_transition=True):
        super().__init__()
        self.use_velocity   = use_velocity
        self.use_static     = use_static and state_dim > 0
        self.use_transition = use_transition and joint_dim > 0
        self.state_head  = _mlp(embed_dim, state_dim, hidden) if self.use_static else None
        self.djoint_head = _mlp(2*embed_dim, joint_dim, hidden) if self.use_transition else None
        self.vel_head    = _mlp(2*embed_dim, vel_dim, hidden) if use_velocity else None


def grounding_loss(heads, emb, state, state_idx=(0,1,2), joint_idx=(0,), vel_idx=(1,)):
    """emb (B,T,D) latents; state (B,T,S) normalised true states [gap, v, a]."""
    B, T, D = emb.shape
    zero = emb.sum() * 0.0                     # keeps dtype/device, contributes nothing
    logs = {}

    # --- static grounding (ablatable) ---
    if heads.use_static:
        s_tgt  = state[..., list(state_idx)]
        s_pred = heads.state_head(emb.reshape(B*T, D)).reshape(B, T, -1)
        l_static = _masked_mse(s_pred, s_tgt); logs["static"] = l_static.detach()
    else:
        l_static = zero

    # --- multi-horizon transition grounding (ablatable) ---
    if heads.use_transition:
        q = state[..., list(joint_idx)]
        dj_terms = []
        for k in range(1, T):
            za = emb[:, :T-k].reshape(B*(T-k), D)
            zb = emb[:,  k:].reshape(B*(T-k), D)
            dj_pred = heads.djoint_head(torch.cat([za, zb], -1)).reshape(B, T-k, -1)
            dj_terms.append(_masked_mse(dj_pred, q[:, k:] - q[:, :T-k]))
        l_djoint = torch.stack(dj_terms).mean()  # EQUAL WEIGHT PER HORIZON, not per pair
        logs["djoint"] = l_djoint.detach()
    else:
        l_djoint = zero
    l_dynamic = l_djoint

    if heads.use_velocity:
        v_tgt = state[:, :-1][..., list(vel_idx)]
        za = emb[:, :-1].reshape(B*(T-1), D); zb = emb[:, 1:].reshape(B*(T-1), D)
        v_pred = heads.vel_head(torch.cat([za, zb], -1)).reshape(B, T-1, -1)
        l_vel = _masked_mse(v_pred, v_tgt)
        l_dynamic = l_dynamic + l_vel
        logs["velocity"] = l_vel.detach()

    logs["loss"] = l_static + l_dynamic
    return logs
