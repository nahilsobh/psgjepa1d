"""PSG-JEPA-1D: encoder + action-conditioned predictor (JEPA forward prediction + anti-collapse),
plus physical state grounding. Grounding heads are training-only and discarded at inference.

Mirrors PSG-JEPA/psgjepa/model.py:  Loss = L_JEPA + lambda_reg*SIGReg + lambda_g*L_grounding
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from .grounding import PSGGroundingHeads, grounding_loss


class JEPA1D(nn.Module):
    """encoder 3->H->H->D ; residual action-conditioned predictor ; decoder D->state.
    decoder_hidden=None: linear decoder (default). int: MLP decoder D->h->state."""
    def __init__(self, embed_dim=64, hidden=512, state_dim=3, decoder_hidden=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(nn.Linear(state_dim,hidden), nn.GELU(),
                                     nn.Linear(hidden,hidden), nn.GELU(),
                                     nn.Linear(hidden,embed_dim))
        self.p1 = nn.Linear(embed_dim+1, hidden); self.p2 = nn.Linear(hidden,hidden)
        self.p3 = nn.Linear(hidden, embed_dim)
        nn.init.normal_(self.p3.weight, 0, 0.02); nn.init.zeros_(self.p3.bias)
        if decoder_hidden is None:
            self.decoder = nn.Linear(embed_dim, state_dim)
        else:
            self.decoder = nn.Sequential(nn.Linear(embed_dim, decoder_hidden), nn.GELU(),
                                         nn.Linear(decoder_hidden, state_dim))
    def encode(self, s): return self.encoder(s)
    def predict(self, z, u):
        h = F.gelu(self.p1(torch.cat([z,u],-1))); h = F.gelu(self.p2(h))
        return z + self.p3(h)
    def decode(self, z): return self.decoder(z)


# ---------- anti-collapse regularisers (LeWM keeps SIGReg; others for the ablation) ----------
def _subsample(z, max_rows=4096):
    """Sliced regularisers cost O(rows x proj x knots). At batch 4096 x T=7 that is 28672 rows,
    which with num_proj=1024, knots=17 allocates ~4 GB and OOMs. The estimator is a population
    statistic, so a random subsample is an unbiased and much cheaper estimate."""
    if z.shape[0] <= max_rows: return z
    idx = torch.randperm(z.shape[0], device=z.device)[:max_rows]
    return z[idx]

def sigreg(z, num_proj=1024, knots=17, tmax=5.0, max_rows=4096):
    z = _subsample(z, max_rows)
    N,D = z.shape
    V = torch.randn(D,num_proj,device=z.device); V = V/V.norm(dim=0,keepdim=True)
    p = z@V; p = (p-p.mean(0))/(p.std(0,unbiased=False)+1e-6)
    t = torch.linspace(-tmax,tmax,knots,device=z.device).view(1,1,-1)
    ang = p.unsqueeze(-1)*t
    re, im = ang.cos().mean(0), ang.sin().mean(0)
    return ((re-torch.exp(-0.5*t.view(1,-1)**2))**2 + im**2).mean()
def vicreg_reg(z, sc=25., cc=1., max_rows=8192):
    z=_subsample(z,max_rows); zc=z-z.mean(0); std=torch.sqrt(zc.var(0,unbiased=False)+1e-4); N,D=z.shape
    cov=(zc.T@zc)/max(N-1,1)
    return sc*F.relu(1.-std).mean()+cc*(cov-torch.diag(torch.diag(cov))).pow(2).sum()/D
def barlow_reg(z, lam=5e-3, max_rows=8192):
    z=_subsample(z,max_rows); N,D=z.shape; zn=(z-z.mean(0))/(z.std(0,unbiased=False)+1e-6); c=(zn.T@zn)/N
    return (torch.diagonal(c)-1).pow(2).sum()+lam*(c-torch.diag(torch.diagonal(c))).pow(2).sum()
def visreg(z, Ks=32, max_rows=4096):
    z=_subsample(z,max_rows)
    N,D=z.shape; mu=z.mean(0); std=torch.sqrt(z.var(0,unbiased=False)+1e-6)
    V=torch.randn(D,Ks,device=z.device); V=V/V.norm(dim=0,keepdim=True)
    ps,_=torch.sort(((z-mu)/(std+1e-6))@V,0)
    q=(torch.arange(N,device=z.device,dtype=z.dtype)+.5)/N
    tg=math.sqrt(2.)*torch.erfinv(2*q-1)
    return ((std-1.)**2).mean()+((ps-tg.unsqueeze(1))**2).mean()+(mu**2).mean()
REG = dict(sigreg=sigreg, vicreg=vicreg_reg, barlow=barlow_reg, visreg=visreg,
           stopgrad=lambda z: z.sum()*0.)
LAM0 = dict(sigreg=0.09, vicreg=1.0, barlow=8.0, visreg=20.0, stopgrad=0.0)


def training_step(model, heads, batch, cfg):
    """batch: dict(states (B,T,3) normalised, actions (B,T-1,1) normalised).
       Teacher-forced one-step forward prediction over the window, as upstream."""
    S, U = batch['states'], batch['actions']
    B,T,_ = S.shape
    emb = model.encode(S)                                   # (B,T,D) -- all frames
    zpred = model.predict(emb[:, :-1], U)                   # (B,T-1,D) teacher-forced
    out = {}
    out['pred_loss'] = (zpred - emb[:, 1:].detach()).pow(2).mean()
    reg_fn = REG[cfg['reg_type']]
    out['reg_loss'] = reg_fn(emb.reshape(B*T,-1))
    loss = out['pred_loss'] + cfg['reg_weight']*out['reg_loss']
    rw = cfg.get('recon_weight', 0.0)
    if rw > 0.0:
        diff2 = (model.decode(emb) - S).pow(2)                          # (B,T,state_dim)
        cw = cfg.get('recon_channel_weights', None)
        if cw is not None:
            w = torch.tensor(cw, device=diff2.device, dtype=diff2.dtype)
            out['recon_loss'] = (diff2 * w).mean()
        else:
            out['recon_loss'] = diff2.mean()
        loss = loss + rw * out['recon_loss']
    gw = cfg.get('grounding_weight', 0.0)
    if heads is not None and gw > 0.0:
        gl = grounding_loss(heads, emb, S,
                            state_idx=cfg['state_idx'], joint_idx=cfg['joint_idx'],
                            vel_idx=cfg['vel_idx'])
        loss = loss + gw*gl['loss']
        out.update({f"g_{k}": v for k,v in gl.items() if k != 'loss'})
    out['loss'] = loss
    return out
