"""PSG-JEPA-1D evaluation, mirroring the upstream three levels:
   (1) latent identifiability via frozen probes   [upstream Table 1/2]
   (2) planning on frozen latents                 [upstream Table 3]  -> here: closed-loop control
   (3) diagnostics unique to an exactly-solvable world: E1-E4 mapping, compounding, gauge test
"""
import numpy as np, torch, torch.nn as nn, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psgjepa1d import JEPA1D
from psgjepa1d import world as W

@torch.no_grad()
def probes(m, dev, sm, ss, n=40000, seed=5):
    """Level 1: linear-ridge AND MLP probes on FROZEN latents.
       single-latent -> s_t ; pair -> (delta gap, v). Reports Pearson r, as upstream."""
    rng = np.random.default_rng(seed)
    S = np.stack([rng.uniform(0.3,22,n), rng.uniform(0,14,n), rng.uniform(-8,8,n)],1)
    U = rng.uniform(-20,20,n)
    N = np.array([W.exact_step(0.,S[i,1],S[i,2],U[i]) for i in range(n)])
    N = np.stack([S[:,0]-N[:,0], N[:,1], N[:,2]],1)
    t = lambda a: torch.tensor(np.asarray(a,np.float32), device=dev)
    Z  = m.encode(t((S-sm)/ss)).cpu().numpy(); Z2 = m.encode(t((N-sm)/ss)).cpu().numpy()
    def ridge(X,Y,lam=1e-3):
        H=np.concatenate([X,np.ones((len(X),1))],1)
        w=np.linalg.solve(H.T@H+lam*np.eye(H.shape[1]),H.T@Y); return H@w
    def rfeat(X,D=512,s=0):
        r=np.random.default_rng(s); W1=r.standard_normal((X.shape[1],D))/np.sqrt(X.shape[1])
        b=r.standard_normal(D)*0.5; return np.maximum(X@W1+b,0)
    pear=lambda a,b: float(np.mean([np.corrcoef(a[:,i],b[:,i])[0,1] for i in range(a.shape[1])]))
    out={}
    out['state_lin'] = pear(ridge(Z,S),S)
    out['state_mlp'] = pear(ridge(rfeat(Z),S),S)
    P = np.concatenate([Z,Z2],1); Ttg = np.stack([N[:,0]-S[:,0], S[:,1]],1)
    out['trans_lin'] = pear(ridge(P,Ttg),Ttg)
    out['trans_mlp'] = pear(ridge(rfeat(P),Ttg),Ttg)
    return out

@torch.no_grad()
def fit_decoder(m, D, dev, K=6, width=1024, n=150000, epochs=40, bs=16384, lr=3e-3, seed=1):
    torch.manual_seed(seed)
    S, U = D['states'], D['actions']
    idx = torch.randperm(len(S), device=dev)[:n]
    z = m.encode(S[idx,0]); Zs=[]; Ys=[]
    for k in range(min(K, S.shape[1]-1)):
        z = m.predict(z, U[idx,k]); Zs.append(z); Ys.append(S[idx,k+1])
    Z=torch.cat(Zs); Y=torch.cat(Ys)
    dec = nn.Sequential(nn.Linear(m.embed_dim,width), nn.GELU(),
                        nn.Linear(width,width), nn.GELU(), nn.Linear(width,3)).to(dev)
    with torch.enable_grad():
        opt=torch.optim.Adam(dec.parameters(), lr=lr)
        for _ in range(epochs):
            pm=torch.randperm(len(Z),device=dev)
            for st in range(0,len(Z),bs):
                b=pm[st:st+bs]; l=((dec(Z[b])-Y[b])**2).mean()
                opt.zero_grad(set_to_none=True); l.backward(); opt.step()
    return dec.eval()

@torch.no_grad()
def taxonomy(m, dec, dev, sm, ss, um, us, n=4000, seed=9):
    """E1..E4 in decoded PHYSICAL units (gauge-invariant) + compounding ratio."""
    rng=np.random.default_rng(seed)
    S=np.stack([rng.uniform(0.3,22,n),rng.uniform(0,14,n),rng.uniform(-8,8,n)],1)
    U=rng.uniform(-20,20,n)[:,None]
    nx=np.array([W.exact_step(0.,S[i,1],S[i,2],U[i,0]) for i in range(n)])
    N=np.stack([S[:,0]-nx[:,0],nx[:,1],nx[:,2]],1)
    t=lambda a: torch.tensor(np.asarray(a,np.float32),device=dev)
    d=lambda q: dec(q).cpu().numpy()*ss+sm
    z=m.encode(t((S-sm)/ss)); z2=m.encode(t((N-sm)/ss)); zh=m.predict(z,t((U-um)/us))
    f=lambda a,b: np.abs(a-b).mean(0).tolist()
    out=dict(E1=f(d(z),S), E2=f(d(z2),N), E3=f(d(zh),N), E4=f(d(z2),d(zh)))
    s=S.copy(); zf=m.encode(t((s-sm)/ss)); tf=fr=0.
    for h in range(10):
        nx2=np.array([W.exact_step(0.,s[i,1],s[i,2],U[i,0]) for i in range(n)])
        s2=np.stack([s[:,0]-nx2[:,0],nx2[:,1],nx2[:,2]],1)
        tf=float(np.abs(d(m.predict(m.encode(t((s-sm)/ss)),t((U-um)/us)))-s2).mean())
        zf=m.predict(zf,t((U-um)/us)); fr=float(np.abs(d(zf)-s2).mean()); s=s2
    out.update(tf10=tf, fr10=fr, compound=fr/max(tf,1e-12))
    return out

@torch.no_grad()
def gauge(m, dec, dev, sm, ss, um, us, trials=3, n=2000, seed=0):
    g=torch.Generator(device=dev).manual_seed(seed); rng=np.random.default_rng(9)
    S=np.stack([rng.uniform(0.3,22,n),rng.uniform(0,14,n),rng.uniform(-8,8,n)],1)
    U=rng.uniform(-20,20,n)[:,None]
    nx=np.array([W.exact_step(0.,S[i,1],S[i,2],U[i,0]) for i in range(n)])
    N=np.stack([S[:,0]-nx[:,0],nx[:,1],nx[:,2]],1)
    t=lambda a: torch.tensor(np.asarray(a,np.float32),device=dev)
    res=[]
    for i in range(trials+1):
        G=torch.eye(m.embed_dim,device=dev) if i==0 else torch.randn(m.embed_dim,m.embed_dim,device=dev,generator=g)
        while abs(torch.det(G).item())<1e-2: G=torch.randn(m.embed_dim,m.embed_dim,device=dev,generator=g)
        Gi=torch.inverse(G)
        z=m.encode(t((S-sm)/ss))@G; zh=m.predict(z@Gi,t((U-um)/us))@G
        e3=float(np.abs(dec(zh@Gi).cpu().numpy()*ss+sm-N).mean())
        Z=(m.encode(t((S-sm)/ss))@G).cpu().numpy()
        sv=np.linalg.svd(Z-Z.mean(0),compute_uv=False); p=sv/(sv.sum()+1e-7)+1e-7
        res.append(dict(E3_decoded=e3, RankMe=float(np.exp(-(p*np.log(p)).sum()))))
    return res
