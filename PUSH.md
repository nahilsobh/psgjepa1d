# Pushing this repo

I cannot push from my environment (no credentials, and GitHub blocks automated access).
Run these on your side.

## New repo

```bash
unzip psgjepa1d.zip && cd psgjepa1d
git init -b main
git add -A
git commit -m "PSG-JEPA-1D: physical state grounding on an exactly-solvable 1-D car world

Parallel implementation of PSG-JEPA (Yan et al., arXiv:2608.06799).
- psgjepa1d/grounding.py is a faithful port (per-horizon weighting, velocity head)
- the 1-D world has an exactly computable optimum (0.0129 m), so grounding can be
  measured against an absolute reference rather than relative planning success
- adds E1-E4 error localisation, compounding ratio, and a GL(d) gauge test"
git remote add origin git@github.com:<YOU>/psgjepa1d.git   # or https://...
git push -u origin main
```

## Attribution — please keep

This is a derivative of https://github.com/Haodong-Yan/PSG-JEPA. Before making it public:

1. Check that repo's LICENSE and comply with it. The `grounding.py` port follows its structure
   closely and is derivative work — the MIT file here covers only the new 1-D code, and should be
   replaced or supplemented if their terms require it.
2. Keep the citation in README.md.
3. `git log` will show me as neither author nor committer — the commit is yours.

## Also worth pushing

`jepa1d_gpu_package.zip` (the cluster experiment runner) is a separate, self-contained project.
I'd keep it in its own repo rather than mixing it with this one.
