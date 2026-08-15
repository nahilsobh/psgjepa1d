# Pushing this repo

The repo is already initialised with a commit on `main`. I cannot push from my environment
(no credentials, and GitHub blocks automated access), so run these two commands:

```bash
cd psgjepa1d
git remote add origin git@github.com:<YOU>/psgjepa1d.git    # or https://github.com/<YOU>/psgjepa1d.git
git push -u origin main
```

## First, set the author on the commit

The commit was created with a placeholder identity. Fix it before pushing:

```bash
git config user.name  "Your Name"
git config user.email "you@example.com"
git commit --amend --reset-author --no-edit
```

## Before making it public — please read NOTICE

This is **derivative work** of https://github.com/Haodong-Yan/PSG-JEPA.
`psgjepa1d/grounding.py` is a close port of their `psgjepa/grounding.py`.

1. Check the upstream LICENSE and comply with it.
2. The MIT `LICENSE` here covers only the newly written 1-D code. If upstream terms require
   different terms for derivatives, replace or supplement it.
3. Keep the citation in `README.md` and `CITATION.cff`.

## CI

`.github/workflows/ci.yml` runs on push:
- `tests/test_world.py`      — physics + data integrity (NumPy only)
- `tests/test_grounding.py`  — grounding shapes, ablation arms, horizon-weighting regression
- a tiny smoke train

CI installs CPU torch, so it will not exercise the GPU path.

## Smoke test before any long run

```bash
python train.py loss.grounding.weight=0.0 data.n_windows=50000 trainer.max_epochs=2
```

This exercises the default `sigreg` regulariser, autograd, and the data pipeline in ~1 minute.
Neither has been executed in the authoring environment (torch was not installable there) — see
the verification table in README.md for exactly what was and was not checked.
