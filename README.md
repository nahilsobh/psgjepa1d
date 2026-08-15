# PSG-JEPA-1D

Parallel implementation of **PSG-JEPA** (Yan et al., arXiv:2608.06799) for the 1-D car world.
Mirrors the upstream repo structure so the two can be compared component by component.

| upstream | here |
|---|---|
| `psgjepa/grounding.py` | `psgjepa1d/grounding.py` — **faithful port**, line-for-line |
| `psgjepa/model.py` | `psgjepa1d/model.py` — MLP encoder/predictor instead of ViT |
| `configs/psgjepa.yaml` | `configs/psgjepa1d.yaml` — same keys, same `lambda_g = 0.1` |
| `train.py` | `train.py` — same override syntax |
| `eval/` | `eval/evaluate.py` — probes + control + **exact-world diagnostics** |
| (video dataset) | `psgjepa1d/data.py` — windows **synthesised** from exact dynamics |

## Why a 1-D parallel?

The upstream world has no ground truth: you cannot compute the optimal action, so grounding can
only be judged by *relative* planning success. Here the optimum is **exactly computable**
(`world.optimal_stop`, mean resting gap **0.0129 m**), so the same grounding objective can be
measured against an absolute reference, and error can be localised to a specific edge of the
encoder/predictor/decoder diagram.

## Variable mapping

| robot arm | 1-D car | role |
|---|---|---|
| proprioceptive state `s_t` | `[gap, v, a]` | static grounding target |
| joint angles `q_t` | `[gap]` | position-like dof; multi-horizon `Δq` target |
| joint velocity | `[v]` | optional velocity-head target |

`Δq` is the change of the **position-like** coordinate; its derivative is supervised separately by
the velocity head. In the car, `gap` plays the role of `q` and `v` of `q̇`.

## Quick start

```bash
pip install torch numpy pyyaml

python train.py                                    # full PSG-JEPA-1D
python train.py loss.grounding.weight=0.0          # LeWM-1D baseline (no grounding)
python train.py loss.grounding.use_velocity=false  # drop the velocity head
bash scripts/run_ablation.sh                       # upstream Table-5 ablation
```

Data is synthesised on first run (2M windows ≈ 2 min) and cached to `window_cache.npz`.

## Two upstream details that are easy to get wrong

Both were wrong in an earlier ad-hoc reimplementation of ours, and both are preserved here:

1. **Horizons are weighted equally, not pairs.** Short horizons yield more pairs, so a per-pair
   mean lets `k=1` dominate. Upstream: `torch.stack(per_horizon_means).mean()`.
2. **The velocity head is ON by default** (`use_velocity: true` in the upstream config). It is a
   third grounding term, not mentioned prominently in the paper text.

## What to measure

`eval/evaluate.py` provides three levels:

1. **Identifiability** — frozen linear-ridge and MLP probes, Pearson *r* (upstream Tables 1–2).
2. **Control** — closed-loop stopping vs the exact optimum. Replaces upstream's relative planning
   success with an absolute number.
3. **Exact-world diagnostics** — the E1/E2/E3/E4 error mapping, the compounding ratio
   (free-run ÷ teacher-forced), and the GL(*d*) gauge test.

The gauge test matters for interpreting upstream Table 1: probe *r* on frozen latents is a
latent-space quantity. Linear ridge probes are invariant under invertible reparametrisation;
MLP probes are not obviously so. Decoded quantities (E1–E4) are invariant by construction.

## Status

Verified without torch: window synthesis matches the exact dynamics to **machine zero**, physics
scales are correct, jerk sequences cover both holds and changes. The torch training path has
**not** been executed here (no GPU/torch in the authoring environment) — run the LeWM-1D baseline
first as a smoke test.

## Verification status

Verified in the authoring environment (**no torch available**, so a minimal NumPy shim was used to
check shapes and control flow; numerics/autograd are **not** verified):

| check | result |
|---|---|
| window synthesis vs exact dynamics | **0.0** (machine zero) |
| physics scales (L0, V0, A, J) | correct |
| jerk sequences cover holds and changes | 65% held / 35% changed |
| encoder / predictor output shapes | (B,T,D) / (B,T−1,D) — OK |
| grounding: all 4 loss terms | finite, correct term sets |
| grounding: `use_velocity=false` path | velocity term correctly absent |
| all four Table-5 ablation arms | finite, correct term sets |
| config override parsing | types correct (float/bool/list/int) |
| `stopgrad` regulariser | OK |
| `sigreg`/`vicreg`/`barlow`/`visreg` | **not executed** (shim lacks the ops) |
| full training loop, autograd, GPU | **not executed** |

**Two bugs were found and fixed during this check:**

1. **`sigreg` OOM.** At the default config (batch 4096, T=7, num_proj=1024, knots=17) the
   intermediate `ang` tensor is 28672×1024×17 ≈ **2 GB**, ~4 GB with cos+sin — an OOM on most
   GPUs. All sliced regularisers now subsample rows (`_subsample`, default 4096), which is an
   unbiased estimate of the same population statistic. Cost drops to 0.29 GB.
2. **Ablation arms produced NaN.** Dropping a grounding term by passing an empty index list gave
   `state_dim=0`, and MSE over an empty tensor is NaN. Replaced with explicit
   `use_static` / `use_transition` flags.

**Run the LeWM-1D baseline first** (`python train.py loss.grounding.weight=0.0
data.n_windows=50000 trainer.max_epochs=2`) as a smoke test before any long run.
