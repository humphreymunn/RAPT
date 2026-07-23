---
license: bsd-3-clause
task_categories:
- time-series-forecasting
tags:
- anomaly-detection
- out-of-distribution
- robotics
- humanoid
- sim-to-real
- unitree-g1
- isaac-lab
pretty_name: RAPT G1 OOD Benchmark
---

# RAPT G1 OOD Benchmark

Simulation datasets for out-of-distribution (OOD) detection on a Unitree G1
humanoid, released with **"RAPT: Model-Predictive Out-of-Distribution
Detection and Failure Diagnosis for Sim-to-Real Humanoid Deployment"**
([arXiv:2602.01515](https://arxiv.org/abs/2602.01515),
[code](https://github.com/humphreymunn/RAPT),
[project page](https://humphreymunn.github.io/RAPT/)).

Each task directory contains proprioceptive trajectories collected from an
expert RL policy in NVIDIA Isaac Lab (50 Hz):

- **`train.npz`** — nominal-only episodes for training self-supervised
  detectors (~1.2M steps, matching the paper's training scale).
- **`train_small.npz`** — a 4x smaller nominal training set (~300k steps),
  for studying detection under limited nominal data (see below).
- **`calibration.npz`** — a dedicated nominal batch collected under the
  evaluation protocol, for calibrating detector operating points (the paper's
  "brief nominal calibration episode precedes evaluation").
- **`test.npz`** — labeled evaluation episodes (1000 steps velocity / 1500
  mimic): for every OOD category, half the parallel environments are
  perturbed from onset step 50 and half run unperturbed as nominal controls.
- **`metadata.json`** — task, policy, dimensions, timestep, onset, category
  list, split sizes.

## Tasks

| directory | task | obs dim | act dim | train (small) | calibration eps | test seqs (OOD / nominal) | size |
|---|---|---|---|---|---|---|---|
| `g1_velocity` | omni-directional velocity tracking | 96 | 29 | ~6.2 h (~1.5 h) | 225 | 432 / 401 | 491 MB |
| `g1_mimic_dance102` | motion mimicry (Dance-102) | 154 | 29 | ~6.8 h (~1.7 h) | 437 | 521 / 523 | 1.25 GB |
| `g1_mimic_gangnam` | motion mimicry (Gangnam Style) | 154 | 29 | ~6.8 h (~1.7 h) | 407 | 268 / 837 | 1.28 GB |

Experts: the paper's GCR-PPO velocity policy and the deployed mimic policies.
(The paper's fourth task, ballistic throwing, requires a policy checkpoint
that is not part of this release yet.)

## Evaluating

Two evaluators ship with the [code release](https://github.com/humphreymunn/RAPT):
`scripts/evaluate_paper.py` replicates the paper's simulation protocol
(RAPT model gates only, thresholds normalized on `calibration.npz`, Safety
Score = TPR at 0.5% FPR from the episode-level ROC, reported pooled and
per-category), and `scripts/benchmark.py` provides a deployment-style
evaluation (hybrid gates + quantile-calibrated operating point with
TPR/FPR/PADD). Note the dataset scale (single seed, 128 parallel envs)
is smaller than the paper's evaluation (4096 envs, 5 seeds, plus the
throwing task).

## Nominal-data efficiency: `train.npz` vs `train_small.npz`

In simulation, nominal data is nearly free — on real systems it is scarce
and expensive to verify. The two training splits let you study that
tradeoff directly: identical test and calibration splits, only the amount
of nominal training data changes. RAPT reference results (single seed,
paper protocol, `scripts/evaluate_paper.py`):

| task | train split | AUROC (pooled) | Safety Score (TPR@0.5% FPR, pooled) |
|---|---|---|---|
| `g1_velocity` | small (~300k) | 0.891 | 0.565 |
| `g1_velocity` | full (~1.2M) | **0.911** | **0.704** |
| `g1_mimic_dance102` | small (~300k) | 0.887 | 0.605 |
| `g1_mimic_dance102` | full (~1.2M) | **0.910** | **0.695** |
| `g1_mimic_gangnam` | small (~300k) | 0.900 | 0.612 |
| `g1_mimic_gangnam` | full (~1.2M) | **0.909** | **0.631** |

**Protocol note.** Reported with `scripts/evaluate_paper.py` defaults: RAPT
model gates only (no range detector), thresholds normalized on
`calibration.npz` with the paper's calibration (per-dimension
``max + 5*sigma``; global ``mean-of-maxes + 3*sigma``), Safety Score =
TPR at 0.5% FPR read from the episode-level ROC step function. Two
practical findings are baked into this protocol: (i) the calibration
formula barely matters (<0.01 across variants) because the per-dimension
gate dominates the combined risk score — consistent with the paper's
ablation that per-dimension max aggregation is the key detection choice;
(ii) TPR at 0.5% FPR is an order statistic of the nominal pool, so
`g1_mimic_gangnam`'s test split ships 837 nominal controls (extended by
+544 sequences, seed 777) — with only ~300 controls the threshold was set
by one or two extreme near-fall nominal episodes and the metric became
unstable. For context, the paper's larger evaluation (4096 envs, 5 seeds)
reports Safety Scores of 0.74 / 0.67 / 0.75 on these tasks.

## Real-robot runs (`g1_velocity_real`)

Proprioceptive logs from the paper's physical Unitree G1 deployment of the
velocity policy (50 Hz, same 96 named dimensions; `actions.csv` values are
raw policy outputs):

- **`calibration.npz`** — the nominal real-world calibration run used by the
  paper's evaluation (~2.3 min).
- **`test.npz`** — 50 labeled runs (~76 min): 11 nominal walks (1.6–10.6 min)
  and 39 anomalous runs across 8 induced fault categories
  (`action_scaling`, `initial_state`, `policy_latency`, `motor_dynamics`,
  `motor_failure`, `observation_ordering`, `sensor_noise`,
  `footwear_contact`), with per-run names. There is no train split —
  detectors are trained in simulation (`g1_velocity/train.npz`) and
  calibrated on the real calibration run, mirroring the paper's
  sim-to-real protocol.

Caveats: fault onset times are unknown (`onset = -1`); several anomalous
logs are only 1–22 steps long because the fault destabilized the robot
immediately (the truncated log is itself the anomaly signature); and the
paper's push, payload, collision/obstruction, and deformable-terrain runs
are not included in this release (N=50 here vs 78 in the paper).

## OOD categories (test split)

Observation-level: `sensor_drift`, `sensor_zero`, `scale_half`,
`scale_double`, `obs_swap`, `action_swap`, `noise`, `latency_offset`,
`latency_slow`, `frozen_sensor`. Physics-level (simulated in Isaac Lab):
`actuator_dynamics`, `init_state`, `env_disturbance` (pushes / payload),
`env_friction`. Sampling ranges follow the paper (Supplementary,
"Simulation OOD categories"). Sequences end early if the robot falls
(safety termination), so lengths vary.

## Format

Ragged NumPy archives (`float16`). Keys per sequence `i`:
`seq_%05d` `[T_i, obs_dim]`, `act_%05d` `[T_i, action_dim]`; plus
`dim_names` (named observation dimensions), and in `test.npz`:
`labels` (0 nominal / 1 anomalous), `onset` (injection step, -1 for
nominal), `fault` (category name).

```python
from huggingface_hub import snapshot_download
path = snapshot_download("hmunn/rapt-g1-ood", repo_type="dataset")

# with the RAPT release (https://github.com/humphreymunn/RAPT):
from rapt import load_sequences
train = load_sequences(f"{path}/g1_velocity/train.npz")   # nominal only
test = load_sequences(f"{path}/g1_velocity/test.npz")     # labeled

# or with plain numpy:
import numpy as np
data = np.load(f"{path}/g1_velocity/test.npz")
obs0 = data["seq_00000"].astype("float32")
```

Train/evaluate RAPT end-to-end (metrics saved as JSON):

```bash
python scripts/benchmark.py <path>/g1_velocity <path>/g1_mimic_* --out results
```

## Citation

```bibtex
@article{munn2026rapt,
  title   = {RAPT: Model-Predictive Out-of-Distribution Detection and Failure
             Diagnosis for Sim-to-Real Humanoid Deployment},
  author  = {Munn, Humphrey and Tidd, Brendan and B{\"o}hm, Peter and
             Gallagher, Marcus and Howard, David},
  journal = {arXiv preprint arXiv:2602.01515},
  year    = {2026}
}
```
