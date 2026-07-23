# Reproducing the paper's results

The paper evaluates on four Unitree G1 tasks in NVIDIA Isaac Lab (velocity
locomotion, Dance-102 mimicry, Gangnam mimicry, ballistic throwing; five
GCR-PPO expert policies per task) plus 78 real-robot runs. This directory
contains everything code-side; the simulation experiments additionally need
an Isaac Lab + [`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab)
environment with trained expert policies.

## Layout

- **`baselines/`** — standalone reimplementations of the baselines with the
  paper's hyperparameters (LSTM-VAE h256/z24 + SVR, Deep SVDD 128/32,
  Isolation Forest 100 trees), runnable on the common dataset format:

  ```bash
  python reproduce/run_baselines.py train.npz eval.npz --rapt checkpoints/my_rapt
  ```

- **`ood_injection.py`** — the paper's observation-space OOD categories
  (drift, dropout, scaling, swap, noise, latency, slow update, frozen sensor)
  with the paper's sampling ranges, for building labeled eval sets from any
  nominal data.

- **`isaaclab/collect_rapt_datasets.py`** — builds the released HuggingFace
  datasets ([`hmunn/rapt-g1-ood`](https://huggingface.co/datasets/hmunn/rapt-g1-ood)):
  nominal `train.npz` + labeled per-category OOD `test.npz` in the common
  sequence format, using either an rsl_rl checkpoint or an exported
  `policy.onnx` as the expert. `ood_injection_lib.py` is the shared
  injector (all 14 categories, observation- and physics-level).

- **`isaaclab/`** — the original simulation scripts, vendored verbatim. They
  run *inside* a `unitree_rl_lab` checkout (copy them to `scripts/rsl_rl/`
  there), not standalone:
  - `collect_and_train_rapt.py` — roll out an expert policy (`--num_envs`,
    `--collection_time 600`), collect `[T,N,D]` obs/action tensors, train
    RAPT (AdamW 1e-3, OneCycleLR, 100 epochs), export calibration losses.
  - `play_with_rapt.py` — the 4096-env evaluation with all 14 in-sim OOD
    injection categories (including the physics-level ones: actuator
    dynamics, pushes, 10 kg payload, friction, initial state) → AUROC +
    Safety Score (TPR @ 0.5% episode FPR).
  - `collect_and_train_lstmvae.py`, `collect_and_train_patchad.py`,
    `forest_and_range_experiment.py`, `svdd_and_range_experiment.py` —
    baseline training/evaluation drivers.
  - `export_to_onnx.py` — the original deployment export.
  - `sweeps/` — the Taguchi L12 ablation (48 models) and FDM sweep runners.
  - `analysis/` — ablation collation + Random-Forest feature importance.

- **`realworld/`** — the original offline evaluation scripts for the 78
  real-robot logs (`test_real_world_rapt.py` and the per-baseline variants).
  They consume the deployment logger's CSV format (`observations.csv`,
  `actions.csv` at 50 Hz per run directory).

## PatchAD

PatchAD is evaluated with the [official implementation](https://github.com/EmorZz1G/PatchAD)
(not vendored here). Paper configuration: window 105 (30 for throwing),
`d_model` 40, 3 encoder layers, patch sizes {3,5}, temporal stride 10,
trained 3 epochs. `isaaclab/collect_and_train_patchad.py` and
`realworld/test_real_world_patchad.py` show the exact wiring; place the
official `patchad` package next to them.

## Headline numbers to expect

- Simulation (5 seeds, 4096 envs/task): RAPT AUROC **0.92**, Safety Score
  0.67–0.75 per task, ~**1.6 ms**/step; strongest baseline (LSTM-VAE) AUROC
  0.77, Safety Score 0.30–0.44.
- Real world (78 runs): RAPT+Range **88.9% TPR / 6.7% FPR**
  (56 TP / 7 FN / 14 TN / 1 FP).
- LLM diagnosis (16 failure logs): top-1 **75%** proprioception-only,
  **87.5%** with a visual keyframe; top-3 87.5% / 100%.
