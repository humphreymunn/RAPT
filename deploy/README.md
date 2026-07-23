# Deploying RAPT

Everything a deployed monitor needs lives in a single checkpoint directory
(produced by `scripts/train.py` / `scripts/calibrate.py`):

| file | used by | contents |
|---|---|---|
| `rapt.onnx` | Python + C++ | model graph: `input [B,input_dim]`, `hidden_in [1,B,256]` → `reconstruction [B,2·obs_dim]` (`[mu\|log_var]`), `hidden_out` |
| `config.json`, `obs_stats.json`, `calibration.json` | Python | architecture/dims, normalization, gate thresholds |
| `deploy_config.csv`, `obs_stats.csv`, `calibration.csv` | C++ | flat twins of the JSON files (no JSON/HDF5 parser needed) |
| `model.pt` | training tools | PyTorch weights (not needed at runtime) |

Both runtimes implement the same loop as the paper's on-robot G1 integration:
normalize the observation, run one recurrent step, compute the per-dimension
`exp(-log_var)·(target-mu)²` score, and evaluate the three gates
(per-dimension, global mean, physical range). Risk > 1.0 ⇒ OOD; wire that to
your safety response (safe stop, controlled fall, recovery). Call `reset()`
at episode boundaries to clear the GRU state.

## Python (onnxruntime, no torch)

```bash
pip install onnxruntime numpy
python deploy/python/replay_example.py <checkpoint> <log>   # offline replay + latency
```

In your control loop:

```python
from rapt_monitor import RaptMonitor          # deploy/python/rapt_monitor.py

monitor = RaptMonitor("checkpoints/robot")    # ~1.6 ms/step on CPU at 50 Hz
monitor.reset()
result = monitor.step(obs, action)            # action only for dynamics models
if result.is_anomaly:
    trigger_safety_response()
```

## C++ (onnxruntime)

```bash
# download an onnxruntime release (the paper used onnxruntime-linux-x64-1.22.0)
cmake -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0 deploy/cpp
cmake --build build
./build/rapt_replay <checkpoint_dir> <log_dir>
```

`rapt_monitor.hpp` is a single self-contained header — drop it into your
stack and call `step()` from the policy thread, exactly like the replay
`main.cpp` does.

## Calibrating for your deployment

Thresholds must be calibrated where the monitor will run. Two options, as in
the paper:

- **Simulation calibration**: calibrate on a large nominal sim batch
  (`scripts/train.py` does this automatically on the validation split). This
  also exposes residual sim-to-real mismatch — useful as a verification tool.
- **Real-world calibration** (recommended for deployment): record a brief
  verified nominal run on the physical system — the paper used **3×1-minute
  nominal walks** — then

  ```bash
  python scripts/calibrate.py checkpoints/robot my_nominal_logs/
  ```

  This absorbs static deployment offsets (sensor noise, latency, contact and
  hardware dynamics) into the thresholds. Rare nominal spikes seen during
  calibration are treated as nominal by construction (max-plus-margin gates).
  Re-run it after any change to the robot, sensors, or environment.

Gate margins are configurable if your false-positive budget differs:
`--k-dim` (per-dimension margin, default 2), `--k-global` (global margin,
default 3), `--range-buffer` (range slack, default 1.0 = 100% of the
calibration span). Increase them if you see false positives on long nominal
runs; decrease for earlier detection.

## Unitree G1 / unitree_rl_lab integration

The full on-robot FSM integration from the paper (policy thread + monitor +
CSV logging + BPTT saliency on anomaly) lives in the `unitree_rl_lab`
deployment stack. `reference/anomaly_detector_fdm.h` is the original on-robot
detector header for comparison — it reads the same `rapt.onnx` plus
`obs_stats.h5`/`calibration_dataset_loss.csv` (this repo's
`calibration_losses.csv` is the same format) and computes identical
thresholds at startup. In that stack the detector is constructed in
`State_RLBase.cpp` with a `rapt_dir` from the robot's `config.yaml` and
`checkTransition(prev_obs, prev_action, curr_obs, t)` is called once per
policy step.
