import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import h5py
from collections import deque
import time
from typing import Optional, Tuple

# --- SETUP PATHS (Matches your training environment) ---
sys.path.append(os.getcwd()) 
sys.path.append(os.path.join(os.getcwd(), 'scripts', 'rsl_rl', 'patchad'))

try:
    from scripts.rsl_rl.patchad.patchad_model.models import PatchMLPAD
except ImportError:
    print("[ERROR] PatchMLPAD not found. Ensure scripts/rsl_rl/patchad/ is in your path.")
    sys.exit(1)
CAL_BASE = 'deploy/robots/g1_29dof/build/logs/cal_real____1'


NOMINAL_TESTS = [
    "deploy/robots/g1_29dof/build/logs/nominal_2m_1",
    "deploy/robots/g1_29dof/build/logs/nominal_2m_2",
    "deploy/robots/g1_29dof/build/logs/nominal_2m_1_day2",
    "deploy/robots/g1_29dof/build/logs/nominal_2m_2_day2",
    "deploy/robots/g1_29dof/build/logs/nominal_4m_1",
    "deploy/robots/g1_29dof/build/logs/nominal_4m_2",
    "deploy/robots/g1_29dof/build/logs/nominal_4m_1_day2",
    "deploy/robots/g1_29dof/build/logs/nominal_4m_2_day2",
    "deploy/robots/g1_29dof/build/logs/nominal_10m_1",
    "deploy/robots/g1_29dof/build/logs/nominal_10m_2",
    "deploy/robots/g1_29dof/build/logs/nominal_10m_1_day2",
    "deploy/robots/g1_29dof/build/logs/nominal_10m_2_day2",
    #"deploy/robots/g1_29dof/build/old_logs/normal1",
    #"deploy/robots/g1_29dof/build/old_logs/normal2",
    #"deploy/robots/g1_29dof/build/old_logs/normal3"
]

ANOMALY_TESTS = [
    "deploy/robots/g1_29dof/build/logs/actionscale1",
    "deploy/robots/g1_29dof/build/logs/actionscale2",
    "deploy/robots/g1_29dof/build/logs/actionscale3",
    "deploy/robots/g1_29dof/build/logs/actionscale4",
    "deploy/robots/g1_29dof/build/logs/actionscale5",
    "deploy/robots/g1_29dof/build/logs/deformability_1",
    "deploy/robots/g1_29dof/build/logs/initpos1",
    "deploy/robots/g1_29dof/build/logs/initpos2",
    "deploy/robots/g1_29dof/build/logs/initpos3",
    "deploy/robots/g1_29dof/build/logs/initpos4",
    "deploy/robots/g1_29dof/build/logs/initpos5",
    "deploy/robots/g1_29dof/build/logs/latency1",
    "deploy/robots/g1_29dof/build/logs/latency2",
    "deploy/robots/g1_29dof/build/logs/latency3",
    "deploy/robots/g1_29dof/build/logs/latency4",
    "deploy/robots/g1_29dof/build/logs/latency5",
    "deploy/robots/g1_29dof/build/logs/motordynamics1",
    "deploy/robots/g1_29dof/build/logs/motor_dynamics2",
    "deploy/robots/g1_29dof/build/logs/motordynamics3",
    "deploy/robots/g1_29dof/build/logs/motordynamics4",
    "deploy/robots/g1_29dof/build/logs/motordynamics5",
    "deploy/robots/g1_29dof/build/logs/motor_failure1",
    "deploy/robots/g1_29dof/build/logs/motorfailure_2",
    "deploy/robots/g1_29dof/build/logs/motorfailure3",
    "deploy/robots/g1_29dof/build/logs/motorfailure4",
    "deploy/robots/g1_29dof/build/logs/motorfailure5",
    "deploy/robots/g1_29dof/build/logs/obsswap1",
    "deploy/robots/g1_29dof/build/logs/obsswap2",
    "deploy/robots/g1_29dof/build/logs/obsswap3",
    "deploy/robots/g1_29dof/build/logs/obsswap4",
    "deploy/robots/g1_29dof/build/logs/obsswap5",
    "deploy/robots/g1_29dof/build/logs/sensornoise1",
    "deploy/robots/g1_29dof/build/logs/sensornoise2",
    "deploy/robots/g1_29dof/build/logs/sensornoise3",
    "deploy/robots/g1_29dof/build/logs/sensornoise4",
    "deploy/robots/g1_29dof/build/logs/sensornoise5",
    "deploy/robots/g1_29dof/build/logs/shoe1",
    "deploy/robots/g1_29dof/build/logs/shoe2",
    "deploy/robots/g1_29dof/build/logs/shoe3",
    "deploy/robots/g1_29dof/build/logs/shoes4",
    "deploy/robots/g1_29dof/build/logs/shoes5",
    #"deploy/robots/g1_29dof/build/logs/attressrealtest2", 
    #"deploy/robots/g1_29dof/build/logs/mattresreal3",
    #"deploy/robots/g1_29dof/build/logs/psuhtestreal3", "deploy/robots/g1_29dof/build/logs/pushtestreal2",
    #"deploy/robots/g1_29dof/build/logs/testrealmattress1", "deploy/robots/g1_29dof/build/logs/ushtest1realv2",
    #"deploy/robots/g1_29dof/build/old_logs_newer/payload3", 
    #"deploy/robots/g1_29dof/build/old_logs_newer/push2", "deploy/robots/g1_29dof/build/old_logs_newer/push5",
    #"deploy/robots/g1_29dof/build/old_logs_newer/ush1", "deploy/robots/g1_29dof/build/old_logs_newer/ush3",
    #"deploy/robots/g1_29dof/build/old_logs/ayload1", #"deploy/robots/g1_29dof/build/old_logs/obstacle1",
    #"deploy/robots/g1_29dof/build/old_logs/obstacle2", "deploy/robots/g1_29dof/build/old_logs/obstacle3",
   # "deploy/robots/g1_29dof/build/old_logs/payload2", 
   # "deploy/robots/g1_29dof/build/old_logs/push1", "deploy/robots/g1_29dof/build/old_logs/pusharm",
   # "deploy/robots/g1_29dof/build/old_logs/pushknees", "deploy/robots/g1_29dof/build/old_logs/shoe2",
   # "deploy/robots/g1_29dof/build/old_logs/shoe3", "deploy/robots/g1_29dof/build/old_logs/shoes1",
    #"deploy/robots/g1_29dof/build/old_logs/mattress1",
   # "deploy/robots/g1_29dof/build/old_logs/mattress2"
]

ALL_EXPERIMENTS = NOMINAL_TESTS + ANOMALY_TESTS

CAL_BASE = 'deploy/robots/g1_29dof/build/logs/cali_real_new_1'

CAL_LOSS_FILE = f'{CAL_BASE}/loss.csv'
CAL_OBS_FILE  = f'{CAL_BASE}/observations.csv'
CAL_ACT_FILE  = f'{CAL_BASE}/actions.csv'

# PatchAD Model Paths
MODEL_BASE = "trained_patchad_models/Unitree-G1-29dof-Velocity_20260328_195507"
CHECKPOINT = f"{MODEL_BASE}/PSM_checkpoint.pth"
STATS_PATH = f"{MODEL_BASE}/obs_stats.h5"

# Architecture (Must match collect_and_train_patchad.py)
WIN_SIZE = 105
D_MODEL = 40
E_LAYER = 3
PATCH_SIZES = [3, 5]

# ==================================================================================
# 1. MODEL WRAPPER (Fixes einops dimension mismatch from your training script)
# ==================================================================================
class BatchExpandingPatchMLPAD(PatchMLPAD):
    def forward(self, x, *args, **kwargs):
        ret = super().forward(x, *args, **kwargs)
        batch_size = x.shape[0]
        
        def expand_list(t_list):
            new_list = []
            for t in t_list:
                if isinstance(t, torch.Tensor) and t.dim() == 2:
                    new_list.append(t.unsqueeze(0).expand(batch_size, -1, -1))
                else:
                    new_list.append(t)
            return new_list

        dist_num, dist_size, mx_num, mx_size, rec_x = ret
        p_num_mx_list = expand_list(mx_num)[:len(dist_size)]
        p_size_mx_list = expand_list(mx_size)[:len(dist_num)]
        return (dist_num, dist_size, p_num_mx_list, p_size_mx_list, rec_x)

class PatchADDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.win_size = WIN_SIZE
        self.batch_size = int(os.getenv("PATCHAD_BATCH_SIZE", "1024"))

        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Load Standardization Stats (Mean/Std)
        with h5py.File(STATS_PATH, 'r') as f:
            self.obs_mean = torch.from_numpy(f['mean'][:]).float().to(device)
            self.obs_std = torch.from_numpy(f['std'][:]).float().to(device)
        
        obs_dim = self.obs_mean.shape[0]
        self.model = BatchExpandingPatchMLPAD(
            win_size=WIN_SIZE, d_model=D_MODEL, e_layer=E_LAYER,
            patch_sizes=PATCH_SIZES, dropout=0.0, activation="relu",
            channel=obs_dim, norm='n', output_attention=True
        ).to(device)
        
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

    def _compute_batch_scores(self, batch: torch.Tensor) -> torch.Tensor:
        autocast_enabled = self.device.startswith('cuda') and torch.cuda.is_available()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
            results = self.model(batch)
            p_num_dist, p_size_dist, rec_x = results[0], results[1], results[4]

            rec_x = rec_x.float()
            batch_f32 = batch.float()

            rec_loss = torch.mean((batch_f32 - rec_x) ** 2, dim=[1, 2])

            kl_loss = torch.zeros(batch.size(0), device=batch.device)
            min_len = min(len(p_num_dist), len(p_size_dist))

            for j in range(min_len):
                pn = p_num_dist[j].float()
                ps = p_size_dist[j].float()

                if pn.shape[1] != ps.shape[1]:
                    if pn.shape[1] == ps.shape[2] and pn.shape[2] == ps.shape[1]:
                        ps = ps.transpose(1, 2)
                    elif ps.shape[1] == pn.shape[2] and ps.shape[2] == pn.shape[1]:
                        pn = pn.transpose(1, 2)
                    else:
                        continue

                kl_loss += torch.mean((pn - ps) ** 2, dim=[1, 2])

        return rec_loss + kl_loss

    @torch.inference_mode()
    def process_sequence(self, obs_tensor):
        """Processes entire sequence using vectorized sliding windows and dual-loss."""
        # Standardize using Mean/Std from training
        obs_norm = (obs_tensor - self.obs_mean) / (self.obs_std + 1e-6)
        
        # Efficient sliding window creation
        windows = obs_norm.unfold(0, self.win_size, 1).transpose(1, 2)
        
        all_scores = []

        for i in range(0, windows.size(0), self.batch_size):
            batch = windows[i : i + self.batch_size].contiguous()
            all_scores.append(self._compute_batch_scores(batch).detach())
        
        scores = torch.cat(all_scores)
        pad = torch.zeros(self.win_size - 1, device=self.device)
        return torch.cat([pad, scores])

    @torch.inference_mode()
    def find_first_anomaly(self, obs_tensor: torch.Tensor, threshold: float) -> Tuple[Optional[int], Optional[float]]:
        """Find first PatchAD anomaly without scoring the whole sequence."""
        obs_norm = (obs_tensor - self.obs_mean) / (self.obs_std + 1e-6)
        windows = obs_norm.unfold(0, self.win_size, 1).transpose(1, 2)

        total_windows = windows.size(0)
        for start in range(0, total_windows, self.batch_size):
            batch = windows[start : start + self.batch_size].contiguous()
            batch_scores = self._compute_batch_scores(batch)
            hit_mask = batch_scores > threshold
            if hit_mask.any():
                first_in_batch = int(torch.argmax(hit_mask.int()).item())
                first_window_idx = start + first_in_batch
                first_time_idx = first_window_idx + self.win_size - 1
                return first_time_idx, float(batch_scores[first_in_batch].item())

        return None, None

def get_observation_names():
    joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ]
    obs_names = []
    obs_names.extend(["root_vel_x", "root_vel_y", "root_vel_z", "gravity_x", "gravity_y", "gravity_z", "cmd_vel_x", "cmd_vel_y", "cmd_vel_yaw"])
    obs_names.extend(["pos_" + n for n in joint_names] + ["vel_" + n for n in joint_names] + ["action_" + n for n in joint_names])
    return obs_names

def load_and_validate(path):
    try: return pd.read_csv(path)
    except FileNotFoundError: sys.exit(f"Error: File not found {path}")


def read_observations_csv(path, obs_dims=None):
    if obs_dims is None:
        usecols = lambda c: c == 'timestamp' or c.startswith('dim_')
    else:
        usecols = ['timestamp', *obs_dims]
    return pd.read_csv(path, usecols=usecols)

def resample_to_50hz(df, obs_dims):
    """
    Resample CSV data to 50Hz (0.02s intervals) using linear interpolation.
    
    Args:
        df: DataFrame with 'timestamp' column and observation columns
        obs_dims: List of observation column names
        
    Returns:
        Resampled numpy array of observations
    """
    timestamps = df['timestamp'].to_numpy(dtype=np.float64, copy=False)
    obs_data = df[obs_dims].to_numpy(dtype=np.float32, copy=False)

    # Ensure monotonic timestamps and remove duplicates for stable interpolation
    if np.any(np.diff(timestamps) <= 0):
        uniq_ts, uniq_idx = np.unique(timestamps, return_index=True)
        timestamps = uniq_ts
        obs_data = obs_data[uniq_idx]
    
    # Create uniform 50Hz timeline
    start_time = timestamps[0]
    end_time = timestamps[-1]
    dt = 0.02  # 50Hz = 0.02s per step
    
    target_timestamps = np.arange(start_time, end_time, dt)
    
    print(f"  Original: {len(timestamps)} samples over {end_time - start_time:.2f}s")
    print(f"  Resampled: {len(target_timestamps)} samples at 50Hz")
    
    # Vectorized interpolation across all dimensions
    right_idx = np.searchsorted(timestamps, target_timestamps, side='left')
    right_idx = np.clip(right_idx, 1, len(timestamps) - 1)
    left_idx = right_idx - 1

    t_left = timestamps[left_idx]
    t_right = timestamps[right_idx]
    denom = np.maximum(t_right - t_left, 1e-9)
    w = ((target_timestamps - t_left) / denom).astype(np.float32)

    left_vals = obs_data[left_idx]
    right_vals = obs_data[right_idx]
    resampled_obs = left_vals + (right_vals - left_vals) * w[:, None]
    
    return resampled_obs, target_timestamps

def run_patchad_batch_testing():
    # ----------------------------------------------------------------------
    # STEP 1: ONE-TIME GLOBAL CALIBRATION
    # ----------------------------------------------------------------------
    print(f"\n{'='*80}\n[PHASE 1] PatchAD GLOBAL CALIBRATION\n{'='*80}")
    print(f"Loading nominal baseline from {CAL_BASE}...")
    
    # Load raw calibration data
    df_cal_raw = read_observations_csv(f'{CAL_BASE}/observations.csv')
    obs_dims = [c for c in df_cal_raw.columns if c.startswith('dim_')]
    
    # Resample to 50Hz (Using your specific resampling helper)
    print("Resampling calibration data to 50Hz...")
    cal_obs_resampled, _ = resample_to_50hz(df_cal_raw, obs_dims)
    
    # Initialize Detector
    detector = PatchADDetector() 
    
    # 1. Calculate PatchAD Threshold
    print(f"Processing sequence windows for calibration...")
    cal_tensor = torch.as_tensor(cal_obs_resampled, dtype=torch.float32, device='cuda')
    
    with torch.no_grad():
        cal_scores = detector.process_sequence(cal_tensor).cpu().numpy()
    
    # Threshold: Max + 1 Std
    #patch_threshold = np.max(cal_scores) + (0.0 * np.std(cal_scores))
    # use 99.73 percentile (3 sigma) to set threshold for better anomaly detection
    patch_threshold = float(np.percentile(cal_scores, 99.73))
    # 2. Calculate Range Thresholds (1.0 buffer for consistency)
    cal_mins = cal_obs_resampled.min(axis=0)
    cal_maxs = cal_obs_resampled.max(axis=0)
    # Range is technically calculated on raw signal, while PatchAD is on windows
    # Buffers = 1.0 * range
    buffers = (cal_maxs - cal_mins) * 1.0
    thresh_obs_lower = cal_mins - buffers
    thresh_obs_upper = cal_maxs + buffers
    
    print(f"PatchAD Calibration Complete. Threshold: {patch_threshold:.6f}")

    # ----------------------------------------------------------------------
    # STEP 2: EXPERIMENT LOOP
    # ----------------------------------------------------------------------
    for test_path in ALL_EXPERIMENTS:
        print(f"\n{'#'*80}\nEXPERIMENT: {test_path}\n{'#'*80}")
        
        obs_file = f'{test_path}/observations.csv'
        if not os.path.exists(obs_file):
            print(f"[SKIP] Files missing for {test_path}")
            continue

        # Load and Resample Test Data
        df_test_raw = read_observations_csv(obs_file, obs_dims=obs_dims)
        test_obs_resampled, test_timestamps = resample_to_50hz(df_test_raw, obs_dims)
        if len(test_obs_resampled) < detector.win_size:
            print(f"[SKIP] Experiment too short ({len(test_obs_resampled)} samples) for PatchAD window ({detector.win_size})")
            continue

        start_time = test_timestamps[0]
        
        # 1. Evaluate PatchAD Anomaly
        test_tensor = torch.as_tensor(test_obs_resampled, dtype=torch.float32, device='cuda')
        first_patch_idx, first_patch_score = detector.find_first_anomaly(test_tensor, patch_threshold)
        
        # 2. Evaluate Range Anomaly
        range_anom_mask = ((test_obs_resampled < thresh_obs_lower) | \
                           (test_obs_resampled > thresh_obs_upper)).any(axis=1)
        first_range_idx = np.where(range_anom_mask)[0][0] if range_anom_mask.any() else None
        
        # 3. Handle Dual-Trigger Termination
        if first_patch_idx is not None and first_range_idx is not None:
            p_time = test_timestamps[first_patch_idx] - start_time
            r_time = test_timestamps[first_range_idx] - start_time
            print(f"\n[TERMINATE] Both PatchAD and Range anomalies detected in {test_path}.")
            # Using .item() if these were single-element arrays to avoid formatting errors
            print(f"  PatchAD Trigger: {p_time:.4f}s (Idx {first_patch_idx}) | Range Trigger: {r_time:.4f}s (Idx {first_range_idx})")

        # 4. PatchAD Reporting
        if first_patch_idx is not None:
            rel_time = test_timestamps[first_patch_idx] - start_time
            print(f"\n[FIRST PATCHAD ANOMALY DETECTED]")
            print(f"Time: {rel_time:.4f}s (Index {first_patch_idx})")
            print(f"Score: {first_patch_score:.6f} > Threshold {patch_threshold:.6f}")
        else:
            print("\nNo PatchAD anomalies detected.")

        # 5. Range Reporting
        if first_range_idx is not None:
            rel_time = test_timestamps[first_range_idx] - start_time
            print(f"\n[FIRST RANGE ANOMALY DETECTED]")
            print(f"Time: {rel_time:.4f}s (Index {first_range_idx})")
            
            # Find detail
            row_obs = test_obs_resampled[first_range_idx]
            for i, dim in enumerate(obs_dims):
                val = row_obs[i]
                if val < thresh_obs_lower[i]:
                    print(f"Type: Out-of-Range Low ('{dim}' {val:.4f} < {thresh_obs_lower[i]:.4f})")
                    break
                elif val > thresh_obs_upper[i]:
                    print(f"Type: Out-of-Range High ('{dim}' {val:.4f} > {thresh_obs_upper[i]:.4f})")
                    break
        else:
            print("No Range anomalies detected.")
if __name__ == "__main__":
    run_patchad_batch_testing()