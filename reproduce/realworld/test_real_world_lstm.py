import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import joblib
import h5py
from collections import deque
import time
torch.backends.cudnn.enabled = False

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

# LSTM-VAE and SVR Model Paths
MODEL_BASE = "collected_data/Unitree-G1-29dof-Velocity_20260328_190239"
VAE_CHECKPOINT = f"{MODEL_BASE}/best_model.pth"
SVR_MODEL_PATH = f"{MODEL_BASE}/svr_model.joblib"
STATS_PATH = f"{MODEL_BASE}/obs_stats.h5"

# Architecture (Must match collect_and_train_lstmvae.py)
LSTM_HIDDEN_DIM = 256
LSTM_LATENT_DIM = 24

# ==================================================================================
# 1. MODEL DEFINITIONS
# ==================================================================================

class TorchSVR(nn.Module):
    def __init__(self, sklearn_svr, device):
        super().__init__()
        self.device = device
        self.support_vectors = torch.from_numpy(sklearn_svr.support_vectors_).float().to(device)
        self.dual_coef = torch.from_numpy(sklearn_svr.dual_coef_).float().to(device).t()
        self.intercept = torch.from_numpy(sklearn_svr.intercept_).float().to(device)
        self.gamma = float(sklearn_svr._gamma if hasattr(sklearn_svr, '_gamma') else sklearn_svr.gamma)

    def forward(self, X):
        x_norm = (X ** 2).sum(dim=1, keepdim=True)
        sv_norm = (self.support_vectors ** 2).sum(dim=1)
        dot = torch.mm(X, self.support_vectors.t())
        sq_dist = torch.clamp(x_norm + sv_norm - 2 * dot, min=0.0)
        kernel_matrix = torch.exp(-self.gamma * sq_dist)
        return (torch.mm(kernel_matrix, self.dual_coef) + self.intercept).flatten()

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTM_VAE, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.fc_z_mu = nn.Linear(hidden_dim, latent_dim)
        self.decoder_projection = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.fc_x_mu = nn.Linear(hidden_dim, input_dim)
        self.fc_x_var = nn.Linear(hidden_dim, input_dim)

class LSTMSVRDetector:
    def __init__(self, device='cuda'):
        self.device = device
        with h5py.File(STATS_PATH, 'r') as f:
            self.obs_min = torch.from_numpy(f['min'][:]).float().to(device)
            self.obs_range = torch.from_numpy(f['range'][:]).float().to(device)
        
        obs_dim = self.obs_min.shape[0]
        self.vae = LSTM_VAE(obs_dim, LSTM_HIDDEN_DIM, LSTM_LATENT_DIM).to(device)
        ckpt = torch.load(VAE_CHECKPOINT, map_location=device, weights_only=False)
        sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items() if k not in ["p_start", "p_end"]}
        self.vae.load_state_dict(sd, strict=False)
        self.vae.eval()
        self.svr = TorchSVR(joblib.load(SVR_MODEL_PATH), device)

    @torch.no_grad()
    def process_sequence(self, obs_tensor):
        """Processes sequence with SVR batching to prevent OOM."""
        obs_norm = (obs_tensor - self.obs_min) / (self.obs_range + 1e-6)
        
        # 1. ENCODER (Whole sequence for LSTM state)
        lstm_input = obs_norm.unsqueeze(0).contiguous()
        out_enc, _ = self.vae.encoder_lstm(lstm_input)
        z_mu = self.vae.fc_z_mu(torch.tanh(out_enc.squeeze(0))) # [Seq, Latent]
        
        # 2. DECODER (Whole sequence for LSTM state)
        proj = self.vae.decoder_projection(z_mu.unsqueeze(0)).contiguous()
        out_dec, _ = self.vae.decoder_lstm(proj)
        h_dec = torch.tanh(out_dec.squeeze(0))
        recon_mu = torch.sigmoid(self.vae.fc_x_mu(h_dec))
        recon_var = F.softplus(self.vae.fc_x_var(h_dec)) + 1e-3
        
        # 3. VECTORIZED NLL
        nll = 0.5 * (torch.log(recon_var) + (obs_norm - recon_mu)**2 / recon_var).sum(dim=1)
        
        # 4. BATCHED SVR INFERENCE (Fix for OOM)
        # Process samples in blocks of 5000 to keep kernel matrix size manageable
        svr_batch_size = 5000 
        expected_nll_list = []
        for i in range(0, z_mu.size(0), svr_batch_size):
            z_chunk = z_mu[i : i + svr_batch_size]
            expected_nll_list.append(self.svr(z_chunk))
        
        expected_nll = torch.cat(expected_nll_list)
        
        return nll, expected_nll

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

def detect_anomalies():
    print(f"\n{'='*80}\n[PHASE 1] GLOBAL CALIBRATION\n{'='*80}")
    print(f"Loading nominal baseline from {CAL_BASE}...")
    
    # Load raw calibration data
    df_cal_obs_raw = pd.read_csv(f'{CAL_BASE}/observations.csv')
    df_cal_act_raw = pd.read_csv(f'{CAL_BASE}/actions.csv')
    
    # Determine Stride for 50Hz
    target_freq = 50.0
    actual_dt = df_cal_obs_raw['timestamp'].diff().median()
    stride = int(round((1.0 / target_freq) / actual_dt)) if actual_dt < (0.02 * 0.9) else 1
    
    if stride > 1:
        print(f"[INFO] Downsampling calibration by factor of {stride} to reach 50Hz...")
    
    # Apply Calibration Preprocessing (Stride + /0.25 Scaling)
    df_cal_obs = df_cal_obs_raw.iloc[::stride].reset_index(drop=True)
    df_cal_act = (df_cal_act_raw.iloc[::stride].reset_index(drop=True))
    
    # Apply action scaling logic from your snippet
    for col in [c for c in df_cal_act.columns if c != 'timestamp']:
        df_cal_act[col] = df_cal_act[col] / 0.25

    obs_dims = [c for c in df_cal_obs.columns if c.startswith('dim_')]
    detector = LSTMSVRDetector()

    # --- 2. SVR OFFSET CALIBRATION (Vectorized) ---
    print("Calibrating SVR C-Offset (Vectorized)...")
    cal_tensor = torch.tensor(df_cal_obs[obs_dims].values).float().to('cuda')
    actual_cal, expected_cal = detector.process_sequence(cal_tensor)
    residuals = (actual_cal - expected_cal).cpu().numpy()
    
    #svr_c_offset = np.max(residuals) + (0.0 * np.std(residuals))
    # use 99.73 percentile as threshold
    svr_c_offset = np.percentile(residuals, 99.73)
    cal_mins = df_cal_obs[obs_dims].min(axis=0)
    cal_maxs = df_cal_obs[obs_dims].max(axis=0)
    buffers = (cal_maxs - cal_mins) * 1.0
    thresh_obs_lower = cal_mins - buffers
    thresh_obs_upper = cal_maxs + buffers

    print(f"Calibration Complete. SVR C-Offset: {svr_c_offset:.4f}")
    for test_path in ALL_EXPERIMENTS:
        print(f"\n{'#'*80}\nEXPERIMENT: {test_path}\n{'#'*80}")
        
        if not os.path.exists(f'{test_path}/observations.csv'):
            print(f"[SKIP] Files missing for {test_path}")
            continue

        # Load and Preprocess Test Data
        df_test_obs_raw = pd.read_csv(f'{test_path}/observations.csv')
        df_test_obs = df_test_obs_raw.iloc[::stride].reset_index(drop=True)


        test_tensor = torch.tensor(df_test_obs[obs_dims].values).float().to('cuda')
        actual_test, expected_test = detector.process_sequence(test_tensor)
        
        # Check LSTM-SVR condition
        test_residuals = actual_test - expected_test
        lstm_anomalies = (test_residuals > svr_c_offset).cpu().numpy()
        start_time = df_test_obs['timestamp'].iloc[0]
        WARMUP_STEPS = 0
        lstm_anomalies[:WARMUP_STEPS] = False
        first_lstm_idx = np.where(lstm_anomalies)[0][0] if len(np.where(lstm_anomalies)[0]) > 0 else None
        lower_viols = (df_test_obs[obs_dims] < thresh_obs_lower).any(axis=1)
        upper_viols = (df_test_obs[obs_dims] > thresh_obs_upper).any(axis=1)
        range_anomalies = lower_viols | upper_viols
        range_anomalies[:WARMUP_STEPS] = False
        # range idx is a numpy array
        first_range_idx = np.where(range_anomalies)[0][0] if len(np.where(range_anomalies)[0]) > 0 else None

        if first_lstm_idx is not None and first_range_idx is not None:
            n_time = df_test_obs.iloc[first_lstm_idx]['timestamp'] - start_time
            r_time = df_test_obs.iloc[first_range_idx]['timestamp'] - start_time
            n_time_val = n_time.iloc[0] if hasattr(n_time, "iloc") else n_time
            r_time_val = r_time.iloc[0] if hasattr(r_time, "iloc") else r_time

            print(f"\n[TERMINATE] Both LSTM and Range anomalies detected in {test_path}.")            
            print(f"  LSTM Trigger: {n_time_val:.4f}s (Idx {first_lstm_idx}) | Range Trigger: {r_time_val:.4f}s (Idx {first_range_idx})")

        real_names = get_observation_names()
        name_map = {d: (real_names[i] if i < len(real_names) else d) for i, d in enumerate(obs_dims)}
        if first_lstm_idx is not None:
            row_idx = first_lstm_idx
            relative_time = df_test_obs.iloc[row_idx]['timestamp'] - start_time
            score_val = test_residuals[row_idx]
            
            print(f"\n[FIRST LSTM ANOMALY DETECTED]")
            print(f"Time: {relative_time:.4f}s (Index {row_idx})")
            print(f"Anomaly Score: {score_val:.4f} > Threshold { svr_c_offset:.4f}")
        else:
            print("\nNo LSTM-SVR anomalies detected.")

        if first_range_idx is not None:
            row_idx = first_range_idx
            relative_time = df_test_obs.iloc[row_idx]['timestamp'] - start_time
            
            print(f"\n[FIRST RANGE ANOMALY DETECTED]")
            print(f"Time: {relative_time:.4f}s (Index {row_idx})")
            
            # Find violation details
            row_obs = df_test_obs.iloc[row_idx][obs_dims]
            lower_v = row_obs[row_obs < thresh_obs_lower]
            upper_v = row_obs[row_obs > thresh_obs_upper]
            
            if not lower_v.empty:
                col = lower_v.index[0]
                val = lower_v[col]
                limit = thresh_obs_lower[col]
                real_name = name_map.get(col, col)
                print(f"Type: Out-of-Range Low ('{real_name}' {val:.4f} < {limit:.4f})")
            elif not upper_v.empty:
                col = upper_v.index[0]
                val = upper_v[col]
                limit = thresh_obs_upper[col]
                real_name = name_map.get(col, col)
                print(f"Type: Out-of-Range High ('{real_name}' {val:.4f} > {limit:.4f})")

        else:
            print("\nNo Range anomalies detected.")

if __name__ == "__main__":
    detect_anomalies()