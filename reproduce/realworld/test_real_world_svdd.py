import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import json
import os
import joblib  # Required to load isoforest.joblib
from collections import deque
import time

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

SVDD_MODEL_PATH = "logs/rsl_rl/unitree_g1_29dof_velocity/2026-01-04_14-25-06_multi_pcgrad_1213/ood_results_svdd/deep_svdd_model.pt"
SVDD_HIDDEN_DIM = 128
SVDD_LATENT_DIM = 32

# ==================================================================================
# 1. MODEL DEFINITIONS
# ==================================================================================
class SVDDNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    # net.0
            nn.BatchNorm1d(hidden_dim),          # net.1
            nn.ReLU(),                           # net.2
            nn.Dropout(0.1),                     # net.3
            nn.Linear(hidden_dim, hidden_dim),   # net.4
            nn.BatchNorm1d(hidden_dim),          # net.5
            nn.ReLU(),                           # net.6
            nn.Dropout(0.1),                     # net.7
            nn.Linear(hidden_dim, latent_dim, bias=False) # net.8
        )

    def forward(self, x):
        return self.net(x)


class DeepSVDDDetector:
    def __init__(self, model_path, obs_dim, device='cuda'):
        self.device = device
        self.network = SVDDNetwork(obs_dim, SVDD_HIDDEN_DIM, SVDD_LATENT_DIM).to(device)
        
        print(f"Loading Deep SVDD from {model_path}...")
        # weights_only=False is necessary for loading the center and R parameters
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # --- ROBUST KEY LOADING ---
        # Try 'network', then 'model_state_dict', then assume the checkpoint IS the state_dict
        if "network_state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["network_state_dict"])
        else:
            print("[ERROR] Could not find 'network_state' in checkpoint")

        # Load the parameters directly using the keys identified in your error
        self.center = checkpoint["center"].to(device)
        self.R = checkpoint["R"].to(device)
        self.mu = checkpoint["mu"].to(device)
        self.std = checkpoint["std"].to(device)
            
        self.network.eval()

    def score_samples(self, obs_tensor):
        """Higher score = More anomalous (distance squared to center)"""
        # Normalize using training stats
        norm_obs = (obs_tensor - self.mu) / (self.std + 1e-6)
        
        with torch.no_grad():
            outputs = self.network(norm_obs)
            # Compute squared Euclidean distance to center c
            dist_sq = torch.sum((outputs - self.center) ** 2, dim=1)
        return dist_sq


# ==================================================================================
# 3. MAIN LOOP (OPTIMIZED)
# ==================================================================================
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

def run_svdd_batch_testing():
    print(f"\n{'='*80}\n[PHASE 1] SVDD GLOBAL CALIBRATION\n{'='*80}")
    print(f"Loading nominal baseline from {CAL_BASE}...")
    
    # Load raw calibration data
    df_cal_obs_raw = pd.read_csv(f'{CAL_BASE}/observations.csv')
    
    # Determine Stride for 50Hz (consistent with RAPT logic)
    target_freq = 50.0
    actual_dt = df_cal_obs_raw['timestamp'].diff().median()
    stride = int(round((1.0 / target_freq) / actual_dt)) if actual_dt < (0.02 * 0.9) else 1
    
    if stride > 1:
        print(f"[INFO] Downsampling calibration by factor of {stride} to reach 50Hz...")
    
    df_cal_obs = df_cal_obs_raw.iloc[::stride].reset_index(drop=True)
    obs_dims = [c for c in df_cal_obs.columns if c.startswith('dim_')]
    
    # Initialize SVDD Model once
    svdd_detector = DeepSVDDDetector(SVDD_MODEL_PATH, len(obs_dims))
    
    # 1. Calculate SVDD Threshold
    print("Computing SVDD scores for calibration...")
    cal_obs_t = torch.tensor(df_cal_obs[obs_dims].values, dtype=torch.float32).to('cuda')
    with torch.no_grad():
        cal_scores = svdd_detector.score_samples(cal_obs_t)
    
    # Threshold: Max + 1 Std (or your preferred 3-sigma equivalent)
    svdd_threshold = (torch.max(cal_scores) + 0*torch.std(cal_scores)).item()
    
    # 2. Calculate Range Thresholds (1.0 buffer as per RAPT snippet)
    cal_mins = df_cal_obs[obs_dims].min(axis=0)
    cal_maxs = df_cal_obs[obs_dims].max(axis=0)
    buffers = (cal_maxs - cal_mins) * 1.0 
    thresh_obs_lower = cal_mins - buffers
    thresh_obs_upper = cal_maxs + buffers
    
    print(f"SVDD Calibration Complete. Threshold: {svdd_threshold:.10f}")
    
    # ----------------------------------------------------------------------
    # STEP 2: EXPERIMENT LOOP
    # ----------------------------------------------------------------------
    for test_path in ALL_EXPERIMENTS:
        print(f"\n{'#'*80}\nEXPERIMENT: {test_path}\n{'#'*80}")
        
        if not os.path.exists(f'{test_path}/observations.csv'):
            print(f"[SKIP] Files missing for {test_path}")
            continue

        # Load and Preprocess Test Data
        df_test_obs_raw = pd.read_csv(f'{test_path}/observations.csv')
        df_test_obs = df_test_obs_raw.iloc[::stride].reset_index(drop=True)
        start_time = df_test_obs['timestamp'].iloc[0]
        
        # 1. Evaluate SVDD Anomaly
        test_obs_t = torch.tensor(df_test_obs[obs_dims].values, dtype=torch.float32).to('cuda')
        with torch.no_grad():
            test_scores = svdd_detector.score_samples(test_obs_t)
        
        anom_mask = test_scores > svdd_threshold
        first_svdd_idx = torch.where(anom_mask)[0][0].item() if anom_mask.any() else None
        
        # 2. Evaluate Range Anomaly
        lower_viols = (df_test_obs[obs_dims] < thresh_obs_lower).any(axis=1)
        upper_viols = (df_test_obs[obs_dims] > thresh_obs_upper).any(axis=1)
        range_anomalies = lower_viols | upper_viols
        first_range_idx = range_anomalies.idxmax() if range_anomalies.any() else None
        
        # Mapping for output names
        real_names = get_observation_names()
        name_map = {d: (real_names[i] if i < len(real_names) else d) for i, d in enumerate(obs_dims)}

        # 3. Handle Dual-Trigger Termination
        if first_svdd_idx is not None and first_range_idx is not None:
            # Using .iloc[0] logic if these were Series, but indexing directly for clarity
            n_time = df_test_obs.iloc[first_svdd_idx]['timestamp'] - start_time
            r_time = df_test_obs.iloc[first_range_idx]['timestamp'] - start_time
            print(f"\n[TERMINATE] Both SVDD and Range anomalies detected in {test_path}.")
            print(f"  SVDD Trigger: {n_time:.4f}s (Idx {first_svdd_idx}) | Range Trigger: {r_time:.4f}s (Idx {first_range_idx})")

        # 4. Standard SVDD Reporting
        if first_svdd_idx is not None:
            row_idx = first_svdd_idx
            rel_time = df_test_obs.iloc[row_idx]['timestamp'] - start_time
            score_val = test_scores[row_idx].item()
            print(f"\n[FIRST SVDD ANOMALY DETECTED]")
            print(f"Time: {rel_time:.4f}s (Index {row_idx})")
            print(f"Anomaly Score: {score_val:.10f} > Threshold {svdd_threshold:.10f}")
        else:
            print("\nNo SVDD anomalies detected.")

        # 5. Standard Range Reporting
        if first_range_idx is not None:
            row_idx = first_range_idx
            rel_time = df_test_obs.iloc[row_idx]['timestamp'] - start_time
            print(f"\n[FIRST RANGE ANOMALY DETECTED]")
            print(f"Time: {rel_time:.4f}s (Index {row_idx})")
            
            row_obs = df_test_obs.iloc[row_idx][obs_dims]
            lower_v = row_obs[row_obs < thresh_obs_lower]
            upper_v = row_obs[row_obs > thresh_obs_upper]
            
            if not lower_v.empty:
                col = lower_v.index[0]
                val, limit = lower_v[col], thresh_obs_lower[col]
                print(f"Type: Out-of-Range Low ('{name_map.get(col, col)}' {val:.4f} < {limit:.4f})")
            elif not upper_v.empty:
                col = upper_v.index[0]
                val, limit = upper_v[col], thresh_obs_upper[col]
                print(f"Type: Out-of-Range High ('{name_map.get(col, col)}' {val:.4f} > {limit:.4f})")
        else:
            print("No Range anomalies detected.")



if __name__ == "__main__":
    txt = ''' I am using reinforcement learning to train a unitree g1 29dof humanoid robot in IsaacLab with parallel PPO, and doing sim2real transfer. I am investigating an anomaly / out of distribution event. I have a trained out of distribution detector (call it RAPT). Essentially, it predicts given the previous observation and action what the current observation should be. It is trained to minimize negative loss likelihood, so it outputs the predicted reconstruction (mu) and log_var. The architecture has residual layers followed by a GRU layer, which is then compressed slightly, and then decoded into the mu and log_var. I detect an anomaly with three criteria:
1. If any observation is outside the range it saw in the simulator +- the range of that observation * 50%. 
2. If the mean NLL of the observations is higher than the maximum mean NLL seen in nominal calibration over a few real-world normal episodes + 3 standard deviations. 
3. If, for any dimension, its NLL is greater than the maximum NLL seen in calibration for that dimension/observation + 5 standard deviations.

I have attached the code used to train this task so you know what the policy is doing. The anomaly detector is trained with 10 million observation vectors collected from massively parallel PPO simulation in IsaacLab. This included domain randomization seen in the training process. I have attached also the robot's CFG. The policy runs at 50Hz and so does the OOD detector (RAPT). 

Along with this, I have provided a saliency heatmap of the anomaly event. This has been computed via the integrated gradients method with 50 steps. This gives root-cause analysis with temporal and probabilistic intuition into the cause of the high negative loss likelihood. I have taken the top 10 highest attribution saliency observations (observations is the obs concatenated with actions) at the moment of the out of distribution event, and plotted them at 10 time-steps up to the exact point of failure. Furthermore, the I have also given the reason for the anomaly trigger (from the three criteria) and the time it happened. 

Furthermore, I have included 10 keyframes evenly spaced starting from 0.25 seconds before the anomaly to 0.25 afterwards.  

Please output the top 3 most likely causes of the out of distribution event, with 1 being most likely. Please underneath each of the three give reasons and details.
 These are the categories to choose from. Keep in mind not to make any assumptions about the operation of the robot, or the general likelihoods of each OOD event.
1. motor failure; 2. sensor failure; 3. sensor noise; 4. motor dynamics mismatch; 5. initial state issue (e.g. pose, joint positions or velocities); 6. observation scaling issue; 7. observation ordering issue; 8. joint outside max limits; 9. ground property mismatch (friction, deformable); 10. external force (collision, push); 11. mass distribution mismatch robot; 12. policy latency (constant offset or not); 13. coordinate frame mismatch with IMU; 14. sensor drift; 15. contact model mismatch; 16. passive mechanical resistance / joint movement constrained; 17. policy action scaling or mismatch; 18. payload mismatch. 19. power supply lag; 20. other (please specify).

[Saliency and OOD information]:'''
    print(txt)
    run_svdd_batch_testing()