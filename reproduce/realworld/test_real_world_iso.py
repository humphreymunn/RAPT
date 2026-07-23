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

ISO_MODEL_PATH = "logs/rsl_rl/unitree_g1_29dof_velocity/2026-01-04_14-25-06_multi_pcgrad_1213/ood_results_isoforest/isolation_forest.joblib"

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

    # 2. Load Isolation Forest
    if not os.path.exists(ISO_MODEL_PATH):
        sys.exit(f"Error: Isolation Forest model not found at {ISO_MODEL_PATH}")
    print(f"Loading Isolation Forest from {ISO_MODEL_PATH}...")
    iso_forest = joblib.load(ISO_MODEL_PATH)

    obs_dims = [c for c in df_cal_obs.columns if c.startswith('dim_')]        

    # Calculate Calibration Thresholds
    cal_scores = -iso_forest.decision_function(df_cal_obs[obs_dims].values)
    # Set threshold at 99.73 percentile of calibration data
    # if_threshold = np.max(cal_scores) + 0*np.std(cal_scores) * 1.0
    # use 99.73 percentile as threshold
    if_threshold = np.percentile(cal_scores, 99.73)
    cal_mins = df_cal_obs[obs_dims].min(axis=0)
    cal_maxs = df_cal_obs[obs_dims].max(axis=0)
    buffers = (cal_maxs - cal_mins) * 1.0
    thresh_obs_lower = cal_mins - buffers
    thresh_obs_upper = cal_maxs + buffers
    print(f"Calibration Complete. IsoForest Threshold: {if_threshold:.4f}")
    

    for test_path in ALL_EXPERIMENTS:
        print(f"\n{'#'*80}\nEXPERIMENT: {test_path}\n{'#'*80}")
        
        if not os.path.exists(f'{test_path}/observations.csv'):
            print(f"[SKIP] Files missing for {test_path}")
            continue

        # Load and Preprocess Test Data
        df_test_obs_raw = pd.read_csv(f'{test_path}/observations.csv')
        df_test_obs = df_test_obs_raw.iloc[::stride].reset_index(drop=True)

        # Compute Test Losses and prepare Tensors
        test_scores = -iso_forest.decision_function(df_test_obs[obs_dims].values)
        if_anomalies = test_scores > if_threshold
        
        start_time = df_test_obs['timestamp'].iloc[0]

        WARMUP_STEPS = 0
        if_anomalies[:WARMUP_STEPS] = False
        # nll idx is a numpy araray
        first_nll_idx = np.where(if_anomalies)[0][0] if len(np.where(if_anomalies)[0]) > 0 else None

        # 2. Evaluate Range Anomaly
        lower_viols = (df_test_obs[obs_dims] < thresh_obs_lower).any(axis=1)
        upper_viols = (df_test_obs[obs_dims] > thresh_obs_upper).any(axis=1)
        range_anomalies = lower_viols | upper_viols
        range_anomalies[:WARMUP_STEPS] = False
        # range idx is a numpy array
        first_range_idx = np.where(range_anomalies)[0][0] if len(np.where(range_anomalies)[0]) > 0 else None

        # 3. Handle Dual-Trigger Termination
        if first_nll_idx is not None and first_range_idx is not None:
            n_time = df_test_obs.iloc[first_nll_idx]['timestamp'] - start_time
            r_time = df_test_obs.iloc[first_range_idx]['timestamp'] - start_time
            print(f"\n[TERMINATE] Both IF and Range anomalies detected in {test_path}.")
            print(f"  IF Trigger: {n_time:.4f}s (Idx {first_nll_idx}) | Range Trigger: {r_time:.4f}s (Idx {first_range_idx})")

        # 4. Standard Reporting & Saliency
        real_names = get_observation_names()
        name_map = {d: (real_names[i] if i < len(real_names) else d) for i, d in enumerate(obs_dims)}
        dim_to_idx = {d: i for i, d in enumerate(obs_dims)}

        if first_nll_idx is not None:
            row_idx = first_nll_idx
            # FIX: Access timestamp from df_test_obs instead of df_test_loss
            relative_time = df_test_obs.iloc[row_idx]['timestamp'] - start_time
            score_val = test_scores[row_idx]
            
            print(f"\n[FIRST ISOFOREST ANOMALY DETECTED]")
            print(f"Time: {relative_time:.4f}s (Index {row_idx})")
            print(f"Anomaly Score: {score_val:.4f} > Threshold {if_threshold:.4f}")

        else:
            print("\nNo Isolation Forest anomalies detected.")

        # --- Report Range ---
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
    detect_anomalies()