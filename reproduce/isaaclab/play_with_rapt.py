"""
Script to systematically test OOD detection using a HYBRID Pipeline:
1. Simple Range Detector (Fast, catches impossible values)
2. Probabilistic rapt (Smart, catches context violations)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from enum import Enum
import os
from isaaclab.app import AppLauncher
import matplotlib.pyplot as plt

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test OOD detection with Hybrid (rapt + Range) detector.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--use_critic_multi", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--calibration_cache",
    type=str,
    default=None,
    help="Path to calibration cache (.pt). If exists, load; otherwise save after calibration."
)

parser.add_argument(
    "--force_recalibrate",
    action="store_true",
    help="Force recalibration even if calibration cache exists."
)

# --- ANOMALY DETECTION ARGUMENTS ---
parser.add_argument("--ood_detect_type", type=str, required=False, default="both", choices=['max','mean','both'])
parser.add_argument("--rapt_checkpoint", type=str, required=True, help="Path to trained rapt checkpoint")
parser.add_argument("--rapt_embed_dim", type=int, default=256,
                    help="rapt embedding dimension (must match training)")
parser.add_argument("--rapt_num_blocks", type=int, default=4,
                    help="rapt number of residual blocks (must match training)")
parser.add_argument("--padding_epsilon", type=float, default=1e-1, 
                    help="Padding added to range bounds to prevent float errors")
parser.add_argument("--margin_percent", type=float, default=0.2, help="Margin percentage for range detector")
parser.add_argument("--mask_ratio", type=float, default=0.25,
                    help="Mask ratio for rapt during OOD detection")

parser.add_argument("--reconstruction_type", type=str, default="bottleneck", choices=["masked", "bottleneck"])
parser.add_argument("--use_residual", action="store_true", default=True)
parser.add_argument("--use_probabilistic", action="store_true", default=True)
parser.add_argument("--use_temporal", action="store_true", default=True)
parser.add_argument("--dropout", type=float, default=0.0) # For init compatibility

# --- OOD TESTING ARGUMENTS ---
parser.add_argument("--episode_length", type=int, default=100,
                    help="Number of steps per episode (default: 100)")
parser.add_argument("--ood_start_step", type=int, default=0,
                    help="Step at which OOD injection begins")
parser.add_argument("--output_dir", type=str, default="ood_results_hybrid",
                    help="Directory to save results")
parser.add_argument("--categories", type=str, default="all",
                    help="Comma-separated list of category indices to test, or 'all'")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import json
import h5py
import numpy as np
import torch.nn as nn
from collections import deque
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
torch.backends.cudnn.enabled = False

# ==================================================================================
# OOD CATEGORIES
# ==================================================================================

class OODCategory(Enum):
    """OOD perturbation categories for testing."""
    NONE = -1               # Control group (no perturbation)
    SENSOR_DRIFT = 0        # Gradual drift in sensor readings
    SENSOR_ZERO = 1         # Sensor outputs zero (failure)
    SCALE_HALF = 2          # Observation scaled by 0.5
    SCALE_DOUBLE = 3        # Observation scaled by 2.0
    OBS_SWAP = 4            # Swap pair of observation indices
    ACTION_SWAP = 5         # Swap pair of action indices
    NOISE = 6               # Add Gaussian noise to observations
    LATENCY_OFFSET = 7      # Constant delay (20-200ms)
    LATENCY_SLOW = 8        # Update rate halved (x2 slower)
    # Physics-based OOD (requires env access)
    ACTUATOR_DYNAMICS = 9   # Motor params: torque, stiffness, damping
    INIT_STATE = 10         # Initial state perturbation (joint pos, joint vel, body pose, body orientation)
    ENV_DISTURBANCE = 11    # External disturbance forces applied to the robot body (payload, push)
    ENV_FRICTION = 12       # Change ground friciton
    FROZEN_SENSOR = 13      # Sensor outputs frozen value (stuck sensor)

CATEGORY_NAMES = {
    OODCategory.NONE: "none",
    OODCategory.SENSOR_DRIFT: "sensor_drift",
    OODCategory.SENSOR_ZERO: "sensor_zero",
    OODCategory.SCALE_HALF: "scale_half",
    OODCategory.SCALE_DOUBLE: "scale_double",
    OODCategory.OBS_SWAP: "obs_swap",
    OODCategory.ACTION_SWAP: "action_swap",
    OODCategory.NOISE: "noise",
    OODCategory.LATENCY_OFFSET: "latency_offset",
    OODCategory.LATENCY_SLOW: "latency_slow",
    OODCategory.ACTUATOR_DYNAMICS: "actuator_dynamics",
    OODCategory.INIT_STATE: "init_state",
    OODCategory.ENV_DISTURBANCE: "env_disturbance",
    OODCategory.ENV_FRICTION: "env_friction",
    OODCategory.FROZEN_SENSOR: "frozen_sensor",
}


class SaliencyExplainer:
    def __init__(self, detector, device='cuda'):
        self.detector = detector
        self.model = detector.model
        self.device = device
        self.obs_names = self._get_observation_names()
        
    def _get_observation_names(self):
        # Names for Unitree G1 (29 DoF) + Base + Commands
        joint_names = [
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "waist_yaw", "waist_roll", "waist_pitch",
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"
        ]
        names = ["root_vel_x", "root_vel_y", "root_vel_z", "gravity_x", "gravity_y", "gravity_z", "cmd_vx", "cmd_vy", "cmd_dyaw"]
        names.extend([f"pos_{n}" for n in joint_names])
        names.extend([f"vel_{n}" for n in joint_names])
        names.extend([f"act_{n}" for n in joint_names])
        return names

    def compute_ig(self, obs, act, hidden, baseline):
        self.model.eval() 
        steps = 50 # Reduced to 30 or 50 for speed is fine
        
        # 1. Interpolation
        alphas = torch.linspace(0, 1, steps, device=self.device).view(steps, 1)
        interpolated_obs = baseline + alphas * (obs - baseline)
        interpolated_obs.requires_grad_(True)
        
        expanded_act = act.unsqueeze(0).expand(steps, -1)
        if hidden is not None:
            context_hidden = hidden.expand(-1, steps, -1).contiguous()
        else:
            context_hidden = None

        # --- FIX STARTS HERE ---
        # Disable CuDNN to allow backprop in eval mode
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False 
        
        try:
            output, _, _ = self.model(interpolated_obs, actions=expanded_act, hidden=context_hidden)
            
            # (Loss calculation same as your code...)
            if self.detector.use_probabilistic:
                mu, log_var = output[:, :self.detector.obs_dim], output[:, self.detector.obs_dim:]
                sigma = torch.exp(0.5 * log_var)
                obs_norm = (interpolated_obs - self.detector.obs_mean) / self.detector.obs_std
                error = mu - obs_norm
                loss = (error ** 2) / (2 * sigma ** 2) + 0.5 * log_var
                total_loss = loss.sum(dim=1).sum()
            else:
                obs_norm = (interpolated_obs - self.detector.obs_mean) / self.detector.obs_std
                loss = ((output - obs_norm) ** 2)
                total_loss = loss.sum()

            grads = torch.autograd.grad(total_loss, interpolated_obs)[0]
            
        finally:
            torch.backends.cudnn.enabled = prev_cudnn
        # --- FIX ENDS HERE ---

        ig = (obs - baseline) * grads.mean(dim=0)
        return ig.detach()

    def plot_top_k_saliency(self, trace_buffer, save_path="top5_saliency.png", k=5):
        """
        Plots only the Top K features with the highest NLL contribution (Saliency).
        """
        if not trace_buffer: return
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Compute Saliency (Same as before)
        window_igs = []
        baseline_stack = [x['obs'] for x in trace_buffer[:5]]
        baseline = torch.stack(baseline_stack).mean(dim=0)
        
        print(f"[EXPLAINER] Ranking Top {k} features for {len(trace_buffer)} steps...")
        
        for frame in trace_buffer:
            ig = self.compute_ig(frame['obs'], frame['act'], frame['hidden'], baseline)
            window_igs.append(ig)
            
        # Matrix: [Obs_Dim, Time]
        saliency_matrix = torch.stack(window_igs).abs().cpu().numpy().T
        
        # 2. FILTERING LOGIC (The new part)
        # We rank features by their MAXIMUM influence at any point in the window.
        # This ensures we catch brief spikes (like impacts).
        max_score_per_dim = np.max(saliency_matrix, axis=1)
        
        # Get indices of the top K features (sorted descending)
        top_indices = np.argsort(max_score_per_dim)[-k:][::-1]
        
        # Extract only those rows
        top_matrix = saliency_matrix[top_indices, :]
        top_names = [self.obs_names[i] for i in top_indices]
        
        # 3. Plotting (Optimized for small number of rows)
        fig, ax = plt.subplots(figsize=(12, k * 0.8)) # Dynamic height
        
        vmax = np.percentile(top_matrix, 99) if np.max(top_matrix) > 0 else 1.0
        im = ax.imshow(top_matrix, cmap='Reds', aspect='auto', 
                       interpolation='nearest', vmin=0, vmax=vmax)
        
        # Labels
        ax.set_yticks(np.arange(k))
        ax.set_yticklabels(top_names, fontsize=12, fontweight='bold')
        
        # Time Axis
        times = [f"{x['time']:.2f}" for x in trace_buffer]
        tick_indices = np.arange(0, len(times), 5)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([times[i] for i in tick_indices])
        ax.set_xlabel("Time (s)")
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NLL Contribution")
        plt.title(f"Top {k} Anomalous Features", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[EXPLAINER] Saved Top-{k} heatmap to {save_path}")
    
    def plot_categorical_heatmap(self, trace_buffer, save_path="categorical_saliency.png"):
        """
        Aggregates dimensions into 6 specific functional categories via Max-Pooling.
        Categories: IMU, Lower/Upper Sensors, Lower/Upper Actions, Commands.
        """
        if not trace_buffer: return
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Define Categories & Sort Indices
        # We define the 6 specific buckets you requested
        categories = {
            "Command": [],
            "Upper Body Action": [],
            "Lower Body Action": [],
            "Upper Body Sensor": [],
            "Lower Body Sensor": [],
            "IMU": []
        }
        
        # Helper to categorize a single feature name
        for idx, name in enumerate(self.obs_names):
            if any(x in name for x in ['cmd_']):
                categories['Command'].append(idx)
            elif any(x in name for x in ['root_vel', 'gravity']):
                categories['IMU'].append(idx)
            elif 'act_' in name:
                # Actions
                if any(x in name for x in ['hip', 'knee', 'ankle']):
                    categories['Lower Body Action'].append(idx)
                else:
                    categories['Upper Body Action'].append(idx)
            else:
                # Sensors (Positions & Velocities)
                if any(x in name for x in ['hip', 'knee', 'ankle']):
                    categories['Lower Body Sensor'].append(idx)
                else:
                    categories['Upper Body Sensor'].append(idx)

        # The order you requested for the Y-axis (Top to Bottom)
        row_labels = [
            "IMU", 
            "Lower Body Sensor", 
            "Upper Body Sensor", 
            "Lower Body Action", 
            "Upper Body Action", 
            "Command"
        ]

        # 2. Compute Saliency (Standard IG Loop)
        window_igs = []
        # Robust baseline from start of window
        baseline_stack = [x['obs'] for x in trace_buffer[:5]]
        baseline = torch.stack(baseline_stack).mean(dim=0)
        
        print(f"[EXPLAINER] Computing Categorical Saliency for {len(trace_buffer)} steps...")
        
        for frame in trace_buffer:
            # Note: Ensure compute_ig has the 'torch.backends.cudnn.enabled = False' fix!
            ig = self.compute_ig(frame['obs'], frame['act'], frame['hidden'], baseline)
            window_igs.append(ig)
            
        # Raw Matrix: [Time, Obs_Dim]
        saliency_matrix = torch.stack(window_igs).abs().cpu().numpy()
        
        # 3. Aggregate (Max Pooling per Category)
        heatmap_data = []
        
        for label in row_labels:
            indices = categories[label]
            if not indices:
                # Handle empty categories gracefully
                heatmap_data.append(np.zeros(len(trace_buffer)))
            else:
                # Extract columns for this category
                group_data = saliency_matrix[:, indices]
                # MAX POOLING: Finds the single most anomalous feature in this group
                heatmap_data.append(np.max(group_data, axis=1))
        
        heatmap_data = np.array(heatmap_data) # Shape: [6, Time]

        # 4. Plotting
        # Height is dynamic but fixed for 6 rows roughly 2 inches
        fig, ax = plt.subplots(figsize=(10, 2.5)) 
        
        # Use 99th percentile for max contrast without washout
        vmax = np.percentile(heatmap_data, 99) if np.max(heatmap_data) > 0 else 1.0
        
        im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto', 
                       interpolation='nearest', vmin=0, vmax=vmax)
        
        # Y-Axis Labels
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=10, fontweight='bold')
        
        # X-Axis Labels (Time)
        times = [f"{x['time']:.2f}" for x in trace_buffer]
        tick_indices = np.arange(0, len(times), 5)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([times[i] for i in tick_indices], fontsize=9)
        ax.set_xlabel("Time (s)")
        
        # Add grid lines to separate the categories clearly
        for r in range(len(row_labels)):
            ax.axhline(r - 0.5, color='black', linewidth=1, alpha=0.1)

        plt.title("Saliency by Functional Category", fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[EXPLAINER] Saved Categorical Heatmap to {save_path}")

    def plot_full_saliency(self, trace_buffer, save_path="saliency_heatmap.png"):
        """
        Generates the raw (ungrouped) heatmap for the provided trace buffer.
        trace_buffer: List of dicts {'obs':, 'act':, 'hidden':, 'time':}
        """
        if not trace_buffer: return
        
        # 1. Compute IG for the whole window
        window_igs = []
        
        # Calculate baseline (mean of first few frames)
        baseline_stack = [x['obs'] for x in trace_buffer[:5]]
        baseline = torch.stack(baseline_stack).mean(dim=0)
        
        print(f"[EXPLAINER] Computing saliency for {len(trace_buffer)} steps...")
        
        for i, frame in enumerate(trace_buffer):
            ig = self.compute_ig(frame['obs'], frame['act'], frame['hidden'], baseline)
            window_igs.append(ig)
            
        # 2. Prepare Matrix: [Obs_Dim, Time]
        saliency_matrix = torch.stack(window_igs).abs().cpu().numpy().T
        
        # 3. Plotting
        n_dims = len(self.obs_names)
        # Dynamic height: 0.2 inches per row ensure readable labels
        fig_height = max(10, n_dims * 0.2)
        
        fig, ax = plt.subplots(figsize=(14, fig_height))
        
        # Robust Scaling: Use 99th percentile to avoid one huge spike flattening the colors
        vmax = np.percentile(saliency_matrix, 99.5)
        im = ax.imshow(saliency_matrix, cmap='Reds', aspect='auto', 
                       interpolation='nearest', vmin=0, vmax=vmax)
        
        # Y-Axis: Feature Names
        if n_dims == saliency_matrix.shape[0]:
            ax.set_yticks(np.arange(n_dims))
            ax.set_yticklabels(self.obs_names, fontsize=9)
        else:
            print(f"Warning: Name count ({n_dims}) != Obs dim ({saliency_matrix.shape[0]})")
        
        # X-Axis: Time
        times = [f"{x['time']:.2f}" for x in trace_buffer]
        tick_indices = np.arange(0, len(times), 5) # Tick every 5 steps
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([times[i] for i in tick_indices])
        ax.set_xlabel("Time (s)")
        
        plt.colorbar(im, ax=ax, label="Influence (IG)")
        plt.title("Full State Saliency (Ungrouped)", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[EXPLAINER] Saved heatmap to {save_path}")

class ConfigurableBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=True, dropout=0.1):
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)
        hidden_dim = int(out_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        return self.relu(x + out) if self.use_residual else self.relu(out)

class UniversalModel(nn.Module):
    """
    Handles both Autoencoder (RAPT) and Forward Dynamics (Baseline) architectures.
    """
    def __init__(self, obs_dim, action_dim, args, train_dynamics=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args
        self.use_temporal = args.use_temporal
        self.embed_dim = args.rapt_embed_dim
        self.train_dynamics = train_dynamics

        # --- Input Dimension Logic ---
        self.input_dim = obs_dim + action_dim if self.train_dynamics else obs_dim

        # --- Bottleneck Calculation (AE Only) ---
        if args.reconstruction_type == 'bottleneck':
            keep_ratio = 1.0 - args.mask_ratio
            self.latent_dim = max(4, int(self.embed_dim * keep_ratio))
        else:
            self.latent_dim = self.embed_dim

        # --- 1. Input Encoder (MLP) ---
        encoder_layers = [nn.Linear(self.input_dim, self.embed_dim), nn.ReLU()]
        for _ in range(args.rapt_num_blocks):
            encoder_layers.append(ConfigurableBlock(self.embed_dim, self.embed_dim, args.use_residual, args.dropout))
        self.encoder_mlp = nn.Sequential(*encoder_layers)

        # --- 2. Temporal Core (GRU) ---
        if self.use_temporal:
            self.gru = nn.GRU(self.embed_dim, self.embed_dim, num_layers=1, batch_first=True)

        # --- 3. Bottleneck Projection (AE Only) ---
        self.use_bottleneck = (args.reconstruction_type == 'bottleneck')
        if self.use_bottleneck:
            self.compress = nn.Sequential(
                nn.Linear(self.embed_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.ReLU()
            )
            self.decompress = nn.Sequential(
                nn.Linear(self.latent_dim, self.embed_dim),
                nn.ReLU()
            )

        # --- 4. Output Decoder (MLP) ---
        decoder_layers = []
        for _ in range(args.rapt_num_blocks):
            decoder_layers.append(ConfigurableBlock(self.embed_dim, self.embed_dim, args.use_residual, args.dropout))
        self.decoder_mlp = nn.Sequential(*decoder_layers)

        # --- 5. Output Head ---
        out_features = obs_dim * 2 if args.use_probabilistic else obs_dim
        self.head = nn.Linear(self.embed_dim, out_features)

    def forward(self, x, actions=None, hidden=None, force_no_mask=False):
        """
        x: Observations [Batch, Seq, Obs_Dim]
        actions: Actions [Batch, Seq, Act_Dim]
        """
        is_sequence = x.dim() == 3
        if is_sequence:
            batch, seq, dim = x.shape
            x_flat = x.reshape(-1, dim)
        else:
            batch, dim = x.shape
            x_flat = x

        # --- 1. APPLY MASKING TO OBSERVATIONS FIRST ---
        mask = None
        x_in = x_flat
        
        # Allow masking if configured (even for Dynamics)
        if (self.args.reconstruction_type == 'masked') and (not force_no_mask):
            if self.args.mask_ratio > 0.0:
                noise = torch.rand_like(x_flat)
                num_masked = int(dim * self.args.mask_ratio)
                _, masked_indices = torch.topk(noise, num_masked, dim=1)
                
                # Create mask for OBSERVATIONS only
                mask_flat = torch.zeros_like(x_flat, dtype=torch.bool)
                mask_flat.scatter_(1, masked_indices, True)
                
                x_in = x_flat.clone()
                x_in[mask_flat] = 0.0
                
                if is_sequence: mask = mask_flat.view(batch, seq, dim)
                else: mask = mask_flat
        
        # --- 2. CONCATENATE ACTIONS (Clean, Unmasked) ---
        if self.train_dynamics:
            if actions is None: raise ValueError("Dynamics model requires 'actions' input!")
            
            # Helper to flatten actions
            if is_sequence: act_flat = actions.reshape(-1, actions.shape[-1])
            else: act_flat = actions
            
            # Check shapes
            if x_in.shape[:-1] != act_flat.shape[:-1]:
                raise ValueError(f"Shape mismatch: Obs {x_in.shape} vs Act {act_flat.shape}")

            # Concat Masked Obs + Clean Actions
            model_input = torch.cat([x_in, act_flat], dim=-1)
        else:
            model_input = x_in

        # --- 3. ENCODE ---
        z = self.encoder_mlp(model_input)

        # 4. Temporal
        if self.use_temporal:
            if not is_sequence:
                z = z.unsqueeze(1)
            else:
                z = z.view(batch, seq, -1)
            z, hidden = self.gru(z, hidden)
            z = z.reshape(-1, z.shape[-1])

        # 5. Bottleneck
        if self.use_bottleneck:
            z = self.compress(z)
            z = self.decompress(z)

        # 6. Decode
        z = self.decoder_mlp(z)
        out = self.head(z)

        if is_sequence:
            out = out.view(batch, seq, -1)
        return out, mask, hidden


# ==================================================================================
# OOD INJECTOR
# ==================================================================================

class OODInjector:
    """
    Injects various OOD perturbations into observations and actions.
    
    The second half of environments (indices num_envs//2 to num_envs-1) receive
    OOD perturbations; the first half are control (no perturbation).
    """
    
    def __init__(self, num_envs: int, obs_dim: int, action_dim: int, 
                 dt: float = 0.02, device: str = 'cuda', env=None):
        #if num_envs < 2:
        #    raise ValueError(f"num_envs must be at least 2 for OOD testing (got {num_envs}). "
        #                   f"Need half for control, half for OOD injection.")
        
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dt = dt  # 20ms per step
        self.device = device
        self.env = env  # Reference to environment for physics perturbations
        
        # OOD mask: True for environments that should have OOD injected
        self.ood_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ood_mask[num_envs // 2:] = True
        self.num_ood_envs = num_envs - (num_envs // 2)  # Ensures at least 1 OOD env
        self.num_control_envs = num_envs // 2

        # OOD environment indices (for physics perturbations)
        self.ood_env_ids = torch.arange(self.num_control_envs, num_envs, device=device)
        self.ood_env_ids_cpu = self.ood_env_ids.cpu()
        
        # Current category
        self.category = OODCategory.NONE
        self.ood_active = False  # Whether OOD injection has started
        
        # Per-environment configurations (only for OOD envs)
        self.affected_obs_indices = None      # Which obs indices are affected
        self.affected_action_indices = None   # Which action indices are affected (for action swap)
        self.swap_pairs = None                # Pairs to swap (for swapping categories)
        self.drift_rate = None                # Drift rate per step
        self.drift_accumulator = None         # Accumulated drift
        self.noise_limit = None                 # Noise standard deviation
        self.latency_buffer = None            # Buffer for latency
        self.latency_steps = None             # Steps of latency per env
        self.latency_update_interval = None   # Steps between updates (for low freq)
        self.latency_counter = None           # Counter for slow/low-freq updates
        self.last_obs = None                  # Last observation (for latency)
        
        # Physics perturbation state
        self.physics_perturbed = False
        self.original_masses = None
        self.original_inertias = None
        # Store per-actuator originals as dictionaries keyed by actuator name
        self.original_effort_limit = {}   # {actuator_name: tensor}
        self.original_stiffness = {}      # {actuator_name: tensor}
        self.original_damping = {}        # {actuator_name: tensor}
        self.mass_scale_factors = None        # Per-env mass scale (0.5 or 2.0)
        self.affected_body_ids = None         # Which bodies are affected
        self.actuator_perturbation_type = None  # 'torque', 'stiffness', 'damping', or 'combined'
        self.actuator_scale_factors = None    # {actuator_name: {param: tensor}} Scale factors for actuator params
        
    def set_all_ood(self):
        """Make EVERY environment receive OOD injection (no control envs)."""
        self.num_control_envs = 0
        self.num_ood_envs = self.num_envs

        self.ood_mask[:] = True
        self.ood_env_ids = torch.arange(0, self.num_envs, device=self.device)
        self.ood_env_ids_cpu = self.ood_env_ids.cpu()
        
    def setup_episode(self, category: OODCategory, seed: int = None):
        """
        Configure OOD effects for a new episode.
        Each OOD environment gets randomized parameters within the category.
        """
        self.category = category
        self.ood_active = False
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Reset all buffers (including physics)
        self._reset_buffers()
        self._reset_physics()
        
        if category == OODCategory.NONE:
            return
        
        # Configure based on category
        if category == OODCategory.SENSOR_DRIFT:
            self._setup_sensor_drift()
        elif category == OODCategory.SENSOR_ZERO:
            self._setup_sensor_zero()
        elif category == OODCategory.SCALE_HALF:
            self._setup_scale(0.5)
        elif category == OODCategory.SCALE_DOUBLE:
            self._setup_scale(2.0)
        elif category == OODCategory.OBS_SWAP:
            self._setup_obs_swap()
        elif category == OODCategory.ACTION_SWAP:
            self._setup_action_swap()
        elif category == OODCategory.NOISE:
            self._setup_noise()
        elif category == OODCategory.LATENCY_OFFSET:
            self._setup_latency_offset()
        elif category == OODCategory.LATENCY_SLOW:
            self._setup_latency_slow()
        elif category == OODCategory.ACTUATOR_DYNAMICS:
            self._setup_actuator_dynamics()
        elif category == OODCategory.INIT_STATE:
            self._setup_init_state_perturbation()
        elif category == OODCategory.ENV_DISTURBANCE:
            self._setup_env_disturbance()
        elif category == OODCategory.ENV_FRICTION:
            self._setup_env_friction()
        elif category == OODCategory.FROZEN_SENSOR:
            self._setup_frozen_sensor()
    
    def _reset_buffers(self):
        """Reset all perturbation buffers."""
        self.affected_obs_indices = None
        self.affected_action_indices = None
        self.swap_pairs = None
        self.drift_rate = None
        self.drift_accumulator = None
        self.noise_limit = None
        self.latency_buffer = None
        self.latency_steps = None
        self.latency_update_interval = None
        self.latency_counter = None
        self.last_obs = None
        self.scale_factor = None
        self.mass_scale_factors = None
        self.affected_body_ids = None
        self.actuator_perturbation_type = None
        self.actuator_scale_factors = None
        # Reset actuator dictionaries
        self.original_effort_limit = {}
        self.original_stiffness = {}
        self.original_damping = {}
    
    def _reset_physics(self):
        """Reset physics parameters to original values."""
        if not self.physics_perturbed or self.env is None:
            return
        
        try:
            # Get robot from scene
            unwrapped_env = self.env.unwrapped
            if hasattr(unwrapped_env, 'scene'):
                robot = unwrapped_env.scene["robot"]
            elif hasattr(unwrapped_env, '_robot'):
                robot = unwrapped_env._robot
            else:
                print("[WARNING] Could not find robot in environment for physics reset")
                return
            
            all_env_ids = torch.arange(self.num_envs, device='cpu')
            
            # Reset masses and inertias
            if self.original_masses is not None:
                # Get current masses, restore OOD envs, set all
                masses = robot.root_physx_view.get_masses()
                masses[self.ood_env_ids_cpu] = self.original_masses
                robot.root_physx_view.set_masses(masses, all_env_ids)
                self.original_masses = None
            
            if self.original_inertias is not None:
                inertias = robot.root_physx_view.get_inertias()
                inertias[self.ood_env_ids_cpu] = self.original_inertias
                robot.root_physx_view.set_inertias(inertias, all_env_ids)
                self.original_inertias = None
            
            # Reset actuator parameters (per actuator group)
            if hasattr(robot, 'actuators'):
                for act_name, actuator in robot.actuators.items():
                    if act_name in self.original_effort_limit:
                        actuator.effort_limit[self.ood_env_ids] = self.original_effort_limit[act_name]
                    
                    if act_name in self.original_stiffness and hasattr(actuator, 'stiffness'):
                        actuator.stiffness[self.ood_env_ids] = self.original_stiffness[act_name]
                    
                    if act_name in self.original_damping and hasattr(actuator, 'damping'):
                        actuator.damping[self.ood_env_ids] = self.original_damping[act_name]
            
            # Clear dictionaries
            self.original_effort_limit = {}
            self.original_stiffness = {}
            self.original_damping = {}
            
            self.physics_perturbed = False
            print("[INFO] Physics parameters reset to original values")
            
        except Exception as e:
            print(f"[WARNING] Failed to reset physics: {e}")
    
    def _get_robot(self):
        """Get robot from environment."""
        if self.env is None:
            return None
        
        unwrapped_env = self.env.unwrapped
        if hasattr(unwrapped_env, 'scene'):
            return unwrapped_env.scene["robot"]
        elif hasattr(unwrapped_env, '_robot'):
            return unwrapped_env._robot
        return None
    
    def _setup_env_friction(self):
        # This function changes the friction of ALL robot bodies to a single value 
        # for each OOD group (low or high).
        if self.env is None:
            print("[WARNING] No environment provided, skipping robot friction setup")
            return
        robot = self._get_robot()
        if robot is None:
            print("[WARNING] Could not find robot for robot friction")
            return

        # Use the CUDA/GPU tensor for environment indexing
        ood_env_ids = self.ood_env_ids.to(robot.device)
        num_ood = self.num_ood_envs
        half_point = num_ood // 2

        # Group 1: Low Friction (First half of OOD envs)
        low_friction_env_ids = ood_env_ids[:half_point]
        # Group 2: High Friction (Second half of OOD envs)
        high_friction_env_ids = ood_env_ids[half_point:]

        # --- Define Friction Values ---
        # [Static Friction, Dynamic Friction, Restitution]
        LOW_FRICTION_VALUE = 0.6
        HIGH_FRICTION_VALUE = 1.4
        RESTITUTION_VALUE = 0.0

        # These tensors represent the material property vector [mu_static, mu_dynamic, mu_restitution]
        LOW_FRICTION_MATERIAL = torch.tensor(
            [LOW_FRICTION_VALUE, LOW_FRICTION_VALUE, RESTITUTION_VALUE],
            device=robot.device, dtype=torch.float
        )
        HIGH_FRICTION_MATERIAL = torch.tensor(
            [HIGH_FRICTION_VALUE, HIGH_FRICTION_VALUE, RESTITUTION_VALUE],
            device=robot.device, dtype=torch.float
        )
        
        # 1. Retrieve the material buffer from the physics simulation
        # Shape is (num_envs, max_num_shapes_in_asset, 3)
        materials = robot.root_physx_view.get_material_properties().to(robot.device)

        # Get the number of shapes per body/link to correctly map indices.
        num_shapes_per_body = []
        try:
            for link_path in robot.root_physx_view.link_paths[0]:
                # Using private methods/attributes of the Articulation view for shape count
                link_physx_view = robot._physics_sim_view.create_rigid_body_view(link_path)
                num_shapes_per_body.append(link_physx_view.max_shapes)
        except Exception as e:
            # Fallback for assets where link_paths is not structured as expected
            print(f"[WARNING] Failed to parse num_shapes_per_body: {e}. Assuming single block.")
            num_shapes_per_body = [robot.root_physx_view.max_shapes]
            
        
        # 2. Loop through all bodies/links and assign the friction value to all shapes.
        start_idx = 0
        
        # Iterate through bodies/links
        for body_id in range(robot.num_bodies):
            # Determine the shape indices belonging to the current body
            end_idx = start_idx + num_shapes_per_body[body_id]
            
            # We need to set the material property for ALL shapes in this body.
            
            # --- Apply Low Friction to Group 1 ---
            if len(low_friction_env_ids) > 0:
                # Broadcast the LOW_FRICTION_MATERIAL [3] vector across the relevant shapes (start_idx:end_idx) 
                # for all environments in the low friction group.
                materials[low_friction_env_ids, start_idx:end_idx] = LOW_FRICTION_MATERIAL
            
            # --- Apply High Friction to Group 2 ---
            if len(high_friction_env_ids) > 0:
                # Broadcast the HIGH_FRICTION_MATERIAL [3] vector across the relevant shapes 
                # for all environments in the high friction group.
                materials[high_friction_env_ids, start_idx:end_idx] = HIGH_FRICTION_MATERIAL
                
            # Move to the start index of the next body
            start_idx = end_idx

        # 3. Apply the modified materials back to the simulation
        # Push the changes for the OOD environments back to the robot asset.
        robot.root_physx_view.set_material_properties(materials.cpu(), ood_env_ids.cpu())

        print(f"[INFO] Applied bimodal robot friction (emulating env friction): "
            f"{len(low_friction_env_ids)} low ({LOW_FRICTION_VALUE}), "
            f"{len(high_friction_env_ids)} high ({HIGH_FRICTION_VALUE})")

    def _setup_env_disturbance(self):
        # 50% is a push at the start of the episode, 50% is adding payload (10kg) to the robot base
        if self.env is None:
            print("[WARNING] No environment provided, skipping env disturbance setup")
            return
        robot = self._get_robot()
        if robot is None:
            print("[WARNING] Could not find robot for env disturbance")
            return
        if True:
            # --- Setup Indices and Split Groups ---
            ood_env_ids = self.ood_env_ids.cpu()
            num_ood = self.num_ood_envs
            half_point = num_ood // 2

            # Group 1: Envs for Initial Push (First half of OOD envs)
            push_env_ids = ood_env_ids[:half_point].cpu()

            # Group 2: Envs for Payload/Mass Change (Second half of OOD envs)
            payload_env_ids = ood_env_ids[half_point:].cpu()
            num_payload_envs = len(payload_env_ids)

            # --- 1. Vectorize Initial Push (Still requires a minimal loop for per-env writing) ---

            if len(push_env_ids) > 0:
                
                # Vectorized calculation of push velocity for all 'push' environments
                # 1. Random direction in x-y plane: [num_push_envs, 2]
                push_velocity = (torch.rand(len(push_env_ids), 2, device=self.device) * 2 - 1)
                # 2. Normalize: [num_push_envs, 2]
                norm = torch.norm(push_velocity, dim=1, keepdim=True)
                push_velocity = push_velocity / norm
                # 3. Random speed (0.05-0.4 m/s): [num_push_envs, 1]
                push_speed = torch.rand(len(push_env_ids), 1, device=self.device) * 0.35 + 0.05
                # 4. Final velocity (x, y components): [num_push_envs, 2]
                final_velocity = push_velocity * push_speed
                
                # Get the root state slices needed for the push
                root_states = robot.data.root_state_w[push_env_ids].clone()
                
                # Apply X and Y velocity components to the root states (indices 3 and 4)
                root_states[:, 3] = final_velocity[:, 0]  # x velocity
                root_states[:, 4] = final_velocity[:, 1]  # y velocity

                # Apply all changes at once for the push group
                # NOTE: Since you used write_root_pose_to_sim and write_root_velocity_to_sim in the loop,
                # we must ensure the batch version is correct for your environment's API.
                # The safest way is often to use the PhysX View API if available.
                
                # Option A: Vectorized Write (Preferred, if API supports writing subsets)
                # If the API doesn't support writing a subset of env_ids, this section needs to be modified.
                # Assuming the following functions accept subsets (or you update the full tensor):
                
                # 1. Update the full tensor
                robot.data.root_state_w[push_env_ids] = root_states 

                # 2. Write the changes for the push environments
                # NOTE: The original code wrote ALL 13 components, but only wrote a 7-component pose and a 6-component velocity.
                # We will write the full state change for the affected root bodies.
                robot.write_root_pose_to_sim(robot.data.root_state_w[:, :7]) 
                robot.write_root_velocity_to_sim(robot.data.root_state_w[:, 7:])

            # --- 2. Vectorize Payload/Mass Change (Executed ONLY ONCE) ---

            if len(payload_env_ids) > 0 and not self.physics_perturbed:
                
                payload_mass = 10.0  # 10kg payload
                body_id = 9 # torso_link body id (Target body index)
                
                # Get current masses and inertias for ALL environments
                masses = robot.root_physx_view.get_masses()
                inertias = robot.root_physx_view.get_inertias()
                
                # Store originals for OOD envs only (for reset)
                if self.original_masses is None:
                    self.original_masses = masses[self.ood_env_ids_cpu].clone()
                if self.original_inertias is None:
                    self.original_inertias = inertias[self.ood_env_ids_cpu].clone()

                # --- Apply Perturbations Vectorized ---
                
                # 1. Calculate scale factor ONLY for the target body and payload environments
                # Default mass of the target body [num_envs, 1] -> [num_payload_envs, 1]
                default_mass_payload_envs = self.original_masses[half_point:, body_id] 
                
                # Scale = (Original Mass + Payload Mass) / Original Mass. Shape: [num_payload_envs]
                scale_factor = (default_mass_payload_envs + payload_mass) / default_mass_payload_envs
                
                # 2. Apply Scale to Masses (Only affect the payload envs, only the target body)
                masses[payload_env_ids, body_id] = default_mass_payload_envs * scale_factor

                # 3. Apply Scale to Inertias
                default_inertia_payload_envs = self.original_inertias[half_point:, body_id]
                
                if inertias.dim() == 3: # Articulation: (num_envs, num_bodies, 9)
                    # Scale for inertia needs to be expanded [num_payload_envs, 1]
                    scale_factor_expanded = scale_factor.unsqueeze(-1)
                    
                    # Apply the scaling to the inertia matrix (9 components)
                    inertias[payload_env_ids, body_id] = default_inertia_payload_envs * scale_factor_expanded
                
                # NOTE: The 'else' block for Rigid object was complex and likely wrong; 
                # we focus on the articulation case which is typical for 'robot'.
                
                # 4. Set modified values for ALL environments (API requirement)
                all_env_ids = torch.arange(self.num_envs, device=self.device) # Use self.device
                
                robot.root_physx_view.set_masses(masses, all_env_ids.cpu())
                robot.root_physx_view.set_inertias(inertias, all_env_ids.cpu())
                
                self.physics_perturbed = True
                print(f"[INFO] Applied mass perturbations to {num_payload_envs} environments")

            print(f"[INFO] Env disturbance configured: {half_point} envs with push, {self.num_ood_envs - half_point} envs with payload")
        #except Exception as e:
        #    print(f"[WARNING] Failed to setup env disturbance: {e}")
            
    def _setup_init_state_perturbation(self):
        # 25% of env has joint pos perturbation
        # 25% has joint vel perturbation
        # 25% has body pose perturbation (just the z)
        # 25% has body orientation perturbation
        if self.env is None:
            print("[WARNING] No environment provided, skipping init state perturbation setup")
            return
        robot = self._get_robot()
        if robot is None:
            print("[WARNING] Could not find robot for init state perturbation")
            return
        
        try:
            env_ids = self.ood_env_ids
            joint_vel = robot.data.default_joint_vel[env_ids].clone()#[env_ids]
            joint_pos = robot.data.default_joint_pos[env_ids].clone()#[env_ids]
            robot_root_state = robot.data.default_root_state[env_ids].clone()#[env_ids] # [pos(3), ori(4), linvel(3), angvel(3)]

            num_ood = self.num_ood_envs
            third = num_ood // 3
            for idx in range(num_ood):
                if idx < third:
                    # Joint velocity perturbation
                    noise = (torch.rand_like(joint_vel[idx]) * 2 - 1) * 0.5
                    joint_vel[idx] += noise
                elif idx < 2 * third:
                    # Body pose perturbation (z only)
                    noise = (torch.rand(1, device=self.device).squeeze() * 0.4 + 0.1)
                    robot_root_state[idx, 2] += noise
                else:
                    # Body orientation perturbation (pitch and roll)
                    pitch_noise = (torch.rand((1,), device=self.device) * 2 - 1) * 0.5
                    roll_noise = (torch.rand((1,), device=self.device) * 2 - 1) * 0.5
                    current_ori = robot_root_state[idx, 3:7]
                    
                    import isaaclab.utils.math as math_utils
                    euler = list(math_utils.euler_xyz_from_quat(current_ori.unsqueeze(0)))
                    euler[0] += roll_noise
                    euler[1] += pitch_noise
                    new_ori = math_utils.quat_from_euler_xyz(*euler)
                    robot_root_state[idx, 3:7] = new_ori

            # Write back ONLY the OOD environments (shapes now match: [2048, 7] and [2048, 7])
            robot.write_root_pose_to_sim(robot_root_state[:, :7], env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids=env_ids)

        except IndexError as e:
            print(f"[WARNING] Failed to setup init state perturbation: {e}")

    def _setup_actuator_dynamics(self):
        """
        Setup actuator dynamics mismatch: modify torque limits, stiffness, or damping.
        Each OOD env gets one of: torque (x0.3-0.7 or x1.5-2.0), stiffness (x0.5-2.0), 
        damping (x0.5-2.0), or combined perturbation.
        
        CHANGE: Now affects 1-5 random motors OVERALL (across all groups).
        """
        if self.env is None:
            print("[WARNING] No environment provided, skipping actuator dynamics setup")
            return
        
        robot = self._get_robot()
        if robot is None or not hasattr(robot, 'actuators'):
            print("[WARNING] Could not find robot actuators for dynamics change")
            return
        
        if len(robot.actuators) == 0:
            print("[WARNING] No actuators found on robot")
            return
        
        try:
            # Randomly assign perturbation type to each OOD env
            # 0: torque_low, 1: stiffness, 2: damping, 3: motor_dead
            perturbation_types = torch.randint(0, 4, (self.num_ood_envs,), device=self.device)
            self.actuator_perturbation_type = perturbation_types
            
            # --- Step 1: Initialize all scales to 1.0 (Identity) and build global map ---
            self.actuator_scale_factors = {}
            global_to_local_map = [] # List of (actuator_name, local_index)
            total_joints = 0
            
            for act_name, actuator in robot.actuators.items():
                num_joints = actuator.effort_limit.shape[-1]
                
                # Create storage for this group
                self.actuator_scale_factors[act_name] = {
                    'effort': torch.ones(self.num_ood_envs, num_joints, device=self.device),
                    'stiffness': torch.ones(self.num_ood_envs, num_joints, device=self.device),
                    'damping': torch.ones(self.num_ood_envs, num_joints, device=self.device),
                }
                
                # Add to global map
                for local_idx in range(num_joints):
                    global_to_local_map.append((act_name, local_idx))
                    
                total_joints += num_joints
                
            # --- Step 2: Apply perturbations to random subset of GLOBAL indices ---
            
            for i in range(self.num_ood_envs):
                ptype = perturbation_types[i].item()
                
                # 1. Pick 1-5 random unique GLOBAL indices
                n_affected = torch.randint(1, min(6, total_joints + 1), (1,)).item()
                global_indices = torch.randperm(total_joints, device=self.device)[:n_affected]
                
                for global_idx in global_indices:
                    # Resolve global index to specific actuator group and local index
                    act_name, local_idx = global_to_local_map[global_idx.item()]
                    
                    if ptype == 0:  # Low torque (x0.4-0.8)
                        scale = torch.rand(1, device=self.device).item() * 0.4 + 0.4
                        self.actuator_scale_factors[act_name]['effort'][i, local_idx] = scale
                        
                    elif ptype == 1:  # Stiffness variation
                        if torch.rand(1).item() > 0.5:
                            scale = torch.rand(1).item() * 0.4 + 0.1 # Low stiffness
                        else:
                            scale = torch.rand(1).item() * 0.4 + 1.1 # High stiffness
                        self.actuator_scale_factors[act_name]['stiffness'][i, local_idx] = scale
                        
                    elif ptype == 2:  # Damping variation
                        if torch.rand(1).item() > 0.5:
                            scale = torch.rand(1).item() * 0.4 + 0.1
                        else:
                            scale = torch.rand(1).item() * 0.4 + 1.1
                        self.actuator_scale_factors[act_name]['damping'][i, local_idx] = scale
                        
                    elif ptype == 3:  # Motor died (zero torque)
                        self.actuator_scale_factors[act_name]['effort'][i, local_idx] = 0.0
            
            # Count perturbation types
            type_counts = [(perturbation_types == i).sum().item() for i in range(4)]
            print(f"[INFO] Actuator dynamics configured ({len(robot.actuators)} groups, {total_joints} total joints):")
            print(f"       [SPARSE MODE] Affecting 1-5 random motors OVERALL per environment")
            print(f"       Low torque: {type_counts[0]}, Stiffness: {type_counts[1]}")
            print(f"       Damping: {type_counts[2]}, Motor dead: {type_counts[3]}")
            
        except Exception as e:
            print(f"[WARNING] Failed to setup actuator dynamics: {e}")
            import traceback
            traceback.print_exc()
    
    def _apply_actuator_perturbation(self):
        """Apply actuator dynamics changes to all actuator groups."""
        #if self.actuator_scale_factors is None or self.env is None:
        #    return
        
        robot = self._get_robot()
        #if robot is None or not hasattr(robot, 'actuators'):
        #    print("[WARNING] Could not find robot actuators for perturbation")
        #    return
        
        try:
            for act_name, actuator in robot.actuators.items():
                #print(actuator.effort_limit)
                #print(act_name, self.actuator_scale_factors.keys())
                #if act_name not in self.actuator_scale_factors:
                #    continue
                
                scales = self.actuator_scale_factors[act_name]
                
                # Store originals for this actuator
                if act_name not in self.original_effort_limit:
                    self.original_effort_limit[act_name] = actuator.effort_limit[self.ood_env_ids].clone()
                if act_name not in self.original_stiffness and hasattr(actuator, 'stiffness'):
                    self.original_stiffness[act_name] = actuator.stiffness[self.ood_env_ids].clone()
                if act_name not in self.original_damping and hasattr(actuator, 'damping'):
                    self.original_damping[act_name] = actuator.damping[self.ood_env_ids].clone()
                
                # Apply perturbations
                effort_scales = scales['effort']
                stiffness_scales = scales['stiffness']
                damping_scales = scales['damping']
                
                # Scale effort limits
                actuator.effort_limit[self.ood_env_ids,:] = self.original_effort_limit[act_name] * effort_scales
                
                # Scale stiffness if available
                if hasattr(actuator, 'stiffness') and act_name in self.original_stiffness:
                    actuator.stiffness[self.ood_env_ids] = self.original_stiffness[act_name] * stiffness_scales
                    robot.write_joint_stiffness_to_sim(
                        actuator.stiffness[self.ood_env_ids], 
                        joint_ids=actuator.joint_indices, 
                        env_ids=self.ood_env_ids
                    )
                    
                # Scale damping if available
                if hasattr(actuator, 'damping') and act_name in self.original_damping:
                    actuator.damping[self.ood_env_ids] = self.original_damping[act_name] * damping_scales
                    robot.write_joint_damping_to_sim(
                        actuator.damping[self.ood_env_ids],
                        joint_ids=actuator.joint_indices,
                        env_ids=self.ood_env_ids
                    )
                    
                robot.write_joint_effort_limit_to_sim(
                    actuator.effort_limit[self.ood_env_ids], 
                    joint_ids=actuator.joint_indices, 
                    env_ids=self.ood_env_ids
                )

            self.physics_perturbed = True
            #print(f"[INFO] Applied actuator perturbations to {self.num_ood_envs} environments ({len(robot.actuators)} groups)")
            
        except Exception as e:
            print(f"[WARNING] Failed to apply actuator perturbation: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_sensor_drift(self):
        """Setup sensor drift: gradual drift on 1-5 random observation indices."""
        # Each OOD env affects 1-5 random obs indices
        num_affected = torch.randint(1, 2, (self.num_ood_envs,), device=self.device)
        
        # Create mask of affected indices per env
        self.affected_obs_indices = torch.zeros(
            self.num_ood_envs, self.obs_dim, dtype=torch.bool, device=self.device
        )
        for i in range(self.num_ood_envs):
            n = num_affected[i].item()
            indices = torch.randperm(self.obs_dim, device=self.device)[:n]
            self.affected_obs_indices[i, indices] = True
        
        # Drift rate: 0.005 to 0.05 per step (random direction)
        self.drift_rate = (torch.rand(self.num_ood_envs, self.obs_dim, device=self.device) * 0.0045 + 0.0005)
        self.drift_rate *= (torch.randint(0, 2, (self.num_ood_envs, self.obs_dim), device=self.device) * 2 - 1).float()
        self.drift_rate *= self.affected_obs_indices.float()
        
        self.drift_accumulator = torch.zeros(self.num_ood_envs, self.obs_dim, device=self.device)
    
    def _setup_sensor_zero(self):
        """Setup sensor zero: 1-5 random observation indices output zero."""
        num_affected = torch.randint(1, 6, (self.num_ood_envs,), device=self.device)
        
        self.affected_obs_indices = torch.zeros(
            self.num_ood_envs, self.obs_dim, dtype=torch.bool, device=self.device
        )
        for i in range(self.num_ood_envs):
            n = num_affected[i].item()
            indices = torch.randperm(self.obs_dim, device=self.device)[:n]
            self.affected_obs_indices[i, indices] = True
    
    def _setup_frozen_sensor(self):
        """Setup frozen sensor: 1-5 random observation indices output constant value."""
        num_affected = torch.randint(1, 6, (self.num_ood_envs,), device=self.device)
        
        self.affected_obs_indices = torch.zeros(
            self.num_ood_envs, self.obs_dim, dtype=torch.bool, device=self.device
        )
        for i in range(self.num_ood_envs):
            n = num_affected[i].item()
            indices = torch.randperm(self.obs_dim, device=self.device)[:n]
            self.affected_obs_indices[i, indices] = True
        
        # Store frozen values (initialized later on first step)
        self.frozen_values = torch.zeros(self.num_ood_envs, self.obs_dim, device=self.device)
        self.frozen_initialized = torch.zeros(self.num_ood_envs, dtype=torch.bool, device=self.device)

    def _setup_scale(self, factor: float):
        """Setup scaling: scale 1-10 random observation indices by factor."""
        num_affected = torch.randint(1, 11, (self.num_ood_envs,), device=self.device)
        
        self.affected_obs_indices = torch.zeros(
            self.num_ood_envs, self.obs_dim, dtype=torch.bool, device=self.device
        )
        for i in range(self.num_ood_envs):
            n = num_affected[i].item()
            indices = torch.randperm(self.obs_dim, device=self.device)[:n]
            self.affected_obs_indices[i, indices] = True
        
        self.scale_factor = factor
    
    def _setup_obs_swap(self):
        """Setup observation swap: swap 1-2 random pairs of obs indices."""
        num_pairs = torch.randint(1, 3, (self.num_ood_envs,), device=self.device)
        
        # Store swap pairs: [num_ood_envs, max_pairs, 2]
        max_pairs = 2
        self.swap_pairs = torch.zeros(
            self.num_ood_envs, max_pairs, 2, dtype=torch.long, device=self.device
        )
        self.num_swap_pairs = num_pairs
        
        for i in range(self.num_ood_envs):
            n = num_pairs[i].item()
            # Pick 2*n unique indices and pair them
            indices = torch.randperm(self.obs_dim, device=self.device)[:2*n]
            for j in range(n):
                self.swap_pairs[i, j, 0] = indices[2*j]
                self.swap_pairs[i, j, 1] = indices[2*j + 1]
    
    def _setup_action_swap(self):
        """Setup action swap: swap 1-2 random pairs of action indices."""
        num_pairs = torch.randint(1, 3, (self.num_ood_envs,), device=self.device)
        
        max_pairs = 2
        self.swap_pairs = torch.zeros(
            self.num_ood_envs, max_pairs, 2, dtype=torch.long, device=self.device
        )
        self.num_swap_pairs = num_pairs
        
        for i in range(self.num_ood_envs):
            n = num_pairs[i].item()
            indices = torch.randperm(self.action_dim, device=self.device)[:2*n]
            for j in range(n):
                self.swap_pairs[i, j, 0] = indices[2*j]
                self.swap_pairs[i, j, 1] = indices[2*j + 1]
    
    def _setup_noise(self):
        """Setup noise: add Gaussian noise with std 0.05 to 0.25 on 1-5 obs."""
        num_affected = torch.randint(1, 6, (self.num_ood_envs,), device=self.device)
        
        self.affected_obs_indices = torch.zeros(
            self.num_ood_envs, self.obs_dim, dtype=torch.bool, device=self.device
        )
        for i in range(self.num_ood_envs):
            n = num_affected[i].item()
            indices = torch.randperm(self.obs_dim, device=self.device)[:n]
            self.affected_obs_indices[i, indices] = True
        
        # Random noise std per env (0.05 to 0.25)
        #self.noise_std = torch.rand(self.num_ood_envs, 1, device=self.device) * 0.2 + 0.05
        self.noise_limit = torch.rand(self.num_ood_envs, 1, device=self.device) * 0.1 + 0.05
    
    def _setup_latency_offset(self):
        """Setup constant latency: 20-100ms delay (1-5 steps at 50Hz) on 1-5 obs."""
        # Random latency 1-5 steps per env
        self.latency_steps = torch.randint(1, 6, (self.num_ood_envs,), device=self.device)
        max_latency = 5
        
        # Buffer to store past observations: [num_ood_envs, max_latency, obs_dim]
        self.latency_buffer = torch.zeros(
            self.num_ood_envs, max_latency, self.obs_dim, device=self.device
        )
        self.latency_buffer_idx = 0
        
        # Change: Limit affected indices to 1-5 (capped by obs_dim)
        num_affected = torch.randint(1, min(6, self.obs_dim + 1), (self.num_ood_envs,), device=self.device)
        
        self.affected_obs_indices = torch.zeros(
            self.num_ood_envs, self.obs_dim, dtype=torch.bool, device=self.device
        )
        for i in range(self.num_ood_envs):
            n = num_affected[i].item()
            indices = torch.randperm(self.obs_dim, device=self.device)[:n]
            self.affected_obs_indices[i, indices] = True
    
    def _setup_latency_slow(self):
        """Setup slow update: update observations every 2-10 steps (x2-10 slower)."""
        self.latency_counter = torch.zeros(self.num_ood_envs, dtype=torch.long, device=self.device)
        self.latency_update_interval = torch.randint(2, 11, (self.num_ood_envs,), device=self.device)  # Update every 2-10 steps
        self.last_obs = None
        
        # Which obs indices to apply slow update to
        max_indices = min(5, self.obs_dim)
        num_affected = torch.randint(1, max_indices + 1, (self.num_ood_envs,), device=self.device)
        
        self.affected_obs_indices = torch.zeros(
            self.num_ood_envs, self.obs_dim, dtype=torch.bool, device=self.device
        )
        for i in range(self.num_ood_envs):
            n = num_affected[i].item()
            indices = torch.randperm(self.obs_dim, device=self.device)[:n]
            self.affected_obs_indices[i, indices] = True
    
    def activate(self):
        """Activate OOD injection (called when OOD should start)."""
        self.ood_active = True
        
        # Apply physics perturbations if this is a physics-based category
        if self.category == OODCategory.ACTUATOR_DYNAMICS:
            self._apply_actuator_perturbation()
    
    def inject_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Inject OOD perturbations into observations.
        Only modifies the second half of environments.
        
        Args:
            obs: [num_envs, obs_dim] observation tensor
            
        Returns:
            Modified observation tensor
        """
        if self.category == OODCategory.NONE or not self.ood_active:
            return obs
        
        if self.category == OODCategory.ACTION_SWAP:
            # Action swap doesn't modify observations directly
            return obs
        
        obs = obs.clone()
        ood_obs = obs[self.num_control_envs:]  # Second half
        
        if self.category == OODCategory.SENSOR_DRIFT:
            self.drift_accumulator += self.drift_rate
            ood_obs += self.drift_accumulator
            
        elif self.category == OODCategory.SENSOR_ZERO:
            ood_obs[self.affected_obs_indices] = 0.0
            
        elif self.category in [OODCategory.SCALE_HALF, OODCategory.SCALE_DOUBLE]:
            # Apply scaling only to affected indices
            mask = self.affected_obs_indices.float()
            ood_obs = ood_obs * (1 - mask) + ood_obs * self.scale_factor * mask
            
        elif self.category == OODCategory.FROZEN_SENSOR:
            for i in range(self.num_ood_envs):
                if not self.frozen_initialized[i]:
                    self.frozen_values[i] = ood_obs[i].clone()
                    self.frozen_initialized[i] = True
                mask = self.affected_obs_indices[i]
                ood_obs[i, mask] = self.frozen_values[i, mask]

        elif self.category == OODCategory.OBS_SWAP:
            for i in range(self.num_ood_envs):
                n = self.num_swap_pairs[i].item()
                for j in range(n):
                    idx1 = self.swap_pairs[i, j, 0].item()
                    idx2 = self.swap_pairs[i, j, 1].item()
                    ood_obs[i, idx1], ood_obs[i, idx2] = ood_obs[i, idx2].clone(), ood_obs[i, idx1].clone()
                    
        elif self.category == OODCategory.NOISE:
            noise = (torch.rand_like(ood_obs) * 2 - 1) * self.noise_limit
            noise *= self.affected_obs_indices.float()
            ood_obs += noise
            #noise = torch.randn_like(ood_obs) * self.noise_std
            #noise *= self.affected_obs_indices.float()
            #ood_obs += noise
            
        elif self.category == OODCategory.LATENCY_OFFSET:
            # Store current obs in buffer
            self.latency_buffer[:, self.latency_buffer_idx] = ood_obs.clone()
            
            # Retrieve delayed obs
            for i in range(self.num_ood_envs):
                delay = self.latency_steps[i].item()
                delayed_idx = (self.latency_buffer_idx - delay) % 5
                # Only apply to affected indices
                mask = self.affected_obs_indices[i]
                ood_obs[i, mask] = self.latency_buffer[i, delayed_idx, mask]
            
            self.latency_buffer_idx = (self.latency_buffer_idx + 1) % 5
            
        elif self.category == OODCategory.LATENCY_SLOW:
            if self.last_obs is None:
                self.last_obs = ood_obs.clone()
            
            self.latency_counter += 1
            update_mask = (self.latency_counter >= self.latency_update_interval)
            
            for i in range(self.num_ood_envs):
                if update_mask[i]:
                    self.last_obs[i] = ood_obs[i].clone()
                    self.latency_counter[i] = 0
                else:
                    # Keep old values for affected indices
                    mask = self.affected_obs_indices[i]
                    ood_obs[i, mask] = self.last_obs[i, mask]

        obs[self.num_control_envs:] = ood_obs
        return obs
    
    def inject_action(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Inject OOD perturbations into actions (only for ACTION_SWAP category).
        
        Args:
            actions: [num_envs, action_dim] action tensor
            
        Returns:
            Modified action tensor
        """
        if self.category != OODCategory.ACTION_SWAP or not self.ood_active:
            return actions
        
        actions = actions.clone()
        ood_actions = actions[self.num_control_envs:]
        
        for i in range(self.num_ood_envs):
            n = self.num_swap_pairs[i].item()
            for j in range(n):
                idx1 = self.swap_pairs[i, j, 0].item()
                idx2 = self.swap_pairs[i, j, 1].item()
                ood_actions[i, idx1], ood_actions[i, idx2] = ood_actions[i, idx2].clone(), ood_actions[i, idx1].clone()
        
        actions[self.num_control_envs:] = ood_actions
        return actions
    
    def get_config_summary(self) -> Dict:
        """Return a summary of current OOD configuration."""
        summary = {
            "category": CATEGORY_NAMES[self.category],
            "num_ood_envs": self.num_ood_envs,
            "num_control_envs": self.num_control_envs,
        }
        
        if self.affected_obs_indices is not None and self.affected_obs_indices.numel() > 0:
            affected_counts = self.affected_obs_indices.sum(dim=1).float()
            if affected_counts.numel() > 0:
                summary["affected_obs_mean"] = affected_counts.mean().item()
                summary["affected_obs_min"] = affected_counts.min().item()
                summary["affected_obs_max"] = affected_counts.max().item()
        
        if self.latency_steps is not None and self.latency_steps.numel() > 0:
            summary["latency_steps_mean"] = self.latency_steps.float().mean().item()
            summary["latency_steps_min"] = self.latency_steps.min().item()
            summary["latency_steps_max"] = self.latency_steps.max().item()
        
        if self.noise_limit is not None and self.noise_limit.numel() > 0:
            summary["noise_limit_mean"] = self.noise_limit.mean().item()
        
        # Physics perturbation summaries
        if self.mass_scale_factors is not None:
            num_half = (self.mass_scale_factors < 1.0).sum().item()
            num_double = (self.mass_scale_factors > 1.0).sum().item()
            summary["mass_x0.5_count"] = num_half
            summary["mass_x2.0_count"] = num_double
            if self.affected_body_ids:
                body_counts = [len(b) for b in self.affected_body_ids]
                summary["affected_bodies_mean"] = np.mean(body_counts)
                summary["affected_bodies_min"] = min(body_counts)
                summary["affected_bodies_max"] = max(body_counts)
        
        if self.actuator_perturbation_type is not None:
            type_counts = [(self.actuator_perturbation_type == i).sum().item() for i in range(5)]
            summary["actuator_low_torque"] = type_counts[0]
            summary["actuator_high_torque"] = type_counts[1]
            summary["actuator_stiffness"] = type_counts[2]
            summary["actuator_damping"] = type_counts[3]
            summary["actuator_combined"] = type_counts[4]
        
        return summary

# ==================================================================================
# HYBRID DETECTOR (Range + rapt)
# ==================================================================================

class SimpleRangeDetector:
    def __init__(self, device: str = 'cuda', margin_percent: float = 0.05, min_epsilon: float = 0.0):
        """
        Args:
            margin_percent (float): Buffer size as a percentage of the data range (e.g. 0.05 = 5%).
                                    This scales automatically for every dimension.
            min_epsilon (float):    A minimum floor for the buffer to handle constant values (range=0).
        """
        self.device = device
        self.margin_percent = margin_percent
        self.min_epsilon = min_epsilon

        # Track RAW min/max (no padding yet)
        self.raw_min = torch.full((1,), float('inf'), device=device)
        self.raw_max = torch.full((1,), float('-inf'), device=device)
        self.obs_dim = 0
        
        # Finalized bounds used for checking
        self.lower_bound = None
        self.upper_bound = None

    def update_bounds(self, obs: torch.Tensor):
        """
        Expand raw bounds based on new valid data.
        Does NOT apply padding yet.
        """
        if obs.device != self.device: obs = obs.to(self.device)
        
        if self.obs_dim == 0:
            self.obs_dim = obs.shape[-1]
            self.raw_min = torch.min(obs, dim=0)[0]
            self.raw_max = torch.max(obs, dim=0)[0]
        else:
            current_min = torch.min(obs, dim=0)[0]
            current_max = torch.max(obs, dim=0)[0]
            
            self.raw_min = torch.min(self.raw_min, current_min)
            self.raw_max = torch.max(self.raw_max, current_max)
            
        # Re-calculate bounds immediately so we can test during training if needed
        self._compute_final_bounds()

    def _compute_final_bounds(self):
        """Calculates the padded bounds based on the data range."""
        # Calculate the range (spread) of each dimension
        data_range = self.raw_max - self.raw_min
        
        # Calculate dynamic buffer: % of range + minimum floor
        buffer = (data_range * self.margin_percent) + 1e-6
        
        self.lower_bound = self.raw_min - buffer
        self.upper_bound = self.raw_max + buffer

    @torch.no_grad()
    def check_batch(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns boolean tensor: True if anomaly, False if normal.
        """
        if self.lower_bound is None:
            raise RuntimeError("Detector not calibrated. Call update_bounds first.")
            
        if obs.device != self.device: obs = obs.to(self.device)
        
        # Check violations
        # [Batch, Dims]
        lower_violation = (obs < self.lower_bound)
        upper_violation = (obs > self.upper_bound)
        
        # If ANY dimension is out of bounds, the sample is anomalous
        is_anomaly = (lower_violation | upper_violation).any(dim=1)
        return is_anomaly

class HybridAnomalyDetector:
    """
    Combines Simple Range Detection with Configurable Autoencoder.
    """
    def __init__(self, args, obs_dim: int, action_dim: int, device: str = 'cuda'):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Thresholds (Separate for Max and Mean)
        # Using [Max_Thresh, Mean_Thresh]
        self.rapt_threshold = [torch.ones(obs_dim, device=device)*3, 3.0]        
        self.ood_detect_type = args.ood_detect_type
        self.use_temporal = args.use_temporal
        self.use_probabilistic = args.use_probabilistic
        self.reconstruction_type = args.reconstruction_type
        
        # Temporal State
        self.hidden_state = None
        
        # Dynamics Model Buffers (for s_t, a_t -> s_{t+1})
        self.prev_obs = None
        self.prev_act = None

        # 1. Initialize Range Detector
        self.range_detector = SimpleRangeDetector(device=device, margin_percent=args.margin_percent, min_epsilon=args.padding_epsilon)
        
        # 2. Check Config to see if this is a Dynamics Model
        # Load experiment config if available to auto-detect mode
        config_path = os.path.join(os.path.dirname(args.rapt_checkpoint), 'experiment_config.json')
        self.is_dynamics_model = False
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                exp_cfg = json.load(f)
                self.is_dynamics_model = exp_cfg.get('train_dynamics', False)
        
        print(f"[INFO] Loading Model from {args.rapt_checkpoint}")
        print(f"       Mode: {'DYNAMICS' if self.is_dynamics_model else 'AUTOENCODER'}")
        print(f"       Config: {args.reconstruction_type.upper()} | Res={args.use_residual} | Prob={args.use_probabilistic} | Temp={args.use_temporal}")
        
        self.model = UniversalModel(obs_dim=obs_dim, action_dim=action_dim, args=args, train_dynamics=self.is_dynamics_model)
        
        checkpoint = torch.load(args.rapt_checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.to(device).eval()
        
        # Load Stats
        checkpoint_dir = os.path.dirname(args.rapt_checkpoint)
        stats_path = os.path.join(checkpoint_dir, 'obs_stats.h5')
        if os.path.exists(stats_path):
            with h5py.File(stats_path, 'r') as f:
                self.obs_mean = torch.from_numpy(f['mean'][:]).float().to(device)
                self.obs_std = torch.from_numpy(f['std'][:]).float().to(device)
        else:
            # Fallback stats if not found
            self.obs_mean = torch.zeros(obs_dim, device=device)
            self.obs_std = torch.ones(obs_dim, device=device)
            assert False, f"Obs stats file not found at {stats_path}"

    def reset(self):
        """Reset temporal state for new episode."""
        self.hidden_state = None
        self.prev_obs = None
        self.prev_act = None

    def update_buffer(self, obs, action):
        """Called AFTER the environment step to store data for the NEXT prediction."""
        self.prev_obs = obs.clone()
        self.prev_act = action.clone()

    @torch.no_grad()
    def calibrate(self, obs: torch.Tensor):
        if obs.device != self.device: obs = obs.to(self.device)
        self.range_detector.update_bounds(obs)
        _, losses, score_mean = self._get_model_score(obs)
        return losses, score_mean

    @torch.no_grad()
    def _get_model_score(self, obs):
        obs_norm = (obs - self.obs_mean) / self.obs_std
        
        # Forward pass (handles temporal state internally via self.hidden_state if needed, 
        # BUT self.model.forward returns hidden. We need to manage it.)
        # If temporal, we update hidden state.
        
        if self.is_dynamics_model:
            # Dynamics Mode: Predict s_{t} using s_{t-1}, a_{t-1}
            if self.prev_obs is None:
                num_envs = obs.shape[0]
                is_high = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                # First step: Cannot predict. Return zero score.
                # Update hidden state with dummy pass to keep GRU synced if needed
                return is_high, torch.zeros((num_envs, self.obs_dim), device=self.device), torch.zeros(num_envs, device=self.device)
            
            prev_obs_norm = (self.prev_obs - self.obs_mean) / self.obs_std
            output, _, new_hidden = self.model(prev_obs_norm, actions=self.prev_act, hidden=self.hidden_state)
        else:
            # Autoencoder Mode: Reconstruct s_{t} using s_{t}
            output, _, new_hidden = self.model(obs_norm, hidden=self.hidden_state)
        
        if self.use_temporal:
            self.hidden_state = new_hidden
        
        # Calculate Error/Z-Score
        if self.use_probabilistic:
            mu = output[:, :self.obs_dim]
            log_var = torch.clamp(output[:, self.obs_dim:], min=-6, max=6)
            sigma = torch.exp(0.5 * log_var)
            
            # Reconstruction Error (Probabilistic)
            # z = |x - mu| / sigma
            error = mu - obs_norm
            loss = (error ** 2) / (2 * sigma ** 2) + 0.5 * log_var + 0.5 * np.log(2 * np.pi)
            z_scores_per_dim = torch.abs(error / (sigma + 1e-8))

        else:
            # Deterministic MSE
            # z = (x - x_hat)^2 ... actually standard Z-score logic requires distribution
            # For MSE models, we typically use Squared Error as the "Score"
            # But to keep API consistent with Probabilistic, let's treat it as error magnitude
            mu = output
            error = mu - obs_norm
            z_scores_per_dim = error ** 2 # MSE per dim
            loss = z_scores_per_dim 

            # NOTE: Thresholding MSE is harder than Z-score. 
            # Calibration will handle the scale difference automatically.

        # Masking Logic for Score Aggregation
        # If 'masked' recon type, we usually care MOST about the masked parts.
        # But during inference for OOD, global inconsistency matters.
        # Simple strategy: Use all dimensions.
        score_mean = z_scores_per_dim.mean(dim=1)

        if self.ood_detect_type == 'both':
            # Restore Original Logic:
            # Mean > 3.0 (Global Statistical Anomaly)
            # OR
            # Max > Threshold (Specific Joint Failure, Calibrated)
            if self.use_probabilistic:
                is_high = score_mean > self.rapt_threshold[1]
            else:
                is_high = score_mean > self.rapt_threshold[1]

            is_high = is_high | (torch.sum(loss.to(self.device) > self.rapt_threshold[0].to(self.device), dim=1) > 0)
            final_score = score_mean  # For reporting 
            
        elif self.ood_detect_type == 'max':
            final_score = (torch.sum(loss.to(self.device) > self.rapt_threshold[0].to(self.device), dim=1)).float() # loss > rapt_threshold[0] -> [N,O] 
            is_high = final_score > 0
        else:  # 'mean'
            final_score = score_mean
            if self.use_probabilistic:
                is_high = final_score > self.rapt_threshold[1]
            else:
                is_high = final_score > self.rapt_threshold[1]
            
        return is_high, loss, score_mean.to(self.device)

    def reset_envs(self, env_ids: torch.Tensor):
        # Clear per-env history/buffers so next step doesn't compare against stale data.
        if self.prev_obs is not None:
            self.prev_obs[env_ids] = 0
        if self.prev_act is not None:
            self.prev_act[env_ids] = 0
        if self.hidden_state is not None:
            self.hidden_state[:, env_ids, :] = 0  # adjust dims to your actual layout

    @torch.no_grad()
    def check_batch(self, obs: torch.Tensor):
        if obs.device != self.device: obs = obs.to(self.device)
        is_range = self.range_detector.check_batch(obs) 
        is_model, losses, score_mean = self._get_model_score(obs)
        is_anomaly = is_range | is_model
        return is_anomaly, is_range, is_model, losses, score_mean

# ==================================================================================
# RESULTS TRACKING
# ==================================================================================

@dataclass
class EnvResult:
    env_idx: int
    is_ood_env: bool 
    detected: bool = False
    detected_by_range: bool = False
    detected_by_rapt: bool = False
    detection_step: int = -1 
    max_score: float = -float('inf')
    max_score_model_only: float = -float('inf')

# ==================================================================================
# ENHANCED RESULTS TRACKING WITH ROC METRICS
# ==================================================================================

@dataclass
class EpisodeResult:
    category: str
    category_idx: int
    episode_length: int
    ood_start_step: int
    env_results: List[EnvResult] = field(default_factory=list)
    
    # Existing metrics...
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    padd: float = 0.0
    tpr: float = 0.0
    fpr: float = 0.0
    tnr: float = 0.0
    fnr: float = 0.0
    avg_detection_step_tp: float = -1
    tp_range_only: int = 0
    tp_rapt_only: int = 0
    tp_both: int = 0
    tpr_range: float = 0.0
    fpr_range: float = 0.0
    tpr_rapt: float = 0.0
    fpr_rapt: float = 0.0
    
    # NEW: ROC-related data (store raw scores for later AUROC calculation)
    scores_hybrid: List[float] = field(default_factory=list)  # Combined score
    scores_model_only: List[float] = field(default_factory=list)  # Model-only score
    labels: List[int] = field(default_factory=list)  # Ground truth (1=OOD, 0=Normal)
    labels_hybrid: List[int] = field(default_factory=list)  # 1 if range OR model detected
    labels_model_only: List[int] = field(default_factory=list)  # 1 if model detected (ignoring range)

    # NEW: Computed ROC metrics
    auroc_hybrid: float = 0.0
    auroc_model_only: float = 0.0
    tpr_at_fpr_005_hybrid: float = 0.0  # TPR when FPR = 0.5%
    tpr_at_fpr_005_model: float = 0.0
    
    # NEW: Timing data
    detection_latencies_ms: List[float] = field(default_factory=list)
    avg_detection_latency_ms: float = 0.0

    def compute_metrics(self):
        """Enhanced to compute existing + ROC metrics."""
        # --- EXISTING LOGIC (unchanged) ---
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.tp_range_only = 0
        self.tp_rapt_only = 0
        self.tp_both = 0
        
        tp_range = fp_range = tn_range = fn_range = 0
        tp_rapt = fp_rapt = tn_rapt = fn_rapt = 0
        tp_steps = []
        padd_accum = 0.0
        ood_total_count = 0
        control_total_count = 0
        
        for r in self.env_results:
            is_ood = r.is_ood_env
            
            if is_ood:
                ood_total_count += 1
                if r.detected:
                    self.true_positives += 1
                    tp_steps.append(r.detection_step)
                    delay = max(0, r.detection_step - self.ood_start_step)
                    padd_accum += delay
                    
                    if r.detected_by_range and r.detected_by_rapt:
                        self.tp_both += 1
                    elif r.detected_by_range:
                        self.tp_range_only += 1
                    elif r.detected_by_rapt:
                        self.tp_rapt_only += 1
                else:
                    self.false_negatives += 1
                    padd_accum += self.episode_length
            else:
                control_total_count += 1
                if r.detected:
                    self.false_positives += 1
                else:
                    self.true_negatives += 1
            
            # Range detector breakdown
            if is_ood:
                if r.detected_by_range: tp_range += 1
                else: fn_range += 1
            else:
                if r.detected_by_range: fp_range += 1
                else: tn_range += 1
            
            # Model detector breakdown
            if is_ood:
                if r.detected_by_rapt: tp_rapt += 1
                else: fn_rapt += 1
            else:
                if r.detected_by_rapt: fp_rapt += 1
                else: tn_rapt += 1
        
        # PADD
        self.padd = padd_accum / ood_total_count if ood_total_count > 0 else 0.0
        
        # Detection latency
        if tp_steps:
            delays = [max(0, s - self.ood_start_step) for s in tp_steps]
            self.avg_detection_step_tp = float(np.mean(delays))
        else:
            self.avg_detection_step_tp = 0.0
        
        # Standard metrics
        self.tpr = self.true_positives / ood_total_count if ood_total_count > 0 else 0.0
        self.fnr = self.false_negatives / ood_total_count if ood_total_count > 0 else 0.0
        self.fpr = self.false_positives / control_total_count if control_total_count > 0 else 0.0
        self.tnr = self.true_negatives / control_total_count if control_total_count > 0 else 0.0
        
        # Split metrics
        total_ood = tp_range + fn_range
        total_control = fp_range + tn_range
        self.tpr_range = tp_range / total_ood if total_ood > 0 else 0.0
        self.fpr_range = fp_range / total_control if total_control > 0 else 0.0
        self.tpr_rapt = tp_rapt / total_ood if total_ood > 0 else 0.0
        self.fpr_rapt = fp_rapt / total_control if total_control > 0 else 0.0
        
        # --- NEW: AUROC CALCULATION ---
        self._compute_roc_metrics()
        
        # --- NEW: AVERAGE DETECTION LATENCY ---
        if self.detection_latencies_ms:
            self.avg_detection_latency_ms = float(np.mean(self.detection_latencies_ms))
        else:
            self.avg_detection_latency_ms = 0.0
    
    def _compute_roc_metrics(self):
        """
        Compute AUROC and TPR @ FPR=0.5% for both hybrid and model-only.
        This operates at the EPISODE level (one label per environment).
        """
        if len(self.labels) == 0 or len(self.scores_hybrid) == 0:
            print(f"[WARNING] No scores/labels for {self.category}, skipping ROC.")
            return
        
        labels_hybrid = np.array(self.labels_hybrid)
        labels_model = np.array(self.labels_model_only)
        scores_hybrid = np.array(self.scores_hybrid)
        scores_model = np.array(self.scores_model_only)

        
        # --- DATA CLEANING: Remove NaN/Inf ---
        def clean_data(scores, labels, name=""):
            """Remove samples with NaN or Inf scores."""
            # Find valid indices (finite scores)
            valid_mask = np.isfinite(scores)
            
            if not np.all(valid_mask):
                n_invalid = np.sum(~valid_mask)
                n_inf = np.sum(np.isinf(scores))
                n_nan = np.sum(np.isnan(scores))
                print(f"[WARNING] {name} for {self.category}: Removing {n_invalid} invalid samples "
                    f"({n_nan} NaN, {n_inf} Inf)")
            
            # Filter both scores and labels
            clean_scores = scores[valid_mask]
            clean_labels = labels[valid_mask]
            
            # Check if we have enough data left
            if len(clean_scores) < 2:
                print(f"[ERROR] {name} for {self.category}: Insufficient valid samples after cleaning "
                    f"({len(clean_scores)}/2 required)")
                return None, None
            
            # Check if we have both classes
            if len(np.unique(clean_labels)) < 2:
                print(f"[ERROR] {name} for {self.category}: Only one class present after cleaning")
                return None, None
            
            return clean_scores, clean_labels

        # Clean hybrid data
        scores_hybrid_clean, labels_hybrid_clean = clean_data(scores_hybrid, labels_hybrid, "Hybrid")
        
        # Clean model-only data
        scores_model_clean, labels_model_clean = clean_data(scores_model, labels_model, "Model-only")
        
        # --- HYBRID AUROC ---
        if scores_hybrid_clean is not None and labels_hybrid_clean is not None:
            try:
                fpr_h, tpr_h, thresholds_h = roc_curve(labels_hybrid_clean, scores_hybrid_clean)
                self.auroc_hybrid = auc(fpr_h, tpr_h)
                
                # TPR @ FPR = 0.5% (0.005)
                self.tpr_at_fpr_005_hybrid = self._interpolate_tpr_at_fpr(fpr_h, tpr_h, 0.005)
            except Exception as e:
                print(f"[ERROR] Hybrid ROC calculation failed for {self.category}: {e}")
                self.auroc_hybrid = 0.0
                self.tpr_at_fpr_005_hybrid = 0.0
        else:
            print(f"[ERROR] Hybrid ROC skipped for {self.category}: Invalid data")
            self.auroc_hybrid = 0.0
            self.tpr_at_fpr_005_hybrid = 0.0
        
        # --- MODEL-ONLY AUROC ---
        if scores_model_clean is not None and labels_model_clean is not None:
            try:
                fpr_m, tpr_m, thresholds_m = roc_curve(labels_model_clean, scores_model_clean)
                self.auroc_model_only = auc(fpr_m, tpr_m)
                
                # TPR @ FPR = 0.5%
                self.tpr_at_fpr_005_model = self._interpolate_tpr_at_fpr(fpr_m, tpr_m, 0.005)
            except Exception as e:
                print(f"[ERROR] Model-only ROC calculation failed for {self.category}: {e}")
                self.auroc_model_only = 0.0
                self.tpr_at_fpr_005_model = 0.0
        else:
            print(f"[ERROR] Model-only ROC skipped for {self.category}: Invalid data")
            self.auroc_model_only = 0.0
            self.tpr_at_fpr_005_model = 0.0
    
    @staticmethod
    def _interpolate_tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
        """
        Interpolate TPR at a specific FPR threshold.
        Handles edge cases where target_fpr is outside the observed range.
        """
        if len(fpr) < 2:
            return 0.0
        
        # Ensure FPR is strictly increasing for interpolation
        # ROC curve sometimes has duplicate FPR values, so we need to handle this
        unique_fpr, unique_indices = np.unique(fpr, return_index=True)
        unique_tpr = tpr[unique_indices]
        
        # Sort by FPR (should already be sorted, but ensure it)
        sort_idx = np.argsort(unique_fpr)
        unique_fpr = unique_fpr[sort_idx]
        unique_tpr = unique_tpr[sort_idx]
        
        if target_fpr < unique_fpr[0]:
            # Target FPR is lower than minimum observed FPR
            return 0.0
        elif target_fpr > unique_fpr[-1]:
            # Target FPR is higher than maximum observed FPR
            return unique_tpr[-1]
        else:
            # Interpolate
            interp_func = interp1d(unique_fpr, unique_tpr, kind='linear')
            return float(interp_func(target_fpr))

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tpr

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0: return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

class ResultsTracker:
    def __init__(self, output_dir: str, num_envs: int):
        self.output_dir = output_dir
        self.num_envs = num_envs
        self.num_control = num_envs // 2
        self.episodes: List[EpisodeResult] = []
        os.makedirs(output_dir, exist_ok=True)

    def start_episode(self, category: OODCategory, episode_length: int, ood_start_step: int):
        episode = EpisodeResult(
            category=CATEGORY_NAMES[category], 
            category_idx=category.value,
            episode_length=episode_length, 
            ood_start_step=ood_start_step,
            env_results=[EnvResult(env_idx=i, is_ood_env=(i >= self.num_control)) for i in range(self.num_envs)]
        )
        self.episodes.append(episode)
        return episode

    def record_detection(self, episode: EpisodeResult, env_idx: int, step: int, 
                         score: float, by_range: bool, by_svdd: bool):
        result = episode.env_results[env_idx]
        if not result.detected:
            result.detected = True
            result.detection_step = step
            if by_range: result.detected_by_range = True
            if by_svdd: result.detected_by_rapt = True
        result.max_score = max(result.max_score, score)

    def update_score(self, episode: EpisodeResult, env_idx: int, score: float):
        episode.env_results[env_idx].max_score = max(episode.env_results[env_idx].max_score, score)
    
    def record_timing(self, episode: EpisodeResult, latency_ms: float):
        """Record the detection latency for this timestep."""
        episode.detection_latencies_ms.append(latency_ms)
    
    def finalize_episode_scores(self, episode: EpisodeResult):
        episode.labels = []  # Keep for compatibility
        episode.labels_hybrid = []
        episode.labels_model_only = []
        episode.scores_hybrid = []
        episode.scores_model_only = []
        
        for result in episode.env_results:
            is_ood = result.is_ood_env
            
            # --- ADD THIS LINE BELOW ---
            episode.labels.append(1 if is_ood else 0) 
            # ---------------------------

            # Hybrid: detected by EITHER range OR model
            episode.labels_hybrid.append(1 if is_ood else 0)  # Ground truth
            episode.labels_model_only.append(1 if is_ood else 0)  # Ground truth
            
            # Scores
            episode.scores_hybrid.append(result.max_score)
            episode.scores_model_only.append(result.max_score_model_only)

    def finalize_episode(self, episode: EpisodeResult):
        episode.compute_metrics()
        
        print(f"\n{'='*60}")
        print(f"Episode Results: {episode.category}")
        print(f"{'='*60}")
        print(f"  [METRICS]  PADD: {episode.padd:.2f} | F1: {episode.f1:.4f} | Acc: {episode.accuracy:.4f}")
        print(f"  [RATES]    TPR: {episode.tpr:.4f} | FPR: {episode.fpr:.4f}")
        print(f"  [ROC]      AUROC (Hybrid): {episode.auroc_hybrid:.4f} | TPR@FPR=0.5%: {episode.tpr_at_fpr_005_hybrid:.4f}")
        print(f"  [ROC]      AUROC (Model):  {episode.auroc_model_only:.4f} | TPR@FPR=0.5%: {episode.tpr_at_fpr_005_model:.4f}")
        print(f"  [TIMING]   Avg Latency: {episode.avg_detection_latency_ms:.2f} ms")
        print(f"{'-'*60}")
        print(f"  Overlap (TPs): Range Only: {episode.tp_range_only}, rapt Only: {episode.tp_rapt_only}, Both: {episode.tp_both}")
        print(f"{'='*60}\n")

    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """
        Compute metrics aggregated across ALL categories (excluding calibration).
        Returns: Dict with AUROC, TPR@FPR=0.5%, avg latency
        """
        all_labels = []
        all_scores_hybrid = []
        all_scores_model = []
        all_latencies = []
        
        for ep in self.episodes:
            if ep.category_idx == OODCategory.NONE.value:
                continue  # Skip calibration episode
            
            all_labels.extend(ep.labels)
            all_scores_hybrid.extend(ep.scores_hybrid)
            all_scores_model.extend(ep.scores_model_only)
            all_latencies.extend(ep.detection_latencies_ms)
        
        if len(all_labels) == 0:
            return {
                'auroc_hybrid': 0.0,
                'auroc_model': 0.0,
                'tpr_at_fpr_005_hybrid': 0.0,
                'tpr_at_fpr_005_model': 0.0,
                'avg_latency_ms': 0.0
            }
        
        # Convert to numpy and clean
        all_labels = np.array(all_labels)
        all_scores_hybrid = np.array(all_scores_hybrid)
        all_scores_model = np.array(all_scores_model)
        
        # --- CLEAN HYBRID DATA ---
        valid_hybrid = np.isfinite(all_scores_hybrid)
        if not np.all(valid_hybrid):
            n_invalid = np.sum(~valid_hybrid)
            print(f"[WARNING] Aggregate: Removing {n_invalid} invalid hybrid samples")
        
        labels_hybrid = all_labels[valid_hybrid]
        scores_hybrid = all_scores_hybrid[valid_hybrid]
        
        # --- CLEAN MODEL DATA ---
        valid_model = np.isfinite(all_scores_model)
        if not np.all(valid_model):
            n_invalid = np.sum(~valid_model)
            print(f"[WARNING] Aggregate: Removing {n_invalid} invalid model samples")
        
        labels_model = all_labels[valid_model]
        scores_model = all_scores_model[valid_model]
        
        # --- CALCULATE METRICS ---
        result = {
            'auroc_hybrid': 0.0,
            'auroc_model': 0.0,
            'tpr_at_fpr_005_hybrid': 0.0,
            'tpr_at_fpr_005_model': 0.0,
            'avg_latency_ms': 0.0
        }
        
        # AUROC Hybrid
        if len(scores_hybrid) >= 2 and len(np.unique(labels_hybrid)) >= 2:
            try:
                fpr_h, tpr_h, _ = roc_curve(labels_hybrid, scores_hybrid)
                result['auroc_hybrid'] = auc(fpr_h, tpr_h)
                result['tpr_at_fpr_005_hybrid'] = EpisodeResult._interpolate_tpr_at_fpr(fpr_h, tpr_h, 0.005)
            except Exception as e:
                print(f"[ERROR] Aggregate Hybrid ROC failed: {e}")
        else:
            print(f"[WARNING] Insufficient valid hybrid data for aggregate ROC")
        
        # AUROC Model-Only
        if len(scores_model) >= 2 and len(np.unique(labels_model)) >= 2:
            try:
                fpr_m, tpr_m, _ = roc_curve(labels_model, scores_model)
                result['auroc_model'] = auc(fpr_m, tpr_m)
                result['tpr_at_fpr_005_model'] = EpisodeResult._interpolate_tpr_at_fpr(fpr_m, tpr_m, 0.005)
            except Exception as e:
                print(f"[ERROR] Aggregate Model ROC failed: {e}")
        else:
            print(f"[WARNING] Insufficient valid model data for aggregate ROC")
        
        # Average Latency (clean latencies too)
        if all_latencies:
            all_latencies = np.array(all_latencies)
            valid_latencies = all_latencies[np.isfinite(all_latencies)]
            if len(valid_latencies) > 0:
                result['avg_latency_ms'] = float(np.mean(valid_latencies))
        
        return result

    def save_results(self, filename: str = "ood_results.json"):
        output_path = os.path.join(self.output_dir, filename)
        
        # Compute aggregate metrics
        aggregate = self.compute_aggregate_metrics()
        
        out_dict = {
            "summary": {
                "num_episodes": len(self.episodes),
                "aggregate_metrics": aggregate
            }, 
            "episodes": []
        }
        
        for ep in self.episodes:
            ep_d = asdict(ep)
            ep_d['metrics'] = {
                'acc': float(ep.accuracy), 
                'f1': float(ep.f1), 
                'padd': float(ep.padd),
                'auroc_hybrid': float(ep.auroc_hybrid),
                'auroc_model': float(ep.auroc_model_only),
                'tpr_at_fpr_005_hybrid': float(ep.tpr_at_fpr_005_hybrid),
                'tpr_at_fpr_005_model': float(ep.tpr_at_fpr_005_model),
                'avg_latency_ms': float(ep.avg_detection_latency_ms)
            }
            out_dict['episodes'].append(ep_d)

        def convert(o):
            if isinstance(o, (np.floating, float, torch.Tensor)): return float(o)
            if isinstance(o, (np.integer, int)): return int(o)
            if isinstance(o, dict): return {k: convert(v) for k,v in o.items()}
            if isinstance(o, list): return [convert(v) for v in o]
            return o

        with open(output_path, 'w') as f: 
            json.dump(convert(out_dict), f, indent=2)
        
        print(f"[INFO] Results saved to {output_path}")
        print(f"\n{'='*60}")
        print(f"AGGREGATE METRICS (All Categories)")
        print(f"{'='*60}")
        print(f"  AUROC (Hybrid):        {aggregate['auroc_hybrid']:.4f}")
        print(f"  AUROC (Model-Only):    {aggregate['auroc_model']:.4f}")
        print(f"  TPR @ FPR=0.5% (Hybrid): {aggregate['tpr_at_fpr_005_hybrid']:.4f}")
        print(f"  TPR @ FPR=0.5% (Model):  {aggregate['tpr_at_fpr_005_model']:.4f}")
        print(f"  Avg Detection Latency: {aggregate['avg_latency_ms']:.2f} ms")
        print(f"{'='*60}\n")

    def save_summary_csv(self, filename: str = "ood_summary.csv"):
        import csv
        output_path = os.path.join(self.output_dir, filename)
        
        header = [
            'category', 
            'PADD', 'FPR', 'TPR', 'FNR', 'TNR', 'F1', 'accuracy',
            'AUROC_Hybrid', 'AUROC_Model', 
            'TPR@FPR0.5%_Hybrid', 'TPR@FPR0.5%_Model',
            'Avg_Latency_ms',
            'range_only', 'rapt_only', 'both'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for ep in self.episodes:
                writer.writerow([
                    ep.category, 
                    f"{ep.padd:.4f}", 
                    f"{ep.fpr:.4f}", 
                    f"{ep.tpr:.4f}", 
                    f"{ep.fnr:.4f}", 
                    f"{ep.tnr:.4f}", 
                    f"{ep.f1:.4f}", 
                    f"{ep.accuracy:.4f}",
                    f"{ep.auroc_hybrid:.4f}",
                    f"{ep.auroc_model_only:.4f}",
                    f"{ep.tpr_at_fpr_005_hybrid:.4f}",
                    f"{ep.tpr_at_fpr_005_model:.4f}",
                    f"{ep.avg_detection_latency_ms:.2f}",
                    ep.tp_range_only, ep.tp_rapt_only, ep.tp_both
                ])
        
        print(f"[INFO] CSV saved to {output_path}")

# ==================================================================================
# MAIN SCRIPT
# ==================================================================================
def remove_per_feature_history(obs, history_len=5):
    # The dimension of a SINGLE frame for each term in your config order
    # Sum must equal 96
    term_dims = [3, 3, 3, 29, 29, 29]
    
    slices = []
    cursor = 0
    
    for dim in term_dims:
        # Calculate the size of this term's entire history block
        block_size = dim * history_len
        
        # We want the LAST 'dim' elements of this block (the most recent frame)
        # Start: cursor + (block_size - dim)
        # End:   cursor + block_size
        start_idx = cursor + block_size - dim
        end_idx = cursor + block_size
        
        # Slice and store
        slices.append(obs[:, start_idx:end_idx])
        
        # Move cursor to the start of the next feature block
        cursor += block_size

    return torch.cat(slices, dim=1)

def format_feature_history(obs_queue, term_dims):
    """
    Rearranges a queue of frames into feature-based history.
    Args:
        obs_queue: list or deque of N tensors of shape (Batch, 96)
        term_dims: list of ints, e.g. [3, 3, 3, 29, 29, 29]
    Returns:
        Tensor of shape (Batch, 96 * N) organized by feature groups.
    """
    # 1. Stack the queue: (Batch, HistoryLen, TotalFeatures)
    # This assumes queue is [t-4, t-3, t-2, t-1, t] (oldest to newest)
    history_stack = torch.stack(list(obs_queue), dim=1)
    
    # 2. Split into individual features along the last dimension
    # Returns a tuple of tensors: ((B, Hist, 3), (B, Hist, 3), ... (B, Hist, 29))
    feature_slices = torch.split(history_stack, term_dims, dim=-1)
    
    # 3. Flatten the History and Feature dimensions for each slice
    # Reshape (Batch, Hist, FeatDim) -> (Batch, Hist * FeatDim)
    flattened_features = []
    for f in feature_slices:
        batch_size = f.shape[0]
        flattened_features.append(f.reshape(batch_size, -1))
        
    # 4. Concatenate the flattened features together
    # Result: [FeatA_t-4...FeatA_t, FeatB_t-4...FeatB_t, ...]
    return torch.cat(flattened_features, dim=1)
    
def save_calibration(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)

def load_calibration(path):
    return torch.load(path, map_location="cpu")

def main():
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.task == 'Unitree-G1-29dof-Velocity': # no curriculum needed for evaluation
        env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    if args_cli.calibration_cache is None:
        calibration_cache_path = os.path.join(args_cli.output_dir, "calibration_cache.pt")
    else:
        calibration_cache_path = args_cli.calibration_cache

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv): env = multi_agent_to_single_agent(env)
    
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, video_folder=os.path.join(log_dir, "videos", "ood_test"),
                                       step_trigger=lambda step: step == 0, video_length=args_cli.video_length, disable_logger=True)
    
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, multihead=args_cli.use_critic_multi)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- RESET TO GET DIMS ---
    obs, _ = env.reset()
    obs, _ = env.get_observations()
    
    if obs is None: raise RuntimeError("Environment failed to return observations!")
    remove_history = lambda x: remove_per_feature_history(x) if args_cli.task == 'Unitree-G1-29dof-Velocity' else x
    obs = remove_history(obs)

    obs_dim = obs.shape[-1]
    try: action_dim = env.action_space.shape[-1]
    except: action_dim = env.action_space['policy'].shape[-1]
    
    num_envs = args_cli.num_envs
    dt = env.unwrapped.step_dt
    print(f"[INFO] Obs dim: {obs_dim}, Action dim: {action_dim}")

    # --- INITIALIZE HYBRID DETECTOR ---
    hybrid_detector = HybridAnomalyDetector(
        args_cli,
        obs_dim=obs.shape[-1],
        action_dim=action_dim,
        device=args_cli.device,
    )

    # Initialize OOD Injector
    ood_injector = OODInjector(num_envs=num_envs, obs_dim=obs.shape[-1], action_dim=action_dim, dt=dt, device=args_cli.device, env=env)
    
    saliency_explainer = SaliencyExplainer(hybrid_detector, device=args_cli.device)
    target_viz_env = num_envs - 1  # We will trace the last environment (guaranteed OOD)
    viz_trace_buffer = []          # Store history for this specific env
    
    results_tracker = ResultsTracker(os.path.join(log_dir, args_cli.output_dir), num_envs)
    
    if args_cli.categories == "all": categories_to_test = [c for c in OODCategory if c != OODCategory.NONE]
    else: categories_to_test = [OODCategory(int(x.strip())) for x in args_cli.categories.split(",")]

    # --- BASELINE / CALIBRATION EPISODE ---
    print(f"\n{'#'*60}\n# Baseline Episode: Calibration\n{'#'*60}")


    run_calibration = (
        args_cli.force_recalibrate
        or not os.path.exists(calibration_cache_path)
    )

    if run_calibration:
        print("[CALIBRATION] Running nominal calibration episode")

        ood_injector.setup_episode(OODCategory.NONE, seed=100)
        hybrid_detector.reset()
        episode_result = results_tracker.start_episode(OODCategory.NONE, args_cli.episode_length, args_cli.ood_start_step)
        
        obs, _ = env.reset()
        obs, _ = env.get_observations()
        obs_policy = obs.clone()
        obs = remove_history(obs)

        baseline_scores = [] # size [T, [N, O]]
        baseline_mean = [] # size [T, [N]]

        for step in range(args_cli.episode_length):
            if not simulation_app.is_running(): break
            
            t_start = time.perf_counter()
            losses, score_mean = hybrid_detector.calibrate(obs)
            is_anomaly, is_rng, is_rapt, _, _ = hybrid_detector.check_batch(obs)
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000  # Convert to milliseconds
            results_tracker.record_timing(episode_result, latency_ms)

            baseline_scores.append(losses.cpu())
            baseline_mean.append(score_mean.cpu())

            # 2. Policy Step
            with torch.no_grad(): actions = policy(obs_policy)
            
            # 3. Update Detector Buffers (Store s_t, a_t for next step validation)
            hybrid_detector.update_buffer(obs, actions)
            
            # 4. Environment Step
            obs, _, _, _ = env.step(actions)
            obs_policy = obs.clone()
            obs = remove_history(obs)

            # Track results
            losses_cpu = losses.cpu().detach().numpy()
            is_rng_cpu = is_rng.cpu().detach().numpy()
            is_rapt_cpu = is_rapt.cpu().detach().numpy()
            for env_idx in range(num_envs):
                if is_anomaly[env_idx]:
                    results_tracker.record_detection(episode_result, env_idx, step, losses_cpu[env_idx].max(), 
                                                    is_rng_cpu[env_idx], is_rapt_cpu[env_idx])
                else:
                    results_tracker.update_score(episode_result, env_idx, losses_cpu[env_idx].max())

        scores = torch.stack(baseline_scores)   # [T, N, O]

        score_max = scores.amax(dim=(0, 1))     # [O]
        score_std = scores.std(dim=(0, 1))      # [O]

        thresh_max = score_max + 5.0 * score_std# + (scores.amax(dim=(0,1)) - scores.amin(dim=(0,1)))*0.2  # Add range to threshold
        thresh_max = torch.clamp(thresh_max, min=1e-6)

        #means = torch.stack(baseline_mean)      # [N, T]

        mean_max = scores.amax(dim=(0, 1)).mean()     # [1]
        mean_std = scores.amax(dim=0).mean(dim=1).std()      # [1]

        thresh_mean = mean_max + 3.0 * mean_std
        thresh_mean = torch.clamp(thresh_mean, min=1e-6)

        # these thresholds don't matter, overriden with ROC-based fixed fpr thresholds for sim experiments:
        hybrid_detector.rapt_threshold = [thresh_max, thresh_mean]

        # After you compute thresholds:
        calib_data = {
            "rapt_threshold": hybrid_detector.rapt_threshold,
            "lower_bound": hybrid_detector.range_detector.lower_bound,
            "upper_bound": hybrid_detector.range_detector.upper_bound,
        }

        save_calibration(calibration_cache_path, calib_data)
        print(f"[CALIBRATION] Saved cache to {calibration_cache_path}")
        print(thresh_max, thresh_mean)
        results_tracker.finalize_episode_scores(episode_result)
        results_tracker.finalize_episode(episode_result)

    else:
        print(f"[CALIBRATION] Loading cache from {calibration_cache_path}")

        calib_data = load_calibration(calibration_cache_path)

        hybrid_detector.rapt_threshold = calib_data["rapt_threshold"]#.to(args_cli.device)
        hybrid_detector.range_detector.lower_bound = calib_data["lower_bound"].to(args_cli.device)
        hybrid_detector.range_detector.upper_bound = calib_data["upper_bound"].to(args_cli.device)

    print(f"\n[CALIBRATION COMPLETE]")
    print(f"  Range Bounds Updated: Min/Max expanded to cover Baseline.")
    print(f"{'='*60}\n")
    ood_injector.set_all_ood()
    results_tracker.num_control = 0

    # --- OOD EPISODES ---
    try:
        for i, category in enumerate(categories_to_test):
            print(f"Running {CATEGORY_NAMES[category]}...")
            ood_injector.setup_episode(category, seed=i*1000)
            hybrid_detector.reset()
            
            episode_result = results_tracker.start_episode(category, args_cli.episode_length, args_cli.ood_start_step)
            history_len = 1 if args_cli.task != 'Unitree-G1-29dof-Velocity' else 5
            obs_history_queue = deque(maxlen=history_len)
            get_latest_step = lambda x: remove_per_feature_history(x) if args_cli.task == 'Unitree-G1-29dof-Velocity' else x

            obs, _ = env.reset()
            obs, _ = env.get_observations()
            
            if args_cli.task == 'Unitree-G1-29dof-Velocity':
                term_dims = [3, 3, 3, 29, 29, 29]
                obs_history_queue.clear()
                obs = remove_per_feature_history(obs)
                for _ in range(history_len):
                    obs_history_queue.append(obs.clone())
                obs_policy = format_feature_history(obs_history_queue, term_dims)
            else:
                obs_policy = obs.clone()
            
            detected_envs = torch.zeros(num_envs, dtype=torch.bool, device=args_cli.device)
            max_scores_hybrid = torch.full((num_envs,), -float('inf'), device=args_cli.device)
            max_scores_model = torch.full((num_envs,), -float('inf'), device=args_cli.device)

            for step in range(args_cli.episode_length-1):
                obs_clean_current = obs.clone()
                if not simulation_app.is_running(): break
                
                # OOD Injection
                if step == args_cli.ood_start_step: ood_injector.activate()
                obs_perturbed_current = ood_injector.inject_obs(obs_clean_current.clone())
                
                if args_cli.task == 'Unitree-G1-29dof-Velocity':
                    obs_history_queue.append(obs_perturbed_current)
                    term_dims = [3, 3, 3, 29, 29, 29]
                    obs_policy = format_feature_history(obs_history_queue, term_dims)

                else:
                    obs_policy = obs_perturbed_current
                
                t_start = time.perf_counter()
                # 2. HYBRID CHECK
                # If Dynamics Model: Checks obs_perturbed against prediction from (prev_obs, prev_act)
                # If AE Model: Checks reconstruction of obs_perturbed
                is_anomaly, is_rng, is_rapt, losses,score_mean = hybrid_detector.check_batch(obs_perturbed_current)
                
                # reset all envs that are anomaly:
                if is_anomaly.any():
                    env_ids = torch.nonzero(is_anomaly, as_tuple=False).squeeze(-1)
                    
                    # 1) Reset those envs (try wrapper, then unwrapped)
                    if hasattr(env.unwrapped, "_reset_idx"):
                        default_root_state = env.unwrapped.scene["robot"].data.default_root_state[env_ids]
                        default_root_state[:, :3] += 1e8  # Move far away to avoid collisions during reset
                        env.unwrapped.scene["robot"].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
                        #env.unwrapped._reset_idx(env_ids)
                    elif hasattr(env.unwrapped, "reset_idx"):
                        env.unwrapped.reset_idx(env_ids)
                    elif hasattr(env, "reset_done"):
                        env.reset_done(env_ids)
                    elif hasattr(env.unwrapped, "reset_done"):
                        env.unwrapped.reset_done(env_ids)
                    else:
                        raise RuntimeError("Env does not expose reset_idx/reset_done for per-env reset.")

                    # 2) Important: reset YOUR detector state for those envs
                    hybrid_detector.reset_envs(env_ids)  # implement this (see below)

                #print(torch.max(z_scores[:2048]),torch.max(z_scores[2048:]),z_scores.shape)
                #assert False, "Debug Break"
                # 3. Record
                t_end = time.perf_counter()
                
                # Keep buffer size manageable (e.g., last 50 steps)
                if len(viz_trace_buffer) > 50: viz_trace_buffer.pop(0)
                
                latency_ms = (t_end - t_start) * 1000
                results_tracker.record_timing(episode_result, latency_ms)
                
                calib_max_thresh = hybrid_detector.rapt_threshold[0].to(args_cli.device)
                calib_mean_thresh = hybrid_detector.rapt_threshold[1].to(args_cli.device)
                norm_max_loss = losses / calib_max_thresh  # [Batch, ObsDim]
                risk_max_local = norm_max_loss.max(dim=1)[0] # [Batch]
                risk_mean_global = score_mean / calib_mean_thresh
                if args_cli.ood_detect_type == 'max':
                    model_risk_score = risk_max_local
                elif args_cli.ood_detect_type == 'mean':
                    model_risk_score = risk_mean_global
                else: # 'both'
                    # Whichever risk ratio is higher determines the anomaly score
                    model_risk_score = torch.maximum(risk_max_local, risk_mean_global)

                hybrid_risk_score = model_risk_score.clone()
                hybrid_risk_score[is_rng] = 1e9 

                max_scores_hybrid = torch.maximum(max_scores_hybrid, hybrid_risk_score)
                max_scores_model = torch.maximum(max_scores_model, model_risk_score)

                losses_cpu = losses.cpu().numpy()
                is_rng_cpu = is_rng.cpu().numpy()
                is_rapt_cpu = is_rapt.cpu().numpy()
                is_anom_cpu = is_anomaly.cpu().numpy()
                
                for env_idx in range(num_envs):
                    if is_anom_cpu[env_idx] and not detected_envs[env_idx]:
                        results_tracker.record_detection(episode_result, env_idx, step, losses_cpu[env_idx].max(), 
                                                       is_rng_cpu[env_idx], is_rapt_cpu[env_idx])
                        detected_envs[env_idx] = True
                    else:
                        results_tracker.update_score(episode_result, env_idx, losses_cpu[env_idx].max())
                
                # 4. Policy Step
                with torch.no_grad(): actions = policy(obs_policy)
                
                # 5. Inject Action anomalies
                actions_perturbed = ood_injector.inject_action(actions)
                
                # 6. Update Detector Buffers (for next step verification)
                hybrid_detector.update_buffer(obs_perturbed_current, actions_perturbed)

                curr_hidden = None
                if hybrid_detector.hidden_state is not None:
                    # Extract slice for target env: [1, 1, Dim]
                    curr_hidden = hybrid_detector.hidden_state[:, target_viz_env:target_viz_env+1, :].clone()
                
                #viz_trace_buffer.append({
                #    'obs': obs_perturbed_current[target_viz_env].detach().clone(),
                #    'act': actions_perturbed[target_viz_env].detach().clone(), # Uses Prev Action? Or Current? 
                    # Note: Dynamics Model predicts S_t using S_t-1, A_t-1. 
                    # If using AE, it reconstructs S_t. IG works best on S_t.
                 #   'hidden': curr_hidden,
                 #   'time': step * dt
                #})

                # 7. Env Step
                obs, _, _, _ = env.step(actions_perturbed)
                obs_policy = obs.clone()
                obs = remove_history(obs)
                if args_cli.real_time: time.sleep(dt)
            
            for env_idx in range(num_envs):
                result = episode_result.env_results[env_idx]
                result.max_score = float(max_scores_hybrid[env_idx].cpu())
                result.max_score_model_only = float(max_scores_model[env_idx].cpu())
                # Store model-only score separately
            
            episode_result.scores_model_only = max_scores_model.cpu().tolist()
            
            results_tracker.finalize_episode_scores(episode_result)
            results_tracker.finalize_episode(episode_result)

            save_name = f"top5_{CATEGORY_NAMES[category]}.png"
            
            # Change this line:
            #saliency_explainer.plot_categorical_heatmap(viz_trace_buffer, 
            #                                       save_path=os.path.join(log_dir, save_name),
            #                                       )
            
            #viz_trace_buffer = [] 

    except KeyboardInterrupt: print("Interrupted.")
    finally:
        ood_injector._reset_physics()
        results_tracker.save_results()
        results_tracker.save_summary_csv()
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()