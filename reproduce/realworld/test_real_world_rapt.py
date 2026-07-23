import pandas as pd
import numpy as np
import matplotlib.animation as animation
import sys
import torch
import torch.nn as nn
import json
import os
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
torch.backends.cudnn.enabled = False

# --- CONFIGURATION ---
CAL_BASE = 'deploy/robots/g1_29dof/build/logs/cali_real_new_1'

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

CAL_LOSS_FILE = f'{CAL_BASE}/loss.csv'
CAL_OBS_FILE  = f'{CAL_BASE}/observations.csv'
CAL_ACT_FILE  = f'{CAL_BASE}/actions.csv'

MODEL_CHECKPOINT = "fdm_models_deploy/DYN_Unitree-G1-29dof-Velocity_20260325_125357/final_model.pth"
RECALCULATE_LOSS = False  # If False, load loss.csv when available and use the exact same threshold logic.

MAE_EMBED_DIM = 256
MAE_NUM_BLOCKS = 4
USE_RESIDUAL = True
DROPOUT = 0.0
MASK_RATIO = 0.25

def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

# ==================================================================================
# 1. MODEL DEFINITIONS
# ==================================================================================
class ConfigurableBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=True, dropout=0.1):
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)
        hidden_dim = int(out_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim), nn.Dropout(dropout)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        return self.relu(x + out) if self.use_residual else self.relu(out)

class UniversalModel(nn.Module):
    def __init__(self, obs_dim, action_dim, train_dynamics=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.train_dynamics = train_dynamics
        self.embed_dim = MAE_EMBED_DIM
        self.input_dim = obs_dim + action_dim if train_dynamics else obs_dim
        
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim), nn.ReLU(),
            *[ConfigurableBlock(self.embed_dim, self.embed_dim, USE_RESIDUAL, DROPOUT) for _ in range(MAE_NUM_BLOCKS)]
        )
        self.gru = nn.GRU(self.embed_dim, self.embed_dim, num_layers=1, batch_first=True)
        
        self.compress = nn.Sequential(nn.Linear(self.embed_dim, int(self.embed_dim * (1-MASK_RATIO))), 
                                      nn.LayerNorm(int(self.embed_dim * (1-MASK_RATIO))), nn.ReLU())
        self.decompress = nn.Sequential(nn.Linear(int(self.embed_dim * (1-MASK_RATIO)), self.embed_dim), nn.ReLU())
        
        self.decoder_mlp = nn.Sequential(
            *[ConfigurableBlock(self.embed_dim, self.embed_dim, USE_RESIDUAL, DROPOUT) for _ in range(MAE_NUM_BLOCKS)]
        )
        self.head = nn.Linear(self.embed_dim, obs_dim * 2)

    def forward(self, x, actions=None, hidden=None):
        # Sequence handling: [Batch, Seq, Dim]
        is_sequence = x.dim() == 3
        if not is_sequence:
            x = x.unsqueeze(1)
            if actions is not None: actions = actions.unsqueeze(1)

        if self.train_dynamics:
            model_input = torch.cat([x, actions], dim=-1)
        else:
            model_input = x
            
        z = self.encoder_mlp(model_input)
        z, hidden = self.gru(z, hidden)
        z_neck = self.compress(z)
        z_rec = self.decompress(z_neck)
        out = self.head(self.decoder_mlp(z_rec))
        
        if not is_sequence:
            out = out.squeeze(1)
            
        return out, hidden

# ==================================================================================
# 2. STATEFUL EXPLAINER
# ==================================================================================
class StatefulExplainer:
    def __init__(self, obs_dim, action_dim, device='cuda'):
        self.device = device
        self.obs_dim = obs_dim
        self.obs_names = get_observation_names()
        self.hidden = None 
        
        print(f"\n[EXPLAINER] Loading model from {MODEL_CHECKPOINT}...")
        config_path = os.path.join(os.path.dirname(MODEL_CHECKPOINT), 'experiment_config.json')
        is_dynamics = True
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                is_dynamics = json.load(f).get('train_dynamics', True)
        
        self.model = UniversalModel(obs_dim, action_dim, train_dynamics=is_dynamics).to(device)
        ckpt = torch.load(MODEL_CHECKPOINT, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}, strict=True)
        self.model.eval()

    def warm_start(self, obs_seq, act_seq):
        """Runs the model on a sequence to prep the hidden state up to the end."""
        with torch.no_grad():
            _, self.hidden = self.model(obs_seq.unsqueeze(0), actions=act_seq.unsqueeze(0))

    def update_state(self, current_obs, current_act):
        """Updates the GRU hidden state for the current timestep (Single Step)."""
        obs = torch.tensor(current_obs.values, dtype=torch.float32).to(self.device).view(1, 1, -1)
        act = torch.tensor(current_act.values, dtype=torch.float32).to(self.device).view(1, 1, -1)
        with torch.no_grad():
            _, self.hidden = self.model(obs, actions=act, hidden=self.hidden)

    def compute_ig(self, obs_tensor, act_tensor, hidden_state, baseline_tensor):
        # Prepare inputs
        obs_input = obs_tensor.unsqueeze(0)
        act_input = act_tensor.unsqueeze(0)
        baseline = baseline_tensor.unsqueeze(0)

        # IG Interpolation
        steps = 50
        alphas = torch.linspace(0, 1, steps, device=self.device).view(steps, 1)
        interpolated_obs = baseline + alphas * (obs_input - baseline)
        interpolated_obs.requires_grad_(True)
        
        expanded_act = act_input.repeat(steps, 1)
        
        context_hidden = hidden_state.detach().clone() if hidden_state is not None else None
        if context_hidden is not None:
             context_hidden = context_hidden.expand(-1, steps, -1).contiguous()

        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False 
        
        try:
            output, _ = self.model(interpolated_obs.unsqueeze(1), actions=expanded_act.unsqueeze(1), hidden=context_hidden)
            output = output.squeeze(1) 
            mu, log_var = output[:, :self.obs_dim], output[:, self.obs_dim:]
            
            mse_term = ((interpolated_obs - mu) ** 2) / (torch.exp(log_var) + 1e-8)
            sigma_term = log_var
            step_nll = (mse_term).sum(dim=1)
            total_nll_sum = step_nll.sum()
            
            grads = torch.autograd.grad(total_nll_sum, interpolated_obs)[0]
        finally:
            torch.backends.cudnn.enabled = prev_cudnn

        ig = (obs_input - baseline) * grads.mean(dim=0, keepdim=True)
        return ig[0]

    def compute_ig_bptt(self, obs_seq, act_seq, hidden_init, baseline_vec):
        """
        Computes Temporal Integrated Gradients using Backpropagation Through Time.
        
        Args:
            obs_seq: Tensor [Seq_Len, Obs_Dim] (The history window, e.g., 10 steps)
            act_seq: Tensor [Seq_Len, Act_Dim]
            hidden_init: Tensor [1, 1, Hidden_Dim] (The hidden state BEFORE the window starts)
            baseline_vec: Tensor [Obs_Dim] (Static baseline to expand)
        """
        # 1. Setup Interpolation (Batch size = 50 steps)
        steps = 50
        # Shape: [50, 1, 1] for broadcasting
        alphas = torch.linspace(0, 1, steps, device=self.device).view(steps, 1, 1)
        
        # Expand inputs to create a batch of sequences: [50, Seq_Len, Dim]
        obs_input = obs_seq.unsqueeze(0)        # [1, Seq, Dim]
        # Create a static baseline sequence (repeated mean)
        base_input = baseline_vec.view(1, 1, -1).expand(1, obs_seq.size(0), -1)
        
        # Interpolate between baseline sequence and actual sequence
        interpolated = base_input + alphas * (obs_input - base_input)
        interpolated.requires_grad_(True)
        
        # Expand actions: [50, Seq_Len, Act_Dim]
        act_expanded = act_seq.unsqueeze(0).repeat(steps, 1, 1)
        
        # Expand hidden state: [1, 1, H] -> [1, 50, H] (GRU expects [Layers, Batch, Hidden])
        if hidden_init is not None:
            h_batch = hidden_init.expand(1, steps, -1).contiguous()
        else:
            h_batch = None

        # 2. Forward Pass (Process whole sequence in parallel)
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False  # Required for higher-order gradients in RNNs
        
        try:
            # Model output: [50, Seq_Len, 2*Obs_Dim]
            output, _ = self.model(interpolated, actions=act_expanded, hidden=h_batch)
            
            # 3. Loss Calculation (Focus on the LAST timestep)
            # We want to know: "What in the history caused the NLL spike at the very END?"
            last_output = output[:, -1, :]           # [50, 2*Obs_Dim]
            last_obs_target = interpolated[:, -1, :] # [50, Obs_Dim]
            
            mu, log_var = last_output[:, :self.obs_dim], last_output[:, self.obs_dim:]
            mse = ((last_obs_target - mu) ** 2) / (torch.exp(log_var) + 1e-8)
            nll = (mse).sum(dim=1)   # Sum over features -> [50]
            total_nll = nll.sum()
            
            # 4. Gradients w.r.t. the WHOLE sequence (BPTT happens here)
            grads = torch.autograd.grad(total_nll, interpolated)[0] # [50, Seq, Dim]
            
        finally:
            torch.backends.cudnn.enabled = prev_cudnn

        # 5. Average and Scale
        avg_grads = grads.mean(dim=0) # [Seq, Dim]
        ig = (obs_seq - base_input.squeeze(0)) * avg_grads
        return ig
    
    def get_observation_groups(self):
        """Returns a dictionary mapping group names to observation indices."""
        obs_names = get_observation_names()
        
        groups = {
            'Base (IMU)': [],
            'Lower Body Sensor': [],
            'Upper Body Sensor': [],
            'Lower Body Action': [],
            'Upper Body Action': [],
            'Commands': [],
        }
        
        # Base group: root velocities and gravity
        for i, name in enumerate(obs_names):
            if name.startswith('root_vel_') or name.startswith('gravity_'):
                groups['Base (IMU)'].append(i)
        
        # Commands
        for i, name in enumerate(obs_names):
            if name.startswith('cmd_'):
                groups['Commands'].append(i)
        
        # Lower body: hips, knees, ankles
        lower_joints = ['hip', 'knee', 'ankle']
        for i, name in enumerate(obs_names):
            if any(joint in name for joint in lower_joints) and 'pos' in name:
                groups['Lower Body Sensor'].append(i)
        
        # Upper body: waist, shoulders, elbows, wrists
        upper_joints = ['waist', 'shoulder', 'elbow', 'wrist']
        for i, name in enumerate(obs_names):
            if any(joint in name for joint in upper_joints) and 'pos' in name:
                groups['Upper Body Sensor'].append(i)

        # Lower body actions
        for i, name in enumerate(obs_names):
            if any(joint in name for joint in lower_joints) and 'action' in name:
                groups['Lower Body Action'].append(i)
        # Upper body actions
        for i, name in enumerate(obs_names):
            if any(joint in name for joint in upper_joints) and 'action' in name:
                groups['Upper Body Action'].append(i)
        
        return groups

    def plot_safety_margin_trajectory(
        self,
        history,
        start_time: float = 9.89,
        end_time: float = 13.87,
        save_path: str = "safety_margin.pdf",
    ):
        """
        RSS/IEEE single-column-ready plot export (vector PDF).

        Figure guidance implemented:
        - Single-column width: 3.5 in
        - Aim height <= 4.5 in (we use 2.2 in here)
        - Default font size ~9 pt (only one size used across the plot)
        - Restrained styling, readable ticks/legend at print size
        - Export as PDF with embedded fonts (pdflatex friendly)
        """
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # ----------------------------
        # Global Matplotlib settings
        # ----------------------------
        # NOTE: usetex=True requires a LaTeX installation. If you don't have it,
        # set to False and keep fonttype=42 for good PDF embedding.
        USE_TEX = False  # set True if texlive is installed and you want LaTeX rendering

        mpl.rcParams.update({
            # Consistent paper-like typography
            "text.usetex": USE_TEX,
            "font.family": "serif",          # IEEE-ish; change to "sans-serif" if your paper uses sans
            "font.size": 9,                  # strict: default inside-figure font size
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,

            # PDF export: embed TrueType fonts (pdflatex friendly)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            # Clean axes
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        })

        # ----------------------------
        # Select window
        # ----------------------------
        subset = [x for x in history if start_time <= x["time"] <= end_time]
        if not subset:
            print(f"[ERROR] No data found between {start_time}s and {end_time}s")
            return

        times = []
        margins = []

        # Threshold = Max_Calib + 5 * Std_Calib
        calib_max = self.detector.calib_max.to(self.device)
        calib_std = self.detector.calib_std.to(self.device)
        thresholds = calib_max + (5 * calib_std)

        print(f"[EXPLAINER] Tracing safety margin for {len(subset)} steps...")

        self.model.eval()
        with torch.no_grad():
            for frame in subset:
                obs = frame["obs"].unsqueeze(0).to(self.device)
                act = frame["act"].unsqueeze(0).to(self.device)
                hidden = frame["hidden"].to(self.device) if frame["hidden"] is not None else None

                output, _ = self.model(obs, actions=act, hidden=hidden)

                mu, log_var = output[:, : self.detector.obs_dim], output[:, self.detector.obs_dim :]
                sigma = torch.exp(0.5 * log_var)

                # Compare reconstruction to model input space (assumes obs already in model space)
                error = mu - obs

                nll_vector = (error**2) / (2 * sigma**2) + 0.5 * log_var

                # Margin = NLL - Threshold (negative = safe)
                diffs = nll_vector - thresholds
                worst_margin = torch.max(diffs).item()

                margins.append(worst_margin)
                times.append(frame["time"])

        times = np.asarray(times, dtype=float)
        margins = np.asarray(margins, dtype=float)

        # ----------------------------
        # Plot (IEEE single-column sizing)
        # ----------------------------
        fig_w, fig_h = 3.5, 2.2  # inches (single column); keep <= ~4.5" height
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Main curve (no extra styling noise)
        ax.plot(times, margins, linewidth=1.5, label="Worst-case margin")

        # Threshold at 0
        ax.axhline(0.0, linestyle="--", linewidth=1.0, label="Threshold")

        # Regions: keep subtle so curve remains primary
        ax.fill_between(times, margins, 0.0, where=(margins < 0.0), alpha=0.08, interpolate=True)
        ax.fill_between(times, margins, 0.0, where=(margins >= 0.0), alpha=0.08, interpolate=True)

        # Labels: concise and readable at print size
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Safety margin")

        # Avoid verbose titles inside single-column figures; captions carry the narrative.
        # If you must, keep it short:
        ax.set_title("Safety margin over time")

        # Subtle grid
        ax.grid(True, alpha=0.25, linewidth=0.6)

        # Legend: compact, not dominant
        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            borderpad=0.3,
            handlelength=1.6,
            handletextpad=0.6,
            labelspacing=0.3,
        )

        # Tight, but not "bbox_inches=tight" (can clip text in some LaTeX workflows)
        fig.tight_layout(pad=0.2)

        # ----------------------------
        # Save as vector PDF (embedded fonts)
        # ----------------------------
        if not save_path.lower().endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"

        fig.savefig(save_path, format="pdf", transparent=True)  # vector export
        plt.close(fig)
        print(f"[EXPLAINER] Saved safety margin plot to {save_path}")

    def explain_heatmap(self, trace_buffer, anomaly_indices=[], save_path=None, fallback_loss_vector=None, fallback_time=None):
        """Generates a heatmap over the last 50 timesteps."""
        BPTT_STEPS = 200  # Horizon H=50
        # --------------------------------
        
        # Ensure we have enough history
        full_window = list(trace_buffer)
        if len(full_window) < BPTT_STEPS + 1:
            print("[WARN] Not enough history for BPTT.")
            if save_path and fallback_loss_vector is not None:
                self.plot_single_step_loss_vector(
                    loss_vector=fallback_loss_vector,
                    save_path=save_path,
                    event_time=fallback_time,
                    k=self.obs_dim
                )
            return
        
        target_window = full_window[-BPTT_STEPS:] 
        
        # Prepare Tensors [Seq, Dim]
        obs_seq = torch.stack([x['obs'] for x in target_window])
        act_seq = torch.stack([x['act'] for x in target_window])
        
        # Get Hidden State from BEFORE the window starts
        # trace_buffer[i]['hidden'] is the state AFTER step i.
        # So we need the hidden state from the step just before our window.
        prev_step_idx = len(full_window) - BPTT_STEPS - 1
        h_init = full_window[prev_step_idx]['hidden']
        
        # Calculate Baseline (Mean of 5 steps before the BPTT window)
        bl_start = max(0, prev_step_idx - 5)
        bl_frames = [x['obs'] for x in full_window[bl_start : prev_step_idx + 1]]
        baseline_vec = torch.stack(bl_frames).mean(dim=0)

        # 2. Compute BPTT Integrated Gradients
        # This returns a [10, Obs_Dim] matrix of attributions
        ig_matrix = self.compute_ig_bptt(obs_seq, act_seq, h_init, baseline_vec)

        # 3. Print Saliency Table
        #print(f"\n  >>> TEMPORAL SALIENCY (BPTT, Last {BPTT_STEPS} steps) <<<")
        #print(f"  {'Feature Name':<30} | " + " | ".join([f"{x['time']:.4f}s" for x in target_window]))
        #print("  " + "-" * (30 + 10 * len(target_window)))

        # Find top contributors based on the LAST frame's total attribution
        final_frame_ig = ig_matrix[-1]
        top_k = 10
        _, top_indices = torch.topk(torch.abs(final_frame_ig), k=top_k)
        idx_set = set(top_indices.cpu().numpy().tolist())
        for idx in anomaly_indices: idx_set.add(idx)
        final_indices = sorted(list(idx_set), key=lambda idx: abs(final_frame_ig[idx].item()), reverse=True)

        for feat_idx in final_indices:
            name = self.obs_names[feat_idx]
            row_str = f"  {name:<30} | "
            for t in range(BPTT_STEPS):
                val = ig_matrix[t][feat_idx].item()
                row_str += f"{val:7.4f} | "
            #print(row_str)
        #print("  " + "-"*50)

        if save_path:
            # We pass the BPTT window and IG matrix to the plotter
            # Note: We wrap ig_matrix in a list to match expected format if needed, 
            # or modify plot_grouped_saliency to accept a tensor.
            # Here I assume plot_grouped_saliency expects a list of tensors (one per step)
            ig_list = [ig_matrix[t] for t in range(BPTT_STEPS)]
            
            # Switch to Top-5 Plotter
            self.plot_top_k_saliency(target_window, ig_list, save_path, k=10)

        return ig_matrix, target_window

    def _get_saliency_output_paths(self, save_path):
        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = '.png'
        return f"{base}_log{ext}", f"{base}_linear{ext}"

    def plot_single_step_loss_vector(self, loss_vector, save_path, event_time=None, k=20):
        """Plots one-step top-k absolute loss heatmaps in log and linear-normalized scales."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        import torch

        if isinstance(loss_vector, pd.Series):
            values = loss_vector.values.astype(np.float32)
            feature_names = list(loss_vector.index)
        elif isinstance(loss_vector, torch.Tensor):
            values = loss_vector.detach().cpu().numpy().astype(np.float32)
            feature_names = self.obs_names
        else:
            values = np.asarray(loss_vector, dtype=np.float32)
            feature_names = self.obs_names

        values_abs = np.abs(values)
        if values_abs.size == 0:
            print("[WARN] Empty loss vector for single-step fallback heatmap.")
            return

        k_use = max(1, min(k, values_abs.shape[0]))
        top_idx = np.argsort(-values_abs)[:k_use]
        top_vals = values_abs[top_idx]

        epsilon = 1e-12
        heatmap_data = (top_vals + epsilon).reshape(k_use, 1)
        max_val = max(float(heatmap_data.max()), epsilon)
        heatmap_linear = heatmap_data / max_val

        positive_vals = heatmap_data[heatmap_data > 0]
        vmin = max(np.percentile(positive_vals, 5), epsilon) if positive_vals.size > 0 else epsilon
        vmax = max(np.percentile(heatmap_data, 99.5), vmin * 10.0)

        cmap = 'Reds'
        labels = [feature_names[i] if i < len(feature_names) else f"dim_{i}" for i in top_idx]
        log_path, linear_path = self._get_saliency_output_paths(save_path)

        for scale_name, data, norm, out_path, cbar_label, title_suffix in [
            (
                'log',
                heatmap_data,
                mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=False),
                log_path,
                '|Loss| (log scale)',
                'Log Scale',
            ),
            (
                'linear',
                heatmap_linear,
                mcolors.Normalize(vmin=0.0, vmax=1.0),
                linear_path,
                'Normalized |Loss|',
                'Linear Normalized',
            ),
        ]:
            fig, ax = plt.subplots(figsize=(6, max(3, k_use * 0.35)))
            im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest', norm=norm)

            ax.set_yticks(np.arange(k_use))
            ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
            ax.set_xticks([0])
            ax.set_xticklabels([f"{event_time:.2f}s" if event_time is not None else "event"], fontsize=11, fontweight='bold')
            ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold')

            title = f"Top Loss Dimensions (Single-Step, {title_suffix})"
            if event_time is not None:
                title += f" — t = {event_time:.2f}s"
            plt.title(title, fontsize=12, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, rotation=270, labelpad=15)

            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n[SAVED] Single-step loss heatmap ({scale_name}): {out_path}")

    def plot_top_k_saliency(self, window, window_igs, save_path, k=20):
        """
        Plots the Top K features in both logarithmic and linear-normalized scales.
        Highlights when saliency first appears.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        import torch

        # 1. Prepare Data
        saliency_matrix = torch.stack(window_igs).cpu()

        max_saliency_per_dim = saliency_matrix.abs().max(dim=0).values
        top_k_values, top_k_indices = torch.topk(max_saliency_per_dim, k)

        top_indices_np = top_k_indices.numpy()
        heatmap_data = saliency_matrix[:, top_indices_np].T.numpy()
        heatmap_data_abs = np.abs(heatmap_data)

        # 2. Log Scale Logic
        epsilon = 1e-12
        heatmap_data_safe = heatmap_data_abs + epsilon
        max_val = max(float(heatmap_data_safe.max()), epsilon)
        heatmap_data_linear = heatmap_data_safe / max_val

        vmin = 1e-5
        vmax = max(np.percentile(heatmap_data_safe, 99.5), vmin * 10.0)

        # Colormap: white below vmin
        cmap = 'Reds'
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=False)

        # Time axis
        times = [frame['time'] for frame in window]

        # --- Detect saliency onset ---
        saliency_mask = heatmap_data_safe > vmin          # non-white
        saliency_any_time = saliency_mask.any(axis=0)     # over features

        if saliency_any_time.any():
            saliency_start_idx = np.argmax(saliency_any_time)
            saliency_start_time = times[saliency_start_idx]
        else:
            saliency_start_idx = None
            saliency_start_time = None

        # 3. Plot (log and linear)
        feature_labels = [self.obs_names[i] for i in top_indices_np]
        tick_indices = np.linspace(0, len(times) - 1, 10, dtype=int)
        log_path, linear_path = self._get_saliency_output_paths(save_path)

        for scale_name, data, local_norm, out_path, title_suffix, cbar_label in [
            (
                'log',
                heatmap_data_safe,
                norm,
                log_path,
                'Log Scale',
                'Influence (log scale)',
            ),
            (
                'linear',
                heatmap_data_linear,
                mcolors.Normalize(vmin=0.0, vmax=1.0),
                linear_path,
                'Linear Normalized',
                'Normalized Influence',
            ),
        ]:
            fig, ax = plt.subplots(figsize=(12, k * 0.25))

            im = ax.imshow(
                data,
                cmap=cmap,
                aspect='auto',
                interpolation='nearest',
                norm=local_norm
            )

            if saliency_start_idx is not None:
                ax.axvline(
                    saliency_start_idx,
                    color='black',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.8
                )

            ax.set_yticks(np.arange(k))
            ax.set_yticklabels(feature_labels, fontsize=10, fontweight='bold')

            ax.set_xticks(tick_indices)
            ax.set_xticklabels(
                [f"{times[i]:.2f}" for i in tick_indices],
                fontsize=14,
                fontweight='bold'
            )
            ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')

            if saliency_start_time is not None:
                title = (
                    f"Top {k} Root Causes ({title_suffix}) — "
                    f"Saliency Onset: t = {saliency_start_time:.2f}s"
                )
            else:
                title = f"Top {k} Root Causes ({title_suffix}) — No Saliency Detected"

            plt.title(title, fontsize=14, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, rotation=270, labelpad=15)

            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\n[SAVED] Top-{k} heatmap ({scale_name}): {out_path}")

    def plot_grouped_saliency(self, window, window_igs, save_path):
        """Creates grouped saliency heatmaps in both log and linear-normalized scales."""
        import matplotlib.colors as mcolors
        groups = self.get_observation_groups()
        times = [frame['time'] for frame in window]
        
        # Aggregate saliency by group (max absolute value)
        group_names = ['Base (IMU)', 'Commands', 'Lower Body Sensor', 'Upper Body Sensor', 'Lower Body Action', 'Upper Body Action']
        group_saliencies = []
        
        for group_name in group_names:
            group_row = []
            indices = groups[group_name]
            for t_idx in range(len(window)):
                ig = window_igs[t_idx]
                if indices:
                    max_saliency = max([abs(ig[idx].item()) for idx in indices])
                    group_row.append(max_saliency)
                else:
                    group_row.append(0.0)
            group_saliencies.append(group_row)
        
        # Convert to numpy array for heatmap
        heatmap_data = np.array(group_saliencies, dtype=np.float32)
        epsilon = 1e-12
        heatmap_log = heatmap_data + epsilon
        max_val = max(float(heatmap_log.max()), epsilon)
        heatmap_linear = heatmap_log / max_val
        vmin = max(np.percentile(heatmap_log[heatmap_log > 0], 5), epsilon) if np.any(heatmap_log > 0) else epsilon
        vmax = max(np.percentile(heatmap_log, 99.5), vmin * 10.0)

        log_path, linear_path = self._get_saliency_output_paths(save_path)

        for scale_name, data, norm, out_path, cbar_label in [
            (
                'log',
                heatmap_log,
                mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=False),
                log_path,
                'Max Absolute Saliency (log scale)',
            ),
            (
                'linear',
                heatmap_linear,
                mcolors.Normalize(vmin=0.0, vmax=1.0),
                linear_path,
                'Max Absolute Saliency (normalized)',
            ),
        ]:
            fig, ax = plt.subplots(figsize=(30, 2.5))
            im = ax.imshow(data, cmap='Reds', aspect='auto', interpolation='nearest', norm=norm)

            ax.set_yticks(np.arange(len(group_names)))
            ax.set_yticklabels(group_names)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=11)
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n[SAVED] Grouped saliency heatmap ({scale_name}): {out_path}")
        
    def plot_joint_history(self, history, save_path="joint_positions.png", duration=4.0, intervals=20):
        """
        Plots a heatmap of JOINT POSITIONS (radians) over the last 'duration' seconds.
        Downsamples the time axis to exactly 'intervals' steps for clarity.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Filter Data (Last 4 seconds)
        if not history: return
        end_time = history[-1]['time']
        start_time = end_time - duration
        
        subset = [x for x in history if x['time'] >= start_time]
        if not subset:
            print(f"[ERROR] No history found for the last {duration}s")
            return

        # 2. Extract Joint Positions
        # Find indices in obs_names that start with 'pos_' (Joint Positions)
        joint_indices = [i for i, name in enumerate(self.obs_names) if name.startswith('pos_')]
        joint_names = [self.obs_names[i].replace('pos_', '') for i in joint_indices]
        
        if not joint_indices:
            print("[ERROR] No 'pos_' features found in obs_names.")
            return

        # Stack data: [Time, All_Obs]
        # Note: These are likely NORMALIZED values if coming from trace_buffer
        full_obs_matrix = torch.stack([x['obs'] for x in subset]).cpu().numpy()
        
        # Slice only joint positions: [Time, Joints]
        joint_matrix = full_obs_matrix[:, joint_indices]
        timestamps = np.array([x['time'] for x in subset])

        # 3. Downsample to Fixed Intervals (20 steps)
        # We use linspace to pick integer indices evenly spaced
        if len(subset) > intervals:
            indices = np.linspace(0, len(subset) - 1, intervals, dtype=int)
            data_resampled = joint_matrix[indices]
            time_resampled = timestamps[indices]
        else:
            # If we have fewer than 20 frames, just use what we have
            data_resampled = joint_matrix
            time_resampled = timestamps

        # 4. Plotting
        # Matrix Shape for Plot: [Joints, Time]
        plot_data = data_resampled.T
        
        fig, ax = plt.subplots(figsize=(10, len(joint_names) * 0.35)) # Dynamic height
        
        # Use 'RdBu_r' (Red-Blue) centered at 0 if normalized, or 'viridis' if raw
        im = ax.imshow(plot_data, cmap='RdBu_r', aspect='auto', interpolation='nearest')
        
        # Y-Axis: Joint Names
        ax.set_yticks(np.arange(len(joint_names)))
        ax.set_yticklabels(joint_names, fontsize=9)
        
        # X-Axis: Time (Formatted relative to end)
        ax.set_xticks(np.arange(len(time_resampled)))
        # Show relative time (e.g., -4.0s ... 0.0s)
        rel_times = time_resampled - time_resampled[-1] 
        ax.set_xticklabels([f"{t:.1f}s" for t in rel_times], rotation=45, fontsize=8)
        ax.set_xlabel("Time (Relative to End)")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Joint Position (Normalized)', rotation=270, labelpad=15)
        
        plt.title(f"Joint Kinematics (Last {duration}s)", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[EXPLAINER] Saved Joint Position heatmap to {save_path}")

    def plot_command_history(self, history, save_path="command_history.png", duration=15.0, intervals=20):
        """
        Plots a heatmap of command velocities over the last 'duration' seconds.
        Downsamples the time axis to exactly 'intervals' steps for clarity.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Filter Data (Last 'duration' seconds)
        if not history: return
        end_time = history[-1]['time']
        start_time = end_time - duration
        
        subset = [x for x in history if x['time'] >= start_time]
        if not subset:
            print(f"[ERROR] No history found for the last {duration}s")
            return

        # 2. Extract Command Velocities
        cmd_indices = [i for i, name in enumerate(self.obs_names) if name.startswith('cmd_')]
        cmd_names = [self.obs_names[i].replace('cmd_', '') for i in cmd_indices]
        
        if not cmd_indices:
            print("[ERROR] No 'cmd_' features found in obs_names.")
            return

        # Stack data: [Time, All_Obs]
        full_obs_matrix = torch.stack([x['obs'] for x in subset]).cpu().numpy()
        
        # Slice only command velocities: [Time, Commands]
        cmd_matrix = full_obs_matrix[:, cmd_indices]
        timestamps = np.array([x['time'] for x in subset])

        # 3. Downsample to Fixed Intervals (20 steps)
        if len(subset) > intervals:
            indices = np.linspace(0, len(subset) - 1, intervals, dtype=int)
            data_resampled = cmd_matrix[indices]
            time_resampled = timestamps[indices]
        else:
            data_resampled = cmd_matrix
            time_resampled = timestamps

        # 4. Plotting
        plot_data = data_resampled.T
        
        fig, ax = plt.subplots(figsize=(8, len(cmd_names) * 0.5)) 
        
        im = ax.imshow(plot_data, cmap='viridis', aspect='auto', interpolation='nearest')
        
        ax.set_yticks(np.arange(len(cmd_names)))
        ax.set_yticklabels(cmd_names, fontsize=10)
        
        ax.set_xticks(np.arange(len(time_resampled)))
        rel_times = time_resampled - time_resampled[-1] 
        ax.set_xticklabels([f"{t:.1f}s" for t in rel_times], rotation=45, fontsize=8)
        ax.set_xlabel("Time (Relative to End)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Command Velocity (Normalized)', rotation=270, labelpad=15)
        plt.title(f"Command Velocities (Last {duration}s)", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[EXPLAINER] Saved Command Velocity heatmap to {save_path}")

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


def normalize_loss_df(df_loss, obs_dims, timestamps=None, source_path="loss.csv"):
    """
    Normalize a loaded/computed loss dataframe into the exact layout expected by
    the thresholding code: columns obs_dims + timestamp, one row per obs row,
    with row 0 padded to zero so that loss[t] corresponds to obs[t].
    """
    df_loss = df_loss.copy()

    # Support common legacy naming.
    rename_map = {}
    if "time" in df_loss.columns and "timestamp" not in df_loss.columns:
        rename_map["time"] = "timestamp"
    df_loss = df_loss.rename(columns=rename_map)

    missing = [c for c in obs_dims if c not in df_loss.columns]
    if missing:
        raise ValueError(
            f"{source_path} is missing expected loss columns: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    if "timestamp" not in df_loss.columns:
        if timestamps is None:
            raise ValueError(f"{source_path} has no timestamp column and no fallback timestamps were provided.")
        df_loss["timestamp"] = np.asarray(timestamps)

    df_loss = df_loss[obs_dims + ["timestamp"]].copy()

    if timestamps is not None:
        target_len = len(timestamps)
        cur_len = len(df_loss)

        # Common case: saved losses are unaligned T-1 sequence losses (no leading pad).
        if cur_len == target_len - 1:
            pad_row = {c: 0.0 for c in obs_dims}
            pad_row["timestamp"] = timestamps[0]
            df_loss = pd.concat([pd.DataFrame([pad_row]), df_loss], ignore_index=True)

        # Trim or pad to exactly match obs length.
        if len(df_loss) > target_len:
            df_loss = df_loss.iloc[:target_len].reset_index(drop=True)
        elif len(df_loss) < target_len:
            pad_count = target_len - len(df_loss)
            pad = pd.DataFrame(0.0, index=np.arange(pad_count), columns=obs_dims)
            pad["timestamp"] = np.asarray(timestamps)[len(df_loss):]
            df_loss = pd.concat([df_loss, pad], ignore_index=True)

        # Force timestamps from observations so downstream indexing matches exactly.
        df_loss["timestamp"] = np.asarray(timestamps)

    return df_loss


def load_or_compute_seq_loss(
    obs_df,
    act_df,
    obs_dims,
    act_dims,
    explainer,
    loss_csv_path=None,
    recalculate_loss=True,
    verbose_prefix=""
):
    """
    Either recompute losses from the model or load them from CSV, then normalize
    them so downstream threshold logic is identical in either case.
    Returns:
        df_loss, obs_t, act_t
    """
    obs_t = torch.tensor(obs_df[obs_dims].values, dtype=torch.float32).to(explainer.device)
    act_t = torch.tensor(act_df[act_dims].values, dtype=torch.float32).to(explainer.device)

    def _compute():
        with torch.no_grad():
            out, _ = explainer.model(
                obs_t.unsqueeze(0).contiguous(),
                actions=act_t.unsqueeze(0).contiguous()
            )
            out = out.squeeze(0)

            # Align: prediction at t uses obs/action at t to predict obs at t+1.
            mu = out[:-1, :len(obs_dims)]
            log_var = out[:-1, len(obs_dims):]
            target = obs_t[1:]

            mse = ((target - mu) ** 2) / (torch.exp(log_var) + 1e-8)
            pad = torch.zeros((1, len(obs_dims)), device=explainer.device)
            nll_aligned = torch.cat([pad, mse], dim=0)

            df = pd.DataFrame(nll_aligned.cpu().numpy(), columns=obs_dims)
            df["timestamp"] = obs_df["timestamp"].values
            return df

    if (not recalculate_loss) and loss_csv_path and os.path.exists(loss_csv_path):
        try:
            df_loss = pd.read_csv(loss_csv_path)
            df_loss = normalize_loss_df(
                df_loss,
                obs_dims=obs_dims,
                timestamps=obs_df["timestamp"].values,
                source_path=loss_csv_path
            )
            print(f"[INFO] {verbose_prefix}loaded loss from {loss_csv_path}")
            return df_loss, obs_t, act_t
        except Exception as e:
            print(f"[WARN] {verbose_prefix}failed to load {loss_csv_path}: {e}")
            print(f"[WARN] {verbose_prefix}falling back to recomputing losses from model.")

    df_loss = _compute()

    # Optional cache so the next RECALCULATE_LOSS=False run works.
    if loss_csv_path:
        try:
            os.makedirs(os.path.dirname(loss_csv_path), exist_ok=True)
            df_loss.to_csv(loss_csv_path, index=False)
            print(f"[INFO] {verbose_prefix}saved loss cache to {loss_csv_path}")
        except Exception as e:
            print(f"[WARN] {verbose_prefix}could not save {loss_csv_path}: {e}")

    return df_loss, obs_t, act_t

def run_batch_testing():
    # ----------------------------------------------------------------------
    # STEP 1: ONE-TIME GLOBAL CALIBRATION
    # ----------------------------------------------------------------------
    print(f"\n{'='*80}\n[PHASE 1] GLOBAL CALIBRATION\n{'='*80}")
    print(f"Loading nominal baseline from {CAL_BASE}...")
    
    # Load raw calibration data
    df_cal_obs_raw = pd.read_csv(f'{CAL_BASE}/observations.csv')
    df_cal_act_raw = pd.read_csv(f'{CAL_BASE}/actions.csv')
    
    # Data is expected at 50Hz; consume every sample (no striding/downsampling)
    target_freq = 50.0
    expected_dt = 1.0 / target_freq
    actual_dt = df_cal_obs_raw['timestamp'].diff().median()
    stride = 1

    if pd.notna(actual_dt):
        print(f"[INFO] Calibration dt≈{actual_dt:.6f}s (target {expected_dt:.6f}s), using all samples.")

    # Apply Calibration Preprocessing (/0.25 scaling only)
    df_cal_obs = df_cal_obs_raw.reset_index(drop=True)
    df_cal_act = df_cal_act_raw.reset_index(drop=True)
    
    # Apply action scaling logic from your snippet
    for col in [c for c in df_cal_act.columns if c != 'timestamp']:
        df_cal_act[col] = df_cal_act[col] / 0.25

    obs_dims = [c for c in df_cal_obs.columns if c.startswith('dim_')]
    act_dims = [c for c in df_cal_act.columns if c != 'timestamp']
    
    # Initialize Model once
    explainer = StatefulExplainer(len(obs_dims), len(act_dims))
    
    # Loss handling is delegated to load_or_compute_seq_loss().

    # Calculate Calibration Thresholds
    df_cal_loss, _, _ = load_or_compute_seq_loss(
        df_cal_obs,
        df_cal_act,
        obs_dims,
        act_dims,
        explainer,
        loss_csv_path=CAL_LOSS_FILE,
        recalculate_loss=RECALCULATE_LOSS,
        verbose_prefix='[CAL] '
    )
    mean_nll_series = df_cal_loss[obs_dims].mean(axis=1)
    thresh_mean = mean_nll_series.max() + 3 * mean_nll_series.std()
    thresh_dim = df_cal_loss[obs_dims].max(axis=0) + (df_cal_loss[obs_dims].max(axis=0) - df_cal_loss[obs_dims].median(axis=0).fillna(0))*2.0
    
    cal_mins = df_cal_obs[obs_dims].min(axis=0)
    cal_maxs = df_cal_obs[obs_dims].max(axis=0)
    buffers = (cal_maxs - cal_mins) * 1.0
    thresh_obs_lower = cal_mins - buffers
    thresh_obs_upper = cal_maxs + buffers

    print(f"Calibration Complete. Mean Threshold: {thresh_mean:.6f}")
    save_npy(f"{CAL_BASE}/nll_thresh.npy", np.array(thresh_mean, dtype=np.float32))

    # If you also want the per-dimension thresholds:
    save_npy(f"{CAL_BASE}/nll_thresh_dim.npy", thresh_dim.values.astype(np.float32))

    explainer.obs_mean = torch.tensor(df_cal_obs[obs_dims].mean().values, dtype=torch.float32).to(explainer.device)
    explainer.obs_std = torch.tensor(df_cal_obs[obs_dims].std().values, dtype=torch.float32).to(explainer.device)
    explainer.calib_max = torch.tensor(df_cal_loss[obs_dims].max().values, dtype=torch.float32).to(explainer.device)
    explainer.calib_std = torch.tensor(df_cal_loss[obs_dims].std().values, dtype=torch.float32).to(explainer.device)
    
    # Mock the 'detector' object structure for the plot function
    class MockDetector:
        def __init__(self, e):
            self.obs_dim = e.obs_dim
            self.obs_mean = e.obs_mean
            self.obs_std = e.obs_std
            self.calib_max = e.calib_max
            self.calib_std = e.calib_std
            self.use_probabilistic = True # Assuming your model outputs mu/log_var

    explainer.detector = MockDetector(explainer)

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
        df_test_act_raw = pd.read_csv(f'{test_path}/actions.csv')
        
        df_test_obs = df_test_obs_raw.reset_index(drop=True)
        df_test_act = df_test_act_raw.reset_index(drop=True)
        min_len = min(len(df_test_obs), len(df_test_act))
        if len(df_test_obs) != len(df_test_act):
            print(f"[FIX] Truncating mismatched lengths: Obs({len(df_test_obs)}) vs Act({len(df_test_act)}) -> {min_len}")
            df_test_obs = df_test_obs.iloc[:min_len].reset_index(drop=True)
            df_test_act = df_test_act.iloc[:min_len].reset_index(drop=True)

        for col in act_dims: df_test_act[col] = df_test_act[col] / 0.25

        # Compute or load test losses and prepare tensors
        test_loss_file = f'{test_path}/loss.csv'
        df_test_loss, test_obs_tensor, test_act_tensor = load_or_compute_seq_loss(
            df_test_obs,
            df_test_act,
            obs_dims,
            act_dims,
            explainer,
            loss_csv_path=test_loss_file,
            recalculate_loss=RECALCULATE_LOSS,
            verbose_prefix='[TEST] '
        )
        
        start_time = df_test_obs['timestamp'].iloc[0]

        # 1. Evaluate NLL Anomaly
        test_mean_nll = df_test_loss[obs_dims].mean(axis=1)
        save_npy(f"{test_path}/nll_global.npy", test_mean_nll.values.astype(np.float32))
        nll_margin = (test_mean_nll.values - float(thresh_mean)).astype(np.float32)
        save_npy(f"{test_path}/nll_margin.npy", nll_margin)

        # --- Save command history (cmd_vel_x/y/yaw) ---
        cmd_cols = ["dim_6", "dim_7", "dim_8"]  # matches get_observation_names ordering: cmd_vel_x, cmd_vel_y, cmd_vel_yaw
        cmd_arr = df_test_obs[cmd_cols].values.astype(np.float32)
        save_npy(f"{test_path}/commands.npy", cmd_arr)

        risk_mean_global = test_mean_nll / thresh_mean
        risk_max_local = (df_test_loss[obs_dims] / thresh_dim).max(axis=1)
        model_risk_score = np.maximum(risk_max_local, risk_mean_global)
        
        nll_anomalies = model_risk_score > 1.0
        WARMUP_STEPS = 0
        nll_anomalies.iloc[:WARMUP_STEPS] = False
        first_nll_idx = nll_anomalies.idxmax() if nll_anomalies.any() else None

        # 2. Evaluate Range Anomaly
        lower_viols = (df_test_obs[obs_dims] < thresh_obs_lower).any(axis=1)
        upper_viols = (df_test_obs[obs_dims] > thresh_obs_upper).any(axis=1)
        range_anomalies = lower_viols | upper_viols
        range_anomalies.iloc[:WARMUP_STEPS] = False
        first_range_idx = range_anomalies.idxmax() if range_anomalies.any() else None

        # 3. Handle Dual-Trigger Termination
        if first_nll_idx is not None and first_range_idx is not None:
            n_time = df_test_obs.iloc[first_nll_idx]['timestamp'] - start_time
            r_time = df_test_obs.iloc[first_range_idx]['timestamp'] - start_time
            print(f"\n[TERMINATE] Both NLL and Range anomalies detected in {test_path}.")
            print(f"  NLL Trigger: {n_time:.4f}s (Idx {first_nll_idx}) | Range Trigger: {r_time:.4f}s (Idx {first_range_idx})")

        # 4. Standard Reporting & Saliency
        real_names = get_observation_names()
        name_map = {d: (real_names[i] if i < len(real_names) else d) for i, d in enumerate(obs_dims)}
        dim_to_idx = {d: i for i, d in enumerate(obs_dims)}

        if first_nll_idx is not None:
            row_idx = first_nll_idx
            rel_time = df_test_obs.iloc[row_idx]['timestamp'] - start_time
            risk_at_trigger = model_risk_score.iloc[row_idx]
            print(f"\n[FIRST HYBRID NLL ANOMALY DETECTED]")
            print(f"Time: {rel_time:.4f}s (Index {row_idx})")
            print(f"Hybrid Risk Score: {risk_at_trigger:.2f}")

            # Check if range caused this trigger
            if risk_at_trigger >= 1e9:
                pass#print("Trigger Source: Range Violation (Threshold exceeded by model via hybrid score)")
            if risk_max_local.iloc[row_idx] > risk_mean_global.iloc[row_idx]:
                print("Trigger Source: Local Dimension NLL Risk")
            else:
                print("Trigger Source: Global Mean NLL Risk")
            
            # Find bad dimensions for heatmap
            row_dim_losses = df_test_loss.iloc[row_idx][obs_dims]
            bad_dims = row_dim_losses[row_dim_losses > thresh_dim]
            bad_indices = [dim_to_idx[d] for d in bad_dims.index if d in dim_to_idx]
            
            if not bad_dims.empty:
                worst_dim = bad_dims.idxmax()
                val = bad_dims[worst_dim]
                thresh_val = thresh_dim[worst_dim]
                real_name = name_map.get(worst_dim, worst_dim)
                print(f"Type: Dimension NLL ('{real_name}' Val {val:.4f} > Threshold {thresh_val:.4f})")
                for dim_name in bad_dims.index:
                    if dim_name in dim_to_idx: bad_indices.append(dim_to_idx[dim_name])

            print(f"Running Saliency for {len(bad_indices)} affected dimensions...")
            trace_buffer = deque(maxlen=1500)
            explainer.hidden = None
            
            # Warm up hidden state up to trigger point
            lookback = 1500
            s_idx = max(0, row_idx - lookback)
            with torch.no_grad():
                for i in range(s_idx, row_idx + 1):
                    current_hidden = explainer.hidden.detach().clone() if explainer.hidden is not None else None
                    o, a = test_obs_tensor[i], test_act_tensor[i]
                    _, explainer.hidden = explainer.model(o.view(1,1,-1), actions=a.view(1,1,-1), hidden=explainer.hidden)
                    trace_buffer.append({'obs': o, 'act': a, 'hidden': current_hidden, 'time': df_test_obs.iloc[i]['timestamp'] - start_time})
            
            plot_save_path = f"{test_path}/grouped_saliency.png"
            heatmap_result = explainer.explain_heatmap(
                trace_buffer,
                anomaly_indices=bad_indices,
                save_path=plot_save_path,
                fallback_loss_vector=row_dim_losses,
                fallback_time=rel_time,
            )

            ig_matrix, target_window = (heatmap_result if heatmap_result is not None else (None, None))

            if ig_matrix is not None:
                sal = ig_matrix.detach().cpu().numpy().astype(np.float32)     # [H_sal, D]
                sal_abs = np.abs(sal)

                # Choose a stable Top-K set (recommended): top-k by max over time in the window
                TOP_K = 8
                max_per_dim = sal_abs.max(axis=0)                              # [D]
                topk_idx = np.argsort(-max_per_dim)[:TOP_K].astype(np.int32)   # [K]

                # Save indices for overlay script
                save_npy(f"{test_path}/saliency_topk_idx.npy", topk_idx)

                # Save labels in the SAME order as indices
                obs_names = get_observation_names()
                topk_names = [obs_names[i] if i < len(obs_names) else f"dim_{i}" for i in topk_idx.tolist()]
                with open(f"{test_path}/saliency_topk_names.json", "w") as f:
                    json.dump(topk_names, f, indent=2)

                # Optional: save the abs saliency too (bars usually want magnitude)
                save_npy(f"{test_path}/saliency_abs.npy", sal_abs)

            margin_save_path = f"{test_path}/safety_margin.png"
            # Calculate start/end relative to the anomaly time
            trigger_time = df_test_obs.iloc[row_idx]['timestamp'] - start_time
            
            explainer.plot_safety_margin_trajectory(
                trace_buffer, 
                start_time=0, # Show 2 seconds before
                end_time=99999999999999999,   # Show 0.5 seconds after
                save_path=margin_save_path
            )

            # plot joint positions over last 10 steps:
            joint_pos_dims = [d for d in obs_dims if 'pos_' in name_map.get(d, d)]
            joint_positions = df_test_obs.iloc[max(0, row_idx-10):row_idx+1][joint_pos_dims]
            time_stamps = df_test_obs.iloc[max(0, row_idx-10):row_idx+1]['timestamp'] - start_time
            plt.figure(figsize=(12, 6))
            for dim in joint_pos_dims:
                plt.plot(time_stamps, joint_positions[dim], label=name_map.get(dim, dim))
            plt.xlabel("Time (s)")
            plt.ylabel("Joint Positions")
            plt.title("Joint Positions Leading Up to Anomaly")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            joint_plot_path = f"{test_path}/joint_positions.png"
            plt.savefig(joint_plot_path, dpi=300)
            plt.close()
            print(f"[SAVED] Joint positions plot: {joint_plot_path}") 

            joint_save_path = f"{test_path}/joint_history.png"
            explainer.plot_joint_history(
                trace_buffer, 
                save_path=joint_save_path, 
                duration=4.0,   # Last 4 seconds
                intervals=20    # Downsample to 20 columns
            )
            explainer.plot_command_history(
                trace_buffer, 
                save_path=f"{test_path}/command_history.png", 
                duration=15.0,  # Last 15 seconds
                intervals=20    # Downsample to 20 columns
            )

        if first_range_idx is not None:
            rel_time = df_test_obs.iloc[first_range_idx]['timestamp'] - start_time
            print(f"\n[REPORT] Range Anomaly at {rel_time:.4f}s.")

            row_idx = first_range_idx
            relative_time = df_test_loss.iloc[row_idx]['timestamp'] - start_time
            
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
1. motor failure; 2. sensor failure; 3. sensor noise; 4. motor dynamics mismatch; 5. initial state issue (e.g. pose, joint positions or velocities); 6. observation scaling issue; 7. observation ordering issue; 8. joint outside max limits; 9. ground friction mismatch; 10. ground deformability mismatch (sand, mattress); 11. external force (collision, push); 12. mass distribution mismatch robot; 13. policy latency (constant offset or not); 14. coordinate frame mismatch with IMU; 15. sensor drift; 16. contact model mismatch; 17. passive mechanical resistance / joint movement constrained; 18. policy action scaling or mismatch; 19. payload mismatch. 20. power supply lag; 21. other (please specify).

[Saliency and OOD information]:'''
    #print(txt)
    run_batch_testing()