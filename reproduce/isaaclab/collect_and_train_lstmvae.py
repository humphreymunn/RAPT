# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to collect observations and train a Recurrent D-VAE with Progress Prior.
- Sequence-Based: Preserves LSTM memory (Stateless=False)
- Progress Prior: Latent target moves from p_start to p_end over time
"""

"""Launch Isaac Sim Simulator first."""
from sklearn.svm import SVR
import joblib
import argparse
from importlib.metadata import version
import json
import h5py
import numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect observations and train Recurrent D-VAE.")

# ============== Collection Arguments ==============
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--use_critic_multi", action="store_true", default=False)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--collection_time", type=int, default=60, help="Collection time in seconds.")
parser.add_argument("--output_dir", type=str, default="collected_data", help="Directory to save collected data and models.")

# ============== D-VAE Training Arguments ==============
parser.add_argument("--batch_size", type=int, default=64, help="Batch size (Number of Environments/Trajectories per batch).")
parser.add_argument("--latent_dim", type=int, default=24, help="Dimension of the latent space (z).")
parser.add_argument("--hidden_dims", type=int, nargs='+', default=[256], help="Hidden dimension for LSTM layers.")
parser.add_argument("--kl_weight", type=float, default=0.0001, help="Weight for KL Divergence loss.")
parser.add_argument("--noise_scale", type=float, default=0.1, help="Sigma noise for input corruption.")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers.")

# ============== Pipeline Control ==============
parser.add_argument("--skip_collection", action="store_true", help="Skip collection and use existing data file.")
parser.add_argument("--data_path", type=str, default=None, help="Path to existing data file (required if --skip_collection).")
parser.add_argument("--skip_training", action="store_true", help="Skip VAE training (only collect data).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# --- OPTIMIZATION: Enable CuDNN benchmarking ---
torch.backends.cudnn.benchmark = True

# ============== Recurrent D-VAE Model with Progress Prior ==============

class RecurrentDenoisingVAE(nn.Module):
    def __init__(self, obs_dim, max_time_steps, latent_dim=3, hidden_dims=[256], noise_scale=0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.noise_scale = noise_scale
        self.hidden_dim = hidden_dims[0]
        self.max_time_steps = float(max_time_steps)
        
        # --- Progress Prior Parameters ---
        # Learnable Start and End points for the prior mean
        self.p_start = nn.Parameter(torch.zeros(latent_dim))
        self.p_end = nn.Parameter(torch.randn(latent_dim))
        
        # --- Encoder (LSTM) ---
        # Takes full sequence: (Batch, Seq_Len, Obs_Dim)
        self.encoder_lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.fc_z_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_z_var = nn.Linear(self.hidden_dim, latent_dim)
        
        # --- Decoder (LSTM) ---
        self.decoder_projection = nn.Linear(latent_dim, self.hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.fc_x_mu = nn.Linear(self.hidden_dim, obs_dim)
        self.fc_x_var = nn.Linear(self.hidden_dim, obs_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def get_prior_mean(self, batch_size, seq_len, device):
        """
        Calculates mu_p(t) for the entire sequence at once.
        Returns: (Batch, Seq_Len, Latent)
        """
        # Create a time vector [0, 1, 2, ... T-1]
        t = torch.arange(seq_len, device=device).float()
        
        # Calculate progress ratio alpha [T, 1]
        alpha = (t / self.max_time_steps).unsqueeze(1)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        
        # Interpolate: p_start + alpha * (p_end - p_start)
        # diff: [1, Latent]
        diff = (self.p_end - self.p_start).unsqueeze(0)
        
        # prior_seq: [T, Latent]
        prior_seq = self.p_start + alpha * diff 
        
        # Expand to batch dimension: [Batch, T, Latent]
        return prior_seq.unsqueeze(0).expand(batch_size, -1, -1)

    def encode(self, x):
        # x: (Batch, Seq_Len, Obs_Dim)
        
        # LSTM processes the whole sequence automatically, maintaining internal state
        lstm_out, _ = self.encoder_lstm(x)
        
        # Activation (Tanh as discussed)
        h = torch.tanh(lstm_out)
        
        # Apply Linear heads to every timestep in the sequence
        z_mu = self.fc_z_mu(h)
        z_logvar = self.fc_z_var(h)
        return z_mu, z_logvar
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # z: (Batch, Seq_Len, Latent)
        
        hidden_input = self.decoder_projection(z)
        
        lstm_out, _ = self.decoder_lstm(hidden_input)
        
        h = torch.tanh(lstm_out)
        
        x_mu = torch.sigmoid(self.fc_x_mu(h))
        # Variance Head -> Softplus (Corrected from Tanh)
        x_var = F.softplus(self.fc_x_var(h)) + 1e-3
        return x_mu, x_var
    
    def forward(self, x):
        # x: (Batch, Seq_Len, Obs_Dim)
        
        if self.training:
            noise = torch.randn_like(x) * self.noise_scale
            x_tilde = x + noise
        else:
            x_tilde = x 
            
        z_mu, z_logvar = self.encode(x_tilde)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_var = self.decode(z)
        x_logvar = torch.log(x_var)
        
        # Calculate Prior for the full sequence
        # We assume the input sequence starts at t=0. 
        # (This is valid because we batch full trajectories)
        prior_mu = self.get_prior_mean(x.shape[0], x.shape[1], x.device)
        
        return x_mu, x_logvar, z_mu, z_logvar, prior_mu

# remove historic observations for velocity task
def remove_per_feature_history(obs, history_len=5):
    term_dims = [3, 3, 3, 29, 29, 29]
    
    slices = []
    cursor = 0
    
    for dim in term_dims:
        block_size = dim * history_len
        
        start_idx = cursor + block_size - dim
        end_idx = cursor + block_size
        
        # Slice and store
        slices.append(obs[:, start_idx:end_idx])
        
        # Move cursor to the start of the next feature block
        cursor += block_size

    return torch.cat(slices, dim=1)

# collect dataset
def collect_observations(args_cli, env_cfg, agent_cfg):
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path: return None, None
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, multihead=args_cli.use_critic_multi)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    collected_observations = []
    episode_length = env.unwrapped.max_episode_length
    
    metadata = {
        'task_name': args_cli.task,
        'num_envs': env.num_envs,
        'max_episode_length': episode_length,
        'structure': '[Num_Envs, Time_Steps, Obs_Dim]'
    }

    obs, _ = env.get_observations()
    obs_policy = obs.clone()
    remove_history = lambda x: remove_per_feature_history(x) if args_cli.task == "Unitree-G1-29dof-Velocity" else x
    obs = remove_history(obs)
    metadata['observation_shape'] = list(obs.shape)
    
    print(f"\n{'='*60}")
    print("STARTING DATA COLLECTION")
    print(f"{'='*60}")
    
    start_time = time.time()
    timestep = 0
    
    while simulation_app.is_running():
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= args_cli.collection_time: break
        
        with torch.inference_mode():

            collected_observations.append(obs.clone())
            actions = policy(obs_policy)
            obs, rewards, dones, _ = env.step(actions)
            obs_policy = obs.clone()
            obs = remove_history(obs)

        
        timestep += 1
        if timestep % 100 == 0:
            print(f"[INFO] Steps: {timestep}, Time: {elapsed_time:.2f}s")
    
    # Shape: [Time, Num_Envs, Dims]
    obs_tensor = torch.stack(collected_observations) 
    # Permute to: [Num_Envs, Time, Dims] (Trajectories)
    obs_tensor = obs_tensor.permute(1, 0, 2)
    
    env.close()
    return obs_tensor, metadata


def save_collected_data(obs_tensor, metadata, output_dir, task_name):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{task_name}_observations_{timestamp}.h5")
    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset('observations', data=obs_tensor.cpu().numpy(), compression="gzip")
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset('metadata', data=json.dumps(metadata, default=str), dtype=dt)
    print(f"[INFO] Data saved to: {filepath}")
    return filepath


def load_collected_data(filepath):
    print(f"[INFO] Loading data from {filepath}...")
    with h5py.File(filepath, 'r') as hf:
        obs_data = torch.from_numpy(hf['observations'][:]).float()
        metadata = json.loads(hf['metadata'][()])
    print(f"[INFO] Loaded data shape {obs_data.shape}.")
    return obs_data, metadata


class SequenceDVAETrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        log_dir,
        obs_min,
        obs_range,
        scheduler=None,
        kl_weight=0.0001
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.kl_weight = kl_weight
        
        # Normalization params need to be reshaped for broadcasting
        # Obs: (Batch, Time, Dim) -> Params: (1, 1, Dim)
        self.obs_min = obs_min.to(device).view(1, 1, -1)
        self.obs_range = obs_range.to(device).view(1, 1, -1)
        
        self.scaler = torch.amp.GradScaler('cuda')
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.log_dir = log_dir

    def normalize_batch(self, obs):
        # uses min-max normalization as stated in original paper
        return (obs - self.obs_min) / self.obs_range

    def negative_log_likelihood(self, x, x_mu, x_logvar):
        x_logvar = torch.clamp(x_logvar, min=-6.0, max=6.0)
        inverse_var = torch.exp(-x_logvar)
        mse_weighted = (x - x_mu)**2 * inverse_var
        loss_per_dim = x_logvar + mse_weighted
        return 0.5 * torch.mean(loss_per_dim)

    def progress_kl_divergence(self, z_mu, z_logvar, prior_mu):
        z_var = torch.exp(z_logvar)
        # Sum over latent dimension (-1)
        trace_term = torch.sum(z_var, dim=-1) 
        mean_diff_term = torch.sum((prior_mu - z_mu)**2, dim=-1)
        k_dim = z_mu.shape[-1]
        log_det_term = torch.sum(z_logvar, dim=-1)
        
        kl = 0.5 * (trace_term + mean_diff_term - k_dim - log_det_term)
        return torch.mean(kl) # Average over batch and time

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        # Batch is (Batch_Size, Time, Dims)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for obs in pbar:
            obs = obs[0].to(self.device, non_blocking=True)
            
            obs = self.normalize_batch(obs)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                # Forward pass takes whole sequence
                # Timesteps are implicit (0...T) inside model for batch
                x_mu, x_logvar, z_mu, z_logvar, prior_mu = self.model(obs)
                
                recon_loss = self.negative_log_likelihood(obs, x_mu, x_logvar)
                kld_loss = self.progress_kl_divergence(z_mu, z_logvar, prior_mu)
                loss = recon_loss + self.kl_weight * kld_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/total_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/recon_loss', recon_loss.item(), self.global_step)
                self.writer.add_scalar('train/kld_loss', kld_loss.item(), self.global_step)
            
            self.global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for obs in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                obs = obs[0].to(self.device, non_blocking=True)
                obs = self.normalize_batch(obs)
                
                with torch.amp.autocast('cuda'):
                    x_mu, x_logvar, z_mu, z_logvar, prior_mu = self.model(obs)
                    recon_loss = self.negative_log_likelihood(obs, x_mu, x_logvar)
                    kld_loss = self.progress_kl_divergence(z_mu, z_logvar, prior_mu)
                    loss = recon_loss + self.kl_weight * kld_loss
                    total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('val/total_loss', avg_loss, epoch)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, filename):
        filepath = os.path.join(self.log_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, filepath)

    def train(self, num_epochs, save_freq=10):
        print(f"\nStarting Recurrent D-VAE training on {self.device}...")
        best_train_loss = float('inf')
        patience = 1000
        counter = 0
        # ---------------------------
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            print(f"Epoch {epoch}: Train={train_loss:.5f} | Val={val_loss:.5f}")
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                counter = 0 
                self.save_checkpoint(epoch, val_loss, 'best_model.pth')
            else:
                counter += 1
                
            if counter >= patience:
                break
            
            if epoch % save_freq == 0:
                self.save_checkpoint(epoch, val_loss, f'checkpoint_epoch_{epoch}.pth')

        self.save_checkpoint(num_epochs, val_loss, 'final_model.pth')
        self.writer.close()


def collect_anomaly_statistics(model, data_loader, device, obs_min, obs_range, log_dir):
    print("\n" + "="*60)
    print("COLLECTING ANOMALY STATISTICS (NLL SCORE)")
    print("="*60)
    
    model.eval()
    all_scores = []
    
    # Obs min/range need to be reshaped for broadcasting
    obs_min = obs_min.to(device).view(1, 1, -1)
    obs_range = obs_range.to(device).view(1, 1, -1)
    
    with torch.no_grad():
        for obs in tqdm(data_loader, desc="Collecting stats"):
            obs = obs[0].to(device, non_blocking=True)
            obs = (obs - obs_min) / obs_range
            
            x_mu, x_logvar, _, _, _ = model(obs)
            
            x_logvar = torch.clamp(x_logvar, min=-6.0, max=6.0)
            inverse_var = torch.exp(-x_logvar)
            mse_weighted = (obs - x_mu)**2 * inverse_var
            nll_per_dim = 0.5 * (x_logvar + mse_weighted)
            
            # Sum over dims. Result is [Batch, Time]
            score = nll_per_dim.sum(dim=-1).cpu().numpy()
            all_scores.extend(score.flatten())
    
    all_scores = np.array(all_scores)
    
    stats = {
        'mean_score': float(np.mean(all_scores)),
        'std_score': float(np.std(all_scores)),
        'percentiles': {
            '90': float(np.percentile(all_scores, 90)),
            '95': float(np.percentile(all_scores, 95)),
            '99': float(np.percentile(all_scores, 99)),
            '99.9': float(np.percentile(all_scores, 99.9)),
        },
    }
    
    stats_path = os.path.join(log_dir, 'anomaly_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n[INFO] Stats saved to: {stats_path}")
    print(f"  Mean NLL Score: {stats['mean_score']:.4f}")
    
    return stats

def train_svr_detector(model, val_loader, device, obs_min, obs_range, log_dir, C=1.0, epsilon=0.1):
    """
    Implements Algorithm 1: Training the SVR Anomaly Detector.
    Maps latent state z -> expected anomaly score s.
    """
    print("\n" + "="*60)
    print("TRAINING SVR ANOMALY DETECTOR (Algorithm 1)")
    print("="*60)
    
    model.eval()
    
    # Storage for Z (State) and S (Score)
    Z_list = []
    S_list = []
    
    # Params for Normalization
    obs_min = obs_min.to(device).view(1, 1, -1)
    obs_range = obs_range.to(device).view(1, 1, -1)

    print("[INFO] Extracting Latent States (Z) and Anomaly Scores (S) from Validation Set...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extraction"):
            # Get Data
            obs = batch[0].to(device, non_blocking=True)
            
            # Normalize (using Xtrain stats) -> min-max normalization as per paper
            obs = (obs - obs_min) / obs_range
            
            # Forward Pass (Reconstruct)
            # x_mu, x_logvar are parameters of reconstructed distribution
            # z_mu is the "State" (Line 7 of Alg 1)
            x_mu, x_logvar, z_mu, _, _ = model(obs)
            
            # Calculate Anomaly Score s (Negative Log Likelihood) - (Line 9 of Alg 1)
            # NLL = 0.5 * (log_var + (x - mu)^2 / var)
            x_logvar = torch.clamp(x_logvar, min=-6.0, max=6.0)
            inverse_var = torch.exp(-x_logvar)
            mse_weighted = (obs - x_mu)**2 * inverse_var
            nll_per_dim = 0.5 * (x_logvar + mse_weighted)
            
            # Score is sum over dimensions
            s = nll_per_dim.sum(dim=-1) # Shape: [Batch, Time]
            
            # Flatten Time Dimension
            # The SVR maps a single point z_t -> s_t
            # We flatten [Batch, Time, ...] -> [Batch*Time, ...]
            z_flat = z_mu.reshape(-1, z_mu.shape[-1]).cpu().numpy()
            s_flat = s.reshape(-1).cpu().numpy()
            
            Z_list.append(z_flat)
            S_list.append(s_flat)
            
    # Concatenate all batches
    Z = np.concatenate(Z_list, axis=0)
    S = np.concatenate(S_list, axis=0)
    
    print(f"[INFO] Training Data Ready. Shapes: Z={Z.shape}, S={S.shape}")
    # SVR cannot handle >20-50k samples reasonably.
    MAX_SAMPLES = 20000 
    
    if len(Z) > MAX_SAMPLES:
        print(f"[WARN] Dataset is too large for SVR ({len(Z)} samples).")
        print(f"[INFO] Downsampling to {MAX_SAMPLES} random samples to allow training to finish...")
        
        # Randomly choose indices
        indices = np.random.choice(len(Z), MAX_SAMPLES, replace=False)
        Z = Z[indices]
        S = S[indices]
        print(f"[INFO] Downsampled Shapes: Z={Z.shape}, S={S.shape}")

    print(f"[INFO] Fitting SVR (RBF Kernel)...")
    
    # Train SVR
    # "We use support vector regression (SVR) ... using a radial basis function (RBF) kernel."
    svr = SVR(kernel='rbf', C=C, epsilon=epsilon)
    svr.fit(Z, S)
    
    print("[INFO] SVR Training Complete.")
    
    # Save SVR Model
    svr_path = os.path.join(log_dir, 'svr_model.joblib')
    joblib.dump(svr, svr_path)
    print(f"[INFO] SVR Model saved to: {svr_path}")
    
    return svr

def train_pipeline(obs_tensor, metadata, args_cli, log_dir):
    print(f"\n{'='*60}")
    print("STARTING RECURRENT D-VAE TRAINING (SEQUENCE)")
    print(f"{'='*60}")
    
    device = args_cli.device
    torch.manual_seed(42)
    
    # Input Shape: [Num_Envs, Time, Dims]
    # We DO NOT flatten, treat each env as a sequence.
    obs_tensor = obs_tensor.cpu()
    num_envs, time_steps, dims = obs_tensor.shape
    
    print(f"[INFO] Training on {num_envs} trajectories of length {time_steps}")
    
    # Calculate Min-Max (across all dims/times)
    # We flatten just to compute these stats easily
    flat_data = obs_tensor.reshape(-1, dims)
    obs_min = flat_data.min(dim=0)[0]
    obs_max = flat_data.max(dim=0)[0]
    obs_range = obs_max - obs_min
    obs_range[obs_range < 1e-6] = 1.0
    
    # Save stats
    stats_path = os.path.join(log_dir, 'obs_stats.h5')
    with h5py.File(stats_path, 'w') as f:
        f.create_dataset('min', data=obs_min.numpy())
        f.create_dataset('range', data=obs_range.numpy())
    
    # Dataset is just the tensor of trajectories
    dataset = TensorDataset(obs_tensor)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoader batches ENVIRONMENTS
    # Batch size 64 means "64 full trajectories at once"
    train_loader = DataLoader(
        train_dataset, batch_size=args_cli.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args_cli.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    max_ep_len = metadata.get('max_episode_length', time_steps)

    model = RecurrentDenoisingVAE(
        obs_dim=dims,
        max_time_steps=max_ep_len,
        latent_dim=args_cli.latent_dim,
        hidden_dims=args_cli.hidden_dims, 
        noise_scale=args_cli.noise_scale
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args_cli.lr, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, 
    )

    trainer = SequenceDVAETrainer(
        model, train_loader, val_loader, optimizer,
        device, log_dir, obs_min, obs_range,
        scheduler=scheduler,
        kl_weight=args_cli.kl_weight
    )
    
    trainer.train(args_cli.num_epochs)

    # We use the validation loader for this as per Algorithm 1 (Lines 4-12 use Xval)
    train_svr_detector(
        model, 
        val_loader, 
        device, 
        obs_min, 
        obs_range, 
        log_dir,
        C=1.0 # irrelevant to training
    )

    collect_anomaly_statistics(model, val_loader, device, obs_min, obs_range, log_dir)
    return model

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = args_cli.task.replace("/", "_").replace(":", "_") if args_cli.task else "unknown"
    log_dir = os.path.join(args_cli.output_dir, f"{task_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    if args_cli.skip_collection:
        if args_cli.data_path is None: raise ValueError("Data path required")
        obs_tensor, metadata = load_collected_data(args_cli.data_path)
    else:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point")
        if args_cli.task == 'Unitree-G1-29dof-Velocity': # no curriculum needed for evaluation
            env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        obs_tensor, metadata = collect_observations(args_cli, env_cfg, agent_cfg)
        if obs_tensor is None: return
        save_collected_data(obs_tensor, metadata, log_dir, task_name)
        with open(os.path.join(log_dir, 'collection_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    if not args_cli.skip_training:
        model = train_pipeline(obs_tensor, metadata, args_cli, log_dir)
    
    print(f"\n[INFO] Completed. Outputs in {log_dir}")
    if not args_cli.skip_training:
        print(f"  --dvae_model {os.path.join(log_dir, 'best_model.pth')}")

if __name__ == "__main__":
    main()
    simulation_app.close()