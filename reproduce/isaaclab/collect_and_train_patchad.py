"""
Script to Collect Data from Isaac Lab and Train using the OFFICIAL PatchAD Solver.
FIXES: 
1. Initializes SimSolver attributes correctly.
2. Wraps PatchAD model to fix 'einops' dimension mismatch (expands buffers).
3. Enforces list length consistency to prevent IndexError.
4. [NEW] Uses Standardization (Mean/Std) instead of Min-Max Normalization.
"""

import argparse
import os
import sys
import time
import json
import gc
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from datetime import datetime

# --- SETUP PATHS ---
sys.path.append(os.getcwd()) 
sys.path.append(os.path.join(os.getcwd(), 'scripts', 'rsl_rl', 'patchad'))

from isaaclab.app import AppLauncher
import cli_args

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Collect Data & Train PatchAD (Official).")

# Simulation
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--use_critic_multi", action="store_true", default=False)
parser.add_argument("--collection_time", type=int, default=60, help="Seconds to collect data.")
parser.add_argument("--output_dir", type=str, default="trained_patchad_models", help="Save directory.")
parser.add_argument(
    "--train_device",
    type=str,
    default="auto",
    choices=["auto", "cpu", "cuda"],
    help="Device for PatchAD training. 'auto' prefers CUDA only if enough free VRAM remains.",
)

# PatchAD (copied from code-base)
parser.add_argument("--win_size", "-ws", type=int, default=105)
parser.add_argument("--stride", "-st", type=int, default=10) 
parser.add_argument("--batch_size", "-bs", type=int, default=32)
parser.add_argument("--epochs", '-ep', type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--patch_size", type=str, default="[3,5]")
parser.add_argument("--d_model", type=int, default=40)
parser.add_argument("--e_layer", type=int, default=3)
parser.add_argument("--patch_mx", type=float, default=0.2)
parser.add_argument("--cont_beta", type=float, default=1.0)
parser.add_argument("--anormly_ratio", type=float, default=1.0)

# Pipeline
parser.add_argument("--skip_collection", action="store_true")
parser.add_argument("--data_path", type=str, default=None)

# RSL-RL
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import unitree_rl_lab.tasks  # noqa: F401

patchad_parent = os.path.abspath(os.path.join(os.getcwd(), 'scripts', 'rsl_rl'))
if patchad_parent not in sys.path:
    sys.path.insert(0, patchad_parent)


def choose_train_device(requested_device: str, min_free_gib: float = 2.0, collected_with_sim: bool = False) -> str:
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        if torch.cuda.is_available():
            if collected_with_sim:
                print("[WARN] Training on CUDA after live simulation may OOM due to Isaac Sim VRAM residency.")
            return "cuda"
        print("[WARN] --train_device=cuda requested but CUDA is unavailable. Falling back to CPU.")
        return "cpu"

    if collected_with_sim:
        print("[INFO] Data was collected with live simulation; using CPU for PatchAD training in auto mode.")
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gib = free_bytes / (1024 ** 3)
        if free_gib >= min_free_gib:
            return "cuda"
        print(
            f"[WARN] CUDA free memory is low ({free_gib:.2f} GiB). "
            "Using CPU for PatchAD training to avoid OOM."
        )
        return "cpu"
    except Exception as exc:
        print(f"[WARN] Could not query CUDA free memory ({exc}). Using CPU for safety.")
        return "cpu"

# Try Import Official Solver & Model
from scripts.rsl_rl.patchad.trainer.patchad_trainer_v2 import Solver
from scripts.rsl_rl.patchad.patchad_model.models import PatchMLPAD

# Model Wrapper
#  
class BatchExpandingPatchMLPAD(PatchMLPAD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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

        # Unpack original lists
        dist_num = ret[0]
        dist_size = ret[1]
        mx_num = ret[2]
        mx_size = ret[3]
        rec_x = ret[4]

        # Expand Memory Matrices
        p_num_mx_list = expand_list(mx_num)
        p_size_mx_list = expand_list(mx_size)
        
        # Ensure memory lists are not longer than distribution lists
        if len(p_num_mx_list) > len(dist_size):
            p_num_mx_list = p_num_mx_list[:len(dist_size)]
            
        if len(p_size_mx_list) > len(dist_num):
            p_size_mx_list = p_size_mx_list[:len(dist_num)]

        # Return reconstructed tuple
        return (dist_num, dist_size, p_num_mx_list, p_size_mx_list, rec_x)

# dataset wrapper
class SlidingWindowDataset(Dataset):
    def __init__(self, data, win_size, stride):
        self.data = data
        self.win_size = win_size
        self.stride = stride
        self.n_windows = (len(data) - win_size) // stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.win_size
        labels = torch.zeros(self.win_size, dtype=torch.long)
        return self.data[start:end], labels

class SimSolver(Solver):
    """Inherits Solver but uses in-memory data and our fixed Model Wrapper."""
    def __init__(self, config, train_loader, val_loader, input_c):
        self.__dict__.update(Solver.DEFAULTS, **config)
        
        self.cont_beta = 1.0
        self.dataset = self.data_name
        self.patch_size = self.patch_sizes
        self.num_epochs = self.epochs
        self.lr = self.learning_rate
        
        # Paths
        self.model_save_path = os.path.join(config['model_save_path'], self.data_name)
        self.res_pth = os.path.join(config['res_pth'], self.data_name)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.res_pth, exist_ok=True)
        
        # Inject Loaders
        self.train_loader = train_loader
        self.vali_loader = val_loader
        self.test_loader = val_loader 
        self.thre_loader = val_loader 
        
        self.input_c = input_c
        
        # Initialize Model
        self.build_model()

    def build_model(self):
        self.model = BatchExpandingPatchMLPAD(
            win_size=self.win_size, 
            e_layer=self.e_layer, 
            patch_sizes=self.patch_size, 
            dropout=0.0, 
            activation="relu", 
            output_attention=True,
            channel=self.input_c,
            d_model=self.d_model,
            cont_model=self.win_size,
            norm='n' 
        )
        
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

# used for velocity task to remove history per feature
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

def collect_data(args, env_cfg, agent_cfg):
    print(f"\n{'='*60}\nSTARTING DATA COLLECTION\n{'='*60}")
    
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv): env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, multihead=args.use_critic_multi)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    collected_data = []

    obs, _ = env.get_observations()
    obs_policy = obs.clone()
    remove_history = lambda x: remove_per_feature_history(x) if args_cli.task == "Unitree-G1-29dof-Velocity" else x
    obs = remove_history(obs)
    
    start_time = time.time()
    steps = 0
    
    while simulation_app is not None and simulation_app.is_running():
        if time.time() - start_time >= args.collection_time: break
        
        with torch.inference_mode():
            collected_data.append(obs.cpu().clone())
            actions = policy(obs_policy)
            obs, rewards, dones, _ = env.step(actions)
            obs_policy = obs.clone()
            obs = remove_history(obs)
        
        steps += 1
        if steps % 100 == 0:
            print(f"Collecting... {steps} steps")

    env.close()

    if len(collected_data) == 0:
        raise RuntimeError("No data was collected. Check simulation runtime/conditions.")
    
    full_tensor = torch.stack(collected_data) # [Time, B, D]
    flat_data = full_tensor.reshape(-1, full_tensor.shape[-1])
    return flat_data

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = args_cli.task if args_cli.task else "SimData"
    save_dir = os.path.join(args_cli.output_dir, f"{task_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Get Data
    if args_cli.skip_collection and args_cli.data_path:
        print("[INFO] Loading from file...")
        with h5py.File(args_cli.data_path, 'r') as f:
            data = torch.from_numpy(f['observations'][:])
            if data.dim() == 3: data = data.reshape(-1, data.shape[-1])
    else:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
        if args_cli.task == 'Unitree-G1-29dof-Velocity': # no curriculum needed for evaluation
            env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
        agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        data = collect_data(args_cli, env_cfg, agent_cfg)
        print("[INFO] Data collection complete. Proceeding to PatchAD training...")
        
        with h5py.File(os.path.join(save_dir, "observations.h5"), 'w') as f:
            f.create_dataset('observations', data=data.numpy())

    print("[INFO] Standardizing (Z-score)...")
    
    obs_mean = data.mean(dim=0)
    obs_std = data.std(dim=0)
    
    obs_std[obs_std < 1e-6] = 1.0 
    
    data_norm = (data - obs_mean) / obs_std
    
    # Save statistics for Inference
    with h5py.File(os.path.join(save_dir, "obs_stats.h5"), 'w') as f:
        f.create_dataset('mean', data=obs_mean.numpy())
        f.create_dataset('std', data=obs_std.numpy())
        # Also saving min/max for range detector
        f.create_dataset('min', data=data.min(dim=0)[0].numpy())
        f.create_dataset('max', data=data.max(dim=0)[0].numpy())

    dataset = SlidingWindowDataset(data_norm, args_cli.win_size, args_cli.stride)
    total_len = len(dataset)
    indices = list(range(total_len))
    split = int(np.floor(0.2 * total_len))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_loader = DataLoader(dataset, batch_size=args_cli.batch_size, 
                              sampler=SubsetRandomSampler(train_indices), num_workers=0)
    val_loader = DataLoader(dataset, batch_size=args_cli.batch_size, 
                            sampler=SubsetRandomSampler(val_indices), num_workers=0)

    # 4. Solver Config
    train_device = choose_train_device(
        args_cli.train_device,
        collected_with_sim=(not args_cli.skip_collection),
    )

    config = {
        'data_path': '', 
        'data_name': task_name,
        'model_save_path': save_dir,
        'res_pth': save_dir,
        'device': train_device,
        'win_size': args_cli.win_size,
        'stride': args_cli.stride,
        'batch_size': args_cli.batch_size,
        'epochs': args_cli.epochs,
        'anormly_ratio': args_cli.anormly_ratio,
        'learning_rate': args_cli.lr,
        'patch_sizes': eval(args_cli.patch_size) if isinstance(args_cli.patch_size, str) else args_cli.patch_size,
        'd_model': args_cli.d_model,
        'e_layer': args_cli.e_layer,
        'save_model': 1,
        'full_res': 0,
        'mode': 'train',
        'patch_mx': args_cli.patch_mx,
        'cont_beta': args_cli.cont_beta,
        'input_c': data.shape[1],
        'seed': 42
    }

    print(f"\n{'='*60}\nINITIALIZING OFFICIAL SOLVER\n{'='*60}")
    print(f"[INFO] PatchAD training device: {config['device']}")
    
    solver = SimSolver(config, train_loader, val_loader, input_c=data.shape[1])
    
    print("======== Train ========")
    solver.train()
    
    # Save Final Checkpoint
    src = os.path.join(save_dir, task_name, f"{task_name}_checkpoint.pth")
    dst = os.path.join(save_dir, "PSM_checkpoint.pth")
    import shutil
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"\n[INFO] Final checkpoint copied to: {dst}")
        print(f"[INFO] Obs Stats saved to: {os.path.join(save_dir, 'obs_stats.h5')}")
    else:
        print(f"[WARN] Could not find checkpoint at {src}")

if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()