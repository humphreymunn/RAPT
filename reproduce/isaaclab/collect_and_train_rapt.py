import argparse
import json
import time
import h5py
import numpy as np
from datetime import datetime
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from isaaclab.app import AppLauncher
import cli_args  # add argparse arguments

parser = argparse.ArgumentParser(description="Collect observations and train RAPT (FDM objective or pure reconstruction).")

# ============== Mode Switch ==============
parser.add_argument("--train_dynamics", action="store_true",default=True,
                    help="If True, trains as Forward Dynamics Model (s_t, a_t -> s_{t+1}). "
                         "If False, trains as Autoencoder (s_t -> s_t).")

# ============== Ablation / Model Arguments ==============
parser.add_argument("--reconstruction_type", type=str, default="bottleneck", choices=["masked", "bottleneck"],
                    help="Type of restriction (Only applies if --train_dynamics is False).")
parser.add_argument("--mask_ratio", type=float, default=0.25,
                    help="Severity. For 'masked': % inputs dropped. For 'bottleneck': % latent reduction.")
parser.add_argument("--use_residual", action="store_true", default=True,
                    help="Use Residual connections in MLPs.")
parser.add_argument("--use_probabilistic", action="store_true", default=True,
                    help="Output Mean+Var (NLL Loss) instead of Mean (MSE).")
parser.add_argument("--use_temporal", action="store_true", default=True,
                    help="Use a GRU to model temporal history.")

# ============== Training Arguments ==============
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--embed_dim", type=int, default=256, help="Base embedding dimension.")
parser.add_argument("--num_blocks", type=int, default=4, help="Depth of Encoder/Decoder MLPs.")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
parser.add_argument("--noise_scale", type=float, default=0.01, help="Input noise scale.")
parser.add_argument("--num_epochs", type=int, default=100, help="Epochs.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for temporal training.")

# ============== Pipeline Arguments ==============
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--use_critic_multi", action="store_true", default=False)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--collection_time", type=int, default=600)
parser.add_argument("--output_dir", type=str, default="collected_data")
parser.add_argument("--skip_collection", action="store_true")
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--skip_training", action="store_true")

parser.add_argument(
    "--distance_throw",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override env cfg `distance_throw` (e.g. --distance-throw / --no-distance-throw).",
)

parser.add_argument(
    "--no_dr",
    action=argparse.BooleanOptionalAction,
    default=None,
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

torch.backends.cudnn.benchmark = True


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
    def __init__(self, obs_dim, action_dim, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args
        self.use_temporal = args.use_temporal
        self.embed_dim = args.embed_dim
        self.train_dynamics = args.train_dynamics

        self.input_dim = obs_dim + action_dim if self.train_dynamics else obs_dim

        if args.reconstruction_type == 'bottleneck':
            keep_ratio = 1.0 - args.mask_ratio
            self.latent_dim = max(4, int(args.embed_dim * keep_ratio))
            print(f"[MODEL] AE Bottleneck Mode: {args.embed_dim} -> {self.latent_dim}")
        else:
            self.latent_dim = args.embed_dim
            if self.train_dynamics:
                print(f"[MODEL] Dynamics Mode: Input {self.input_dim} -> Output {obs_dim}")
            else:
                print(f"[MODEL] AE Masked Mode: Constant width {args.embed_dim}")

        encoder_layers = [nn.Linear(self.input_dim, self.embed_dim), nn.ReLU()]
        for _ in range(args.num_blocks):
            encoder_layers.append(ConfigurableBlock(self.embed_dim, self.embed_dim, args.use_residual, args.dropout))
        self.encoder_mlp = nn.Sequential(*encoder_layers)

        if self.use_temporal:
            print("[MODEL] Using GRU Backbone")
            self.gru = nn.GRU(self.embed_dim, self.embed_dim, num_layers=1, batch_first=True)

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

        decoder_layers = []
        for _ in range(args.num_blocks):
            decoder_layers.append(ConfigurableBlock(self.embed_dim, self.embed_dim, args.use_residual, args.dropout))
        self.decoder_mlp = nn.Sequential(*decoder_layers)

        out_features = obs_dim * 2 if args.use_probabilistic else obs_dim
        self.head = nn.Linear(self.embed_dim, out_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

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

        mask = None
        x_in = x_flat
        
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
        
        if self.train_dynamics:
            if actions is None: raise ValueError("Dynamics model requires 'actions' input!")
            
            if is_sequence: act_flat = actions.reshape(-1, actions.shape[-1])
            else: act_flat = actions
            
            if x_in.shape[:-1] != act_flat.shape[:-1]:
                raise ValueError(f"Shape mismatch: Obs {x_in.shape} vs Act {act_flat.shape}")

            # Concat Masked Obs + Clean Actions
            model_input = torch.cat([x_in, act_flat], dim=-1)
        else:
            model_input = x_in

        z = self.encoder_mlp(model_input)

        if self.use_temporal:
            if not is_sequence:
                z = z.unsqueeze(1)
            else:
                z = z.view(batch, seq, -1)

            z, hidden = self.gru(z, hidden)
            z = z.reshape(-1, z.shape[-1])

        if self.use_bottleneck:
            z = self.compress(z)
            z = self.decompress(z)

        z = self.decoder_mlp(z)
        out = self.head(z)

        if is_sequence:
            out = out.view(batch, seq, -1)

        return out, mask, hidden

class UniversalTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, log_dir, obs_mean, obs_std, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        self.scaler = torch.amp.GradScaler('cuda')
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        self.obs_mean = obs_mean.to(device)
        self.obs_std = obs_std.to(device)
        self.best_val_loss = float('inf')

    def process_batch(self, batch):
        if self.args.train_dynamics:
            obs = batch[0].to(self.device, non_blocking=True)
            act = batch[1].to(self.device, non_blocking=True)
            target = batch[2].to(self.device, non_blocking=True) # Next Obs

            # Normalize Obs and Target
            obs = (obs - self.obs_mean) / self.obs_std
            target = (target - self.obs_mean) / self.obs_std
            
            if self.args.noise_scale > 0:
                obs_in = obs + torch.randn_like(obs) * self.args.noise_scale
            else:
                obs_in = obs

            return obs_in, act, target
        else:
            # Batch: [Obs]
            obs = batch[0].to(self.device, non_blocking=True)
            # Normalize
            obs = (obs - self.obs_mean) / self.obs_std
            
            # Input Noise for AE Denoising
            if self.args.noise_scale > 0:
                obs_in = obs + torch.randn_like(obs) * self.args.noise_scale
            else:
                obs_in = obs
            
            # Target is the input obs
            return obs_in, None, obs

    def loss_function(self, output, target, mask):
        obs_dim = self.model.obs_dim

        use_masked_loss = (self.args.reconstruction_type == 'masked') and \
                              (mask is not None) and \
                              (not self.args.train_dynamics) 

        if self.model.args.use_probabilistic:
            mu = output[..., :obs_dim]
            log_var = output[..., obs_dim:]

            log_var = torch.clamp(log_var, min=-6.0, max=6.0)
            precision = torch.exp(-log_var)
            mse = (mu - target) ** 2
            nll = 0.5 * (precision * mse + log_var)

            
            if use_masked_loss:
                masked_loss = (nll * mask.float()).sum() / (mask.sum() + 1e-6)
                unmasked_loss = (nll * (~mask).float()).mean()
                loss = masked_loss + 0.1 * unmasked_loss
            else:
                loss = nll.mean()

            return loss, mse.mean()
        else:
            # Deterministic
            mu = output
            mse = (mu - target) ** 2

            if use_masked_loss:
                masked_loss = (mse * mask.float()).sum() / (mask.sum() + 1e-6)
                unmasked_loss = (mse * (~mask).float()).mean()
                loss = masked_loss + 0.1 * unmasked_loss
            else:
                loss = mse.mean()

            return loss, mse.mean()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_mse = 0.0, 0.0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]"):
            obs_in, act_in, target = self.process_batch(batch)
            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # Pass actions if dynamics model
                output, mask, _ = self.model(obs_in, actions=act_in)
                loss, mse = self.loss_function(output, target, mask)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item()
            total_mse += mse.item()

        return total_loss / len(self.train_loader), total_mse / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_loss, total_mse = 0.0, 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                obs_in, act_in, target = self.process_batch(batch)
                with torch.amp.autocast('cuda'):
                    output, mask, _ = self.model(obs_in, actions=act_in)
                    loss, mse = self.loss_function(output, target, mask)
                total_loss += loss.item()
                total_mse += mse.item()
        return total_loss / len(self.val_loader), total_mse / len(self.val_loader)

    def save(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, filename))

    def run(self):
        mode = "DYNAMICS (s,a -> s')" if self.args.train_dynamics else f"AE ({self.args.reconstruction_type.upper()})"
        print(f"\nConfig: {mode} | Res={self.args.use_residual} | Prob={self.args.use_probabilistic} | Temp={self.args.use_temporal}")
        
        for epoch in range(1, self.args.num_epochs + 1):
            t_loss, t_mse = self.train_epoch(epoch)
            v_loss, v_mse = self.validate(epoch)
            print(f"Epoch {epoch}: Train Loss={t_loss:.4f} | Val MSE={v_mse:.4f}")
            if v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                self.save('best_model.pth')
        self.save('final_model.pth')


# used for velocity task (removing past historic observations)
def remove_per_feature_history(obs, history_len=5):
    term_dims = [3, 3, 3, 29, 29, 29]
    
    slices = []
    cursor = 0
    
    for dim in term_dims:
        # Calculate the size of this term's entire history block
        block_size = dim * history_len
        
        # Start: cursor + (block_size - dim)
        # End:   cursor + block_size
        start_idx = cursor + block_size - dim
        end_idx = cursor + block_size
        
        # Slice and store
        slices.append(obs[:, start_idx:end_idx])
        
        # Move cursor to the start of the next feature block
        cursor += block_size

    return torch.cat(slices, dim=1)

def collect_observations(args_cli, env_cfg, agent_cfg):
    """Collect observations (and actions/next_obs) from the RL agent."""
    
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, multihead=args_cli.use_critic_multi)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    if args_cli.task == 'Unitree-G1-29dof-Throwing': # no curriculum needed for evaluation
        env.unwrapped.distance_range = [2., env_cfg.max_throw_dist] #if args_cli.task == 'Unitree-G1-29dof-Throwing' else env.unwrapped.distance_range
        env.unwrapped.update_curriculum = lambda _: None  # Disable curriculum updates during collection
        print(f"[INFO] Environment distance range set to: {env.unwrapped.distance_range} for evaluation")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    collected_obs = []
    collected_acts = []
    collected_next_obs = [] # Only used for dynamics

    metadata = {
        'task_name': args_cli.task,
        'num_envs': env.num_envs,
        'collection_time_seconds': args_cli.collection_time,
        'is_dynamics': args_cli.train_dynamics
    }

    obs, _ = env.get_observations()
    obs_policy = obs.clone()
    remove_history = lambda x: remove_per_feature_history(x) if args_cli.task == "Unitree-G1-29dof-Velocity" else x
    obs = remove_history(obs)
    print(f"\n{'=' * 60}")
    print("STARTING DATA COLLECTION")
    print(f"{'=' * 60}")

    start_time = time.time()
    timestep = 0

    while simulation_app.is_running():
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= args_cli.collection_time:
            break

        with torch.inference_mode():

            current_obs = obs.clone()
            actions = policy(obs_policy)
            next_obs, rewards, dones, _ = env.step(actions)
            obs_policy = next_obs.clone()
            next_obs = remove_history(next_obs)

            # Store data
            collected_obs.append(current_obs)
            
            if args_cli.train_dynamics:
                collected_acts.append(actions.clone())
                collected_next_obs.append(next_obs.clone())
            
            obs = next_obs
            timestep += 1

            if timestep % 100 == 0:
                print(f"[INFO] Steps: {timestep}, Time: {elapsed_time:.2f}s")

    print(f"\n[INFO] Collection complete! Total steps: {timestep}")

    # Stack
    obs_tensor = torch.stack(collected_obs) # [T, N, D]
    if args_cli.train_dynamics:
        act_tensor = torch.stack(collected_acts)  # [T, N, A]
        next_obs_tensor = torch.stack(collected_next_obs)  # [T, N, D]
    else:
        act_tensor = None
        next_obs_tensor = None

    env.close()
    return obs_tensor, act_tensor, next_obs_tensor, metadata


def save_collected_data(obs, act, next_obs, metadata, output_dir, task_name):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{task_name}_data_{timestamp}.h5")
    
    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset('observations', data=obs.cpu().numpy(), compression="gzip")
        if act is not None:
            hf.create_dataset('actions', data=act.cpu().numpy(), compression="gzip")
        if next_obs is not None:
            hf.create_dataset('next_observations', data=next_obs.cpu().numpy(), compression="gzip")
        
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset('metadata', data=json.dumps(metadata, default=str), dtype=dt)
        print(f"[INFO] Data saved to: {filepath}")
    return filepath


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "DYN" if args_cli.train_dynamics else "AE"
    log_dir = os.path.join(args_cli.output_dir, f"{prefix}_{args_cli.task}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # collect dataset
    if args_cli.skip_collection and args_cli.data_path:
        with h5py.File(args_cli.data_path, 'r') as f:
            obs_tensor = torch.from_numpy(f['observations'][:])
            if 'actions' in f:
                act_tensor = torch.from_numpy(f['actions'][:])
                next_obs_tensor = torch.from_numpy(f['next_observations'][:])
            else:
                act_tensor, next_obs_tensor = None, None
    else:
        from rsl_rl.runners import OnPolicyRunner
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
        if args_cli.task == 'Unitree-G1-29dof-Velocity': # no curriculum needed for evaluation
            env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
        agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        
        if args_cli.no_dr is not None:
            env_cfg.arm_dr_range = 0.0
            env_cfg.events = None

        obs_tensor, act_tensor, next_obs_tensor, metadata = collect_observations(args_cli, env_cfg, agent_cfg)
        
        if obs_tensor is None:
            print("[ERROR] Data collection failed!")
            return
        
        task_name = args_cli.task.replace("/", "_").replace(":", "_") if args_cli.task else "unknown"
        data_path = save_collected_data(obs_tensor, act_tensor, next_obs_tensor, metadata, log_dir, task_name)

    if args_cli.distance_throw is not None:
        if hasattr(env_cfg, "distance_throw"):
            env_cfg.distance_throw = args_cli.distance_throw
        elif hasattr(env_cfg, "cfg") and hasattr(env_cfg.cfg, "distance_throw"):
            env_cfg.cfg.distance_throw = args_cli.distance_throw
        else:
            raise AttributeError("Couldn't find `distance_throw` in the loaded env cfg.")

   
    
    
    # Normalize Stats
    obs_tensor = obs_tensor.float()
    if args_cli.use_temporal:
        obs_flat = obs_tensor.view(-1, obs_tensor.shape[-1])
        obs_mean = obs_flat.mean(dim=0)
        obs_std = obs_flat.std(dim=0) + 1e-6
    else:
        obs_mean = obs_tensor.mean(dim=0)
        obs_std = obs_tensor.std(dim=0) + 1e-6

    # Save per-timestep min/max as matrices over [time_step, obs_dim]
    # For collected rollouts obs_tensor is typically [T, N, D], so reduce over env dim.
    # If data is already flattened [S, D], keep it as a single-timestep matrix [1, D].
    if obs_tensor.dim() == 3:
        obs_min_ts, _ = obs_tensor.min(dim=1)  # [T, D]
        obs_max_ts, _ = obs_tensor.max(dim=1)  # [T, D]
    elif obs_tensor.dim() == 2:
        obs_min_ts = obs_tensor.min(dim=0).values.unsqueeze(0)  # [1, D]
        obs_max_ts = obs_tensor.max(dim=0).values.unsqueeze(0)  # [1, D]
    else:
        raise ValueError(f"Expected obs_tensor to have 2 or 3 dims, got shape {tuple(obs_tensor.shape)}")

    with h5py.File(os.path.join(log_dir, 'obs_stats.h5'), 'w') as f:
        f.create_dataset('mean', data=obs_mean.cpu().numpy())
        f.create_dataset('std', data=obs_std.cpu().numpy())
        f.create_dataset('min', data=obs_min_ts.cpu().numpy())
        f.create_dataset('max', data=obs_max_ts.cpu().numpy())
        f.create_dataset('min_per_timestep', data=obs_min_ts.cpu().numpy())
        f.create_dataset('max_per_timestep', data=obs_max_ts.cpu().numpy())

    stats_data = {
        "mean": obs_mean.cpu().numpy().tolist(),
        "std": obs_std.cpu().numpy().tolist(),
        "min": obs_min_ts.cpu().numpy().tolist(),
        "max": obs_max_ts.cpu().numpy().tolist(),
        "min_per_timestep": obs_min_ts.cpu().numpy().tolist(),
        "max_per_timestep": obs_max_ts.cpu().numpy().tolist()
    }
    
    stats_path = os.path.join(log_dir, 'obs_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)

    # Shape Data for Training
    if args_cli.use_temporal:
        T, N, D = obs_tensor.shape
        seq_len = args_cli.seq_len
        act_tensor_dyn = None
        next_obs_tensor_dyn = None

        if args_cli.train_dynamics and (act_tensor is None or next_obs_tensor is None):
            raise RuntimeError("Temporal dynamics mode requires action and next-observation tensors.")
        if args_cli.train_dynamics:
            act_tensor_dyn = act_tensor
            next_obs_tensor_dyn = next_obs_tensor

        obs_seqs = []
        act_seqs = []
        next_obs_seqs = []

        for env in range(N):
            # --- Observations ---
            env_obs = obs_tensor[:, env, :]          # [T, D]
            num_seq = env_obs.shape[0] // seq_len
            env_obs = env_obs[:num_seq * seq_len]
            env_obs = env_obs.view(num_seq, seq_len, D)
            obs_seqs.append(env_obs)

            if args_cli.train_dynamics:
                assert act_tensor_dyn is not None and next_obs_tensor_dyn is not None
                # --- Actions ---
                env_act = act_tensor_dyn[:, env, :]      # [T, A]
                env_act = env_act[:num_seq * seq_len]
                env_act = env_act.view(num_seq, seq_len, -1)
                act_seqs.append(env_act)

                # --- Next observations ---
                env_next = next_obs_tensor_dyn[:, env, :]  # [T, D]
                env_next = env_next[:num_seq * seq_len]
                env_next = env_next.view(num_seq, seq_len, D)
                next_obs_seqs.append(env_next)

        obs_tensor = torch.cat(obs_seqs, dim=0)  # [N*num_seq, seq_len, D]

        if args_cli.train_dynamics:
            act_tensor = torch.cat(act_seqs, dim=0)
            next_obs_tensor = torch.cat(next_obs_seqs, dim=0)

        print(f"[INFO] Temporal data reshaped per-env:")
        print(f"       Obs: {obs_tensor.shape}")
        if args_cli.train_dynamics:
            assert act_tensor_dyn is not None and next_obs_tensor_dyn is not None
            print(f"       Act: {act_tensor_dyn.shape}")
            print(f"       Next: {next_obs_tensor_dyn.shape}")

    # Create Dataset
    if args_cli.train_dynamics:
        if act_tensor is None or next_obs_tensor is None:
            raise RuntimeError("Dynamics training requires action and next-observation tensors.")
        dataset = TensorDataset(obs_tensor, act_tensor, next_obs_tensor)
        action_dim = act_tensor.shape[-1]
    else:
        dataset = TensorDataset(obs_tensor)
        action_dim = 0

    # Train
    if not args_cli.skip_training:
        train_len = int(0.8 * len(dataset))
        train, val = random_split(dataset, [train_len, len(dataset) - train_len])

        train_loader = DataLoader(train, batch_size=args_cli.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val, batch_size=args_cli.batch_size, shuffle=False, num_workers=0)

        # Note: UniversalModel handles whether input is Obs or Obs+Act
        model = UniversalModel(obs_dim=obs_tensor.shape[-1], action_dim=action_dim, args=args_cli).to(args_cli.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=args_cli.lr)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args_cli.lr, epochs=args_cli.num_epochs, steps_per_epoch=len(train_loader))

        trainer = UniversalTrainer(model, train_loader, val_loader, optimizer, scheduler, args_cli.device, log_dir, obs_mean, obs_std, args_cli)
        trainer.run()
        
        # Save config
        with open(os.path.join(log_dir, 'experiment_config.json'), 'w') as f:
            json.dump(vars(args_cli), f, indent=2, default=str)
        
        # Compute per-dimension NLL loss on full calibration dataset
        try:
            model.eval()
            D = obs_tensor.shape[-1]

            # Flatten dataset to [S, D] on CPU to avoid a giant GPU allocation.
            obs_flat = obs_tensor.view(-1, D).detach().float().cpu()
            if args_cli.train_dynamics:
                if act_tensor is None or next_obs_tensor is None:
                    raise RuntimeError("Dynamics calibration export requires both actions and next observations.")
                act_flat = act_tensor.view(-1, act_tensor.shape[-1]).detach().float().cpu()
                target_flat = next_obs_tensor.view(-1, D).detach().float().cpu()
            else:
                act_flat = None
                target_flat = obs_flat

            obs_mean_dev = obs_mean.detach().to(args_cli.device)
            obs_std_dev = obs_std.detach().to(args_cli.device)
            num_rows = obs_flat.shape[0]
            chunk_size = max(128, min(4096, args_cli.batch_size * 8))

            # Write CSV
            import csv
            csv_path = os.path.join(log_dir, 'calibration_dataset_loss.csv')
            header = ['timestamp'] + [f"dim_{i}" for i in range(D)]

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

                with torch.no_grad():
                    for start_idx in range(0, num_rows, chunk_size):
                        end_idx = min(start_idx + chunk_size, num_rows)

                        obs_batch = obs_flat[start_idx:end_idx].to(args_cli.device, non_blocking=True)
                        obs_norm = (obs_batch - obs_mean_dev.unsqueeze(0)) / obs_std_dev.unsqueeze(0)

                        if args_cli.train_dynamics:
                            if act_flat is None:
                                raise RuntimeError("Missing actions for dynamics calibration export.")
                            act_batch = act_flat[start_idx:end_idx].to(args_cli.device, non_blocking=True)
                            target_batch = target_flat[start_idx:end_idx].to(args_cli.device, non_blocking=True)
                            target_norm = (target_batch - obs_mean_dev.unsqueeze(0)) / obs_std_dev.unsqueeze(0)
                        else:
                            act_batch = None
                            target_norm = obs_norm

                        output, _, _ = model(obs_norm, actions=act_batch, force_no_mask=True)

                        if model.args.use_probabilistic:
                            mu = output[..., :D]
                            log_var = output[..., D:]
                            log_var = torch.clamp(log_var, min=-6.0, max=6.0)
                            precision = torch.exp(-log_var)
                            sq_err = (mu - target_norm) ** 2
                            nll_batch = precision * sq_err
                        else:
                            nll_batch = (output - target_norm) ** 2

                        nll_np = nll_batch.detach().cpu().numpy()
                        for row_offset, row_vals in enumerate(nll_np):
                            row_idx = start_idx + row_offset
                            writer.writerow([str(row_idx)] + [f"{v:.6f}" for v in row_vals.tolist()])

                        del obs_batch, obs_norm, output, target_norm, nll_batch
                        if args_cli.train_dynamics:
                            del act_batch, target_batch

            print(f"[INFO] Calibration CSV saved to: {csv_path} with {num_rows} observations")
        except Exception as e:
            print(f"[WARN] Failed to save calibration CSV: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()