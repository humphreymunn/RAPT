"""Collect public RAPT datasets from Isaac Lab: nominal train split + labeled OOD test split.

For each task this rolls out the expert policy and produces two files in the
RAPT release's common sequence format (ragged .npz, float16):

  train.npz  — nominal-only episodes (seq_%05d [T,D], act_%05d [T,A], dim_names)
  test.npz   — per-category OOD episodes with paired nominal control envs
               (+ labels, onset, fault arrays compatible with rapt evaluate.py)

Examples (run with the Isaac Lab python):
  python scripts/rsl_rl/collect_rapt_datasets.py --task Unitree-G1-29dof-Velocity \
      --load_run 2026-01-04_14-25-06_multi_pcgrad_1213 --use_critic_multi \
      --out datasets/g1_velocity --headless
  python scripts/rsl_rl/collect_rapt_datasets.py --task Unitree-G1-29dof-Mimic-Dance-102 \
      --policy_onnx deploy/robots/g1_29dof/config/policy/mimic/dance_102/exported/policy.onnx \
      --out datasets/g1_mimic_dance102 --headless
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect RAPT train/test datasets.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--policy_onnx", type=str, default=None,
                    help="Exported policy.onnx (used instead of an rsl_rl checkpoint).")
parser.add_argument("--use_critic_multi", action="store_true", default=False)
parser.add_argument("--train_steps", type=int, default=2400,
                    help="Nominal collection steps per env for the train split (48 s @ 50 Hz).")
parser.add_argument("--skip_train", action="store_true",
                    help="Skip the train split (reuse an existing train.npz).")
parser.add_argument("--test_episode_len", type=int, default=None,
                    help="Steps per OOD test episode (default: paper protocol — "
                         "1000 for velocity, 1500 otherwise).")
parser.add_argument("--ood_start_step", type=int, default=50,
                    help="Injection onset step within each test episode.")
parser.add_argument("--cal_episodes", type=int, default=2,
                    help="Nominal calibration batches at test conditions -> calibration.npz.")
parser.add_argument("--skip_test", action="store_true",
                    help="Skip the OOD test sweep (e.g. to extend calibration only).")
parser.add_argument("--categories", type=str, default=None,
                    help="Comma-separated OOD category names to collect (default all). "
                         "With --merge_test, replaces those categories in an existing test.npz.")
parser.add_argument("--merge_test", action="store_true",
                    help="Merge into existing test.npz: keep other categories' sequences, "
                         "replace the selected ones (new nominal controls are not added).")
parser.add_argument("--min_seq_len", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

# prefer the repo's rsl_rl fork (supports multihead critics) over site-packages
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rsl_rl"))

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401  (register tasks)

from ood_injection_lib import CATEGORY_NAMES, OODCategory, OODInjector


class LegacyRslRlWrapper(RslRlVecEnvWrapper):
    """Adapt the new TensorDict wrapper API to the repo's rsl_rl fork
    (obs tensor + extras['observations'] dict, 4-tuple step)."""

    def get_observations(self):
        td = super().get_observations()
        return td["policy"], {"observations": dict(td)}

    def reset(self):
        td, extras = super().reset()
        return td["policy"], extras

    def step(self, actions):
        td, rew, dones, extras = super().step(actions)
        extras = dict(extras)
        extras.setdefault("observations", dict(td))
        return td["policy"], rew, dones, extras

G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

TERM_DIMS_VELOCITY = [3, 3, 3, 29, 29, 29]
HISTORY_LEN = 5


def velocity_dim_names():
    names = ["root_vel_x", "root_vel_y", "root_vel_z",
             "gravity_x", "gravity_y", "gravity_z",
             "cmd_vel_x", "cmd_vel_y", "cmd_vel_yaw"]
    for prefix in ("pos_", "vel_", "action_"):
        names += [prefix + j for j in G1_JOINTS]
    return names


def mimic_dim_names():
    names = ["ref_pos_" + j for j in G1_JOINTS] + ["ref_vel_" + j for j in G1_JOINTS]
    names += [f"ref_anchor_rot_6d_{i}" for i in range(6)]
    names += ["root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z"]
    for prefix in ("pos_", "vel_", "action_"):
        names += [prefix + j for j in G1_JOINTS]
    return names


def remove_per_feature_history(obs, history_len=HISTORY_LEN):
    slices, cursor = [], 0
    for dim in TERM_DIMS_VELOCITY:
        block = dim * history_len
        slices.append(obs[:, cursor + block - dim: cursor + block])
        cursor += block
    return torch.cat(slices, dim=1)


def format_feature_history(obs_queue, term_dims):
    stack = torch.stack(list(obs_queue), dim=1)  # [B, H, D]
    feats = torch.split(stack, term_dims, dim=-1)
    return torch.cat([f.reshape(f.shape[0], -1) for f in feats], dim=1)


class OnnxPolicy:
    """Batched inference on an exported policy.onnx (batch dim patched dynamic)."""

    def __init__(self, path, device):
        import onnx
        import onnxruntime as ort

        model = onnx.load(path)
        for tensor in list(model.graph.input) + list(model.graph.output):
            tensor.type.tensor_type.shape.dim[0].dim_param = "batch"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.detach().cpu().numpy().astype(np.float32)
        (out,) = self.session.run(None, {self.input_name: x})
        return torch.from_numpy(out).to(self.device)


def load_policy(env, device):
    if args_cli.policy_onnx:
        print(f"[INFO] Using ONNX policy: {args_cli.policy_onnx}")
        return OnnxPolicy(args_cli.policy_onnx, device), args_cli.policy_onnx
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading rsl_rl checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device,
                            multihead=args_cli.use_critic_multi)
    runner.load(resume_path)
    return runner.get_inference_policy(device=device), resume_path


def save_ragged(path, sequences, actions, dim_names, extras=None):
    out = {"dim_names": np.array(dim_names)}
    for i, (o, a) in enumerate(zip(sequences, actions)):
        out[f"seq_{i:05d}"] = o
        out[f"act_{i:05d}"] = a
    if extras:
        out.update(extras)
    np.savez_compressed(path, **out)
    mb = os.path.getsize(path) / 1e6
    print(f"[SAVE] {path}: {len(sequences)} sequences, {mb:.1f} MB")


def main():
    is_velocity = args_cli.task == "Unitree-G1-29dof-Velocity"
    strip = remove_per_feature_history if is_velocity else (lambda x: x)

    try:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point",
        )
    except TypeError:  # older isaaclab_tasks: no entry_point_key kwarg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        env_cfg = load_cfg_from_registry(args_cli.task, "play_env_cfg_entry_point")
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
    if is_velocity:
        env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = LegacyRslRlWrapper(env)
    device = env.unwrapped.device
    num_envs = env.num_envs
    dt = env.unwrapped.step_dt

    policy, policy_source = load_policy(env, device)

    obs, _ = env.reset()
    obs, _ = env.get_observations()
    obs_s = strip(obs)
    obs_dim = obs_s.shape[-1]
    action_dim = env.unwrapped.action_manager.total_action_dim

    if is_velocity:
        dim_names = velocity_dim_names()
    else:
        dim_names = mimic_dim_names()
    if len(dim_names) != obs_dim:
        print(f"[WARN] dim_names ({len(dim_names)}) != obs_dim ({obs_dim}); using generic names")
        dim_names = [f"dim_{i}" for i in range(obs_dim)]

    os.makedirs(args_cli.out, exist_ok=True)
    print(f"[INFO] task={args_cli.task} envs={num_envs} obs_dim={obs_dim} "
          f"action_dim={action_dim} dt={dt}")

    if args_cli.test_episode_len is None:
        args_cli.test_episode_len = 1000 if is_velocity else 1500

    # ------------------------------------------------------------------
    # TRAIN split: nominal rollout, chopped into episodes at resets
    # ------------------------------------------------------------------
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    def f16(x: torch.Tensor) -> np.ndarray:
        return np.clip(x.detach().cpu().numpy(), -6e4, 6e4).astype(np.float16)

    if args_cli.skip_train:
        print("[TRAIN] skipped (--skip_train)")
    else:
        T = args_cli.train_steps
        obs_buf = np.zeros((num_envs, T, obs_dim), dtype=np.float16)
        act_buf = np.zeros((num_envs, T, action_dim), dtype=np.float16)
        done_steps = [[] for _ in range(num_envs)]

        with torch.no_grad():
            for t in range(T):
                actions = policy(obs)
                obs_buf[:, t] = f16(strip(obs))
                act_buf[:, t] = f16(actions)
                obs, _, dones, _ = env.step(actions)
                for i in torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist():
                    done_steps[i].append(t)
                if (t + 1) % 400 == 0:
                    print(f"[TRAIN] step {t + 1}/{T}")

        train_seqs, train_acts = [], []
        for i in range(num_envs):
            bounds = [-1] + done_steps[i] + [T - 1]
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b - a >= args_cli.min_seq_len:
                    train_seqs.append(obs_buf[i, a + 1: b + 1].copy())
                    train_acts.append(act_buf[i, a + 1: b + 1].copy())
        del obs_buf, act_buf
        total = sum(len(s) for s in train_seqs)
        print(f"[TRAIN] {len(train_seqs)} nominal episodes, {total} steps "
              f"({total * dt / 60:.1f} min of sim data)")
        save_ragged(os.path.join(args_cli.out, "train.npz"), train_seqs, train_acts, dim_names)
        del train_seqs, train_acts

    # Warmup episode: the first post-boot reset yields mass instant
    # terminations on mimic tasks — burn one short episode so real
    # collection starts from a healthy reset.
    if args_cli.skip_train:
        print("[WARMUP] running throwaway episode")
        obs, _ = env.reset()
        obs, _ = env.get_observations()
        with torch.no_grad():
            for _ in range(min(200, args_cli.test_episode_len)):
                obs, _, _, _ = env.step(policy(obs))

    # ------------------------------------------------------------------
    # CALIBRATION split: nominal batched episodes at evaluation conditions
    # ("a brief nominal calibration episode precedes evaluation")
    # ------------------------------------------------------------------
    E = args_cli.test_episode_len
    cal_seqs, cal_acts = [], []
    for ep in range(args_cli.cal_episodes):  # 0 = keep existing calibration.npz
        print(f"[CAL] nominal calibration episode {ep + 1}/{args_cli.cal_episodes}")
        obs, _ = env.reset()
        obs, _ = env.get_observations()
        ep_obs = np.zeros((num_envs, E, obs_dim), dtype=np.float16)
        ep_act = np.zeros((num_envs, E, action_dim), dtype=np.float16)
        length = np.full(num_envs, E, dtype=np.int64)
        with torch.no_grad():
            for step in range(E):
                actions = policy(obs)
                ep_obs[:, step] = f16(strip(obs))
                ep_act[:, step] = f16(actions)
                obs, _, dones, _ = env.step(actions)
                for i in torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist():
                    if length[i] == E:
                        length[i] = step + 1
        for i in range(num_envs):
            if length[i] >= args_cli.min_seq_len:
                cal_seqs.append(ep_obs[i, : length[i]].copy())
                cal_acts.append(ep_act[i, : length[i]].copy())
    if args_cli.cal_episodes > 0:
        save_ragged(os.path.join(args_cli.out, "calibration.npz"), cal_seqs, cal_acts, dim_names)
    del cal_seqs, cal_acts

    if args_cli.skip_test:
        print("[DONE] calibration-only run complete")
        env.close()
        return

    # ------------------------------------------------------------------
    # TEST split: one batched episode per OOD category, half control envs
    # ------------------------------------------------------------------
    injector = OODInjector(num_envs, obs_dim, action_dim, dt=dt, device=device, env=env)
    categories = [c for c in OODCategory if c != OODCategory.NONE]
    if args_cli.categories:
        wanted = set(args_cli.categories.split(","))
        categories = [c for c in categories if CATEGORY_NAMES[c] in wanted]
        print(f"[TEST] restricted to categories: {[CATEGORY_NAMES[c] for c in categories]}")
    E = args_cli.test_episode_len
    onset = args_cli.ood_start_step

    import time as _time

    test_seqs, test_acts, labels, onsets, faults = [], [], [], [], []
    for ci, category in enumerate(categories):
        name = CATEGORY_NAMES[category]
        _t0 = _time.time()
        print(f"[TEST] category {ci + 1}/{len(categories)}: {name}", flush=True)
        injector.setup_episode(category, seed=args_cli.seed + ci * 1000)

        obs, _ = env.reset()
        obs, _ = env.get_observations()
        obs_s = strip(obs)
        if is_velocity:
            from collections import deque

            queue = deque(maxlen=HISTORY_LEN)
            for _ in range(HISTORY_LEN):
                queue.append(obs_s.clone())

        ep_obs = np.zeros((num_envs, E, obs_dim), dtype=np.float16)
        ep_act = np.zeros((num_envs, E, action_dim), dtype=np.float16)
        length = np.full(num_envs, E, dtype=np.int64)

        with torch.no_grad():
            for step in range(E):
                if step == onset:
                    injector.activate()
                obs_pert = injector.inject_obs(strip(obs).clone())
                if is_velocity:
                    queue.append(obs_pert)
                    obs_policy = format_feature_history(queue, TERM_DIMS_VELOCITY)
                else:
                    obs_policy = obs_pert
                actions = policy(obs_policy)
                actions_pert = injector.inject_action(actions)
                ep_obs[:, step] = np.clip(obs_pert.cpu().numpy(), -6e4, 6e4).astype(np.float16)
                ep_act[:, step] = np.clip(actions_pert.cpu().numpy(), -6e4, 6e4).astype(np.float16)
                obs, _, dones, _ = env.step(actions_pert)
                if dones.any():
                    for i in torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist():
                        if length[i] == E:
                            length[i] = step + 1

        injector._reset_physics()
        ood_mask = injector.ood_mask.cpu().numpy()
        kept_ood = kept_nom = 0
        for i in range(num_envs):
            L = int(length[i])
            is_ood = bool(ood_mask[i])
            # OOD sequences need only brief post-onset evidence: anomalies severe
            # enough to fell the robot immediately must not be filtered out.
            min_keep = (onset + 10) if is_ood else args_cli.min_seq_len
            if L < min_keep:
                continue
            test_seqs.append(ep_obs[i, :L].copy())
            test_acts.append(ep_act[i, :L].copy())
            labels.append(int(is_ood))
            onsets.append(onset if is_ood else -1)
            faults.append(name if is_ood else "none")
            kept_ood += int(is_ood)
            kept_nom += int(not is_ood)
        print(f"[TEST]   kept {kept_ood} OOD + {kept_nom} control sequences "
              f"({_time.time() - _t0:.0f}s)", flush=True)

    test_path = os.path.join(args_cli.out, "test.npz")
    if args_cli.merge_test and os.path.exists(test_path):
        replaced = {CATEGORY_NAMES[c] for c in categories}
        old = np.load(test_path)
        old_seqs = sorted(k for k in old.files if k.startswith("seq_"))
        merged_seqs, merged_acts = [], []
        merged_labels, merged_onsets, merged_faults = [], [], []
        for j, key in enumerate(old_seqs):
            if str(old["fault"][j]) in replaced:
                continue  # replaced by this run
            merged_seqs.append(old[key])
            merged_acts.append(old[key.replace("seq_", "act_")])
            merged_labels.append(int(old["labels"][j]))
            merged_onsets.append(int(old["onset"][j]))
            merged_faults.append(str(old["fault"][j]))
        added = 0
        for seq, act, lab, ons, flt in zip(test_seqs, test_acts, labels, onsets, faults):
            if not lab:
                continue  # don't double-count nominal controls
            merged_seqs.append(seq)
            merged_acts.append(act)
            merged_labels.append(lab)
            merged_onsets.append(ons)
            merged_faults.append(flt)
            added += 1
        print(f"[MERGE] kept {len(merged_seqs) - added} existing + {added} new OOD sequences")
        test_seqs, test_acts = merged_seqs, merged_acts
        labels, onsets, faults = merged_labels, merged_onsets, merged_faults

    extras = {
        "labels": np.array(labels),
        "onset": np.array(onsets),
        "fault": np.array(faults),
    }
    save_ragged(test_path, test_seqs, test_acts, dim_names, extras)

    meta = {
        "task": args_cli.task,
        "policy": os.path.basename(str(policy_source)),
        "num_envs": num_envs,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "dt": dt,
        "train_steps_per_env": args_cli.train_steps,
        "test_episode_len": E,
        "cal_episodes": args_cli.cal_episodes,
        "ood_start_step": onset,
        "categories": [CATEGORY_NAMES[c] for c in categories],
        "test_counts": {"ood": int(sum(labels)), "nominal": int(len(labels) - sum(labels))},
        "dtype": "float16",
        "seed": args_cli.seed,
    }
    with open(os.path.join(args_cli.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] dataset written to {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
