"""OOD injection library — extracted verbatim from play_with_rapt.py (OODCategory,
CATEGORY_NAMES, OODInjector) so collection/eval scripts can share it."""
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

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
        self._pending_init_state = None
        self._pending_push = None
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
                
                # Defer the impulse to activate(): applying at setup would be
                # wiped by the env.reset() that follows setup_episode().
                self._pending_push = (push_env_ids, final_velocity)

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

            # Defer the sim writes to activate(): writing here would be wiped
            # by the env.reset() that follows setup_episode() in the eval loop.
            self._pending_init_state = (joint_pos, joint_vel, robot_root_state[:, :7], env_ids)

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

        # Deferred state perturbations (computed at setup, applied at onset so
        # they land after env.reset()).
        if getattr(self, "_pending_init_state", None) is not None:
            robot = self._get_robot()
            joint_pos, joint_vel, root_pose, env_ids = self._pending_init_state
            robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids=env_ids)
            self._pending_init_state = None
        if getattr(self, "_pending_push", None) is not None:
            robot = self._get_robot()
            push_env_ids, final_velocity = self._pending_push
            root_states = robot.data.root_state_w[push_env_ids].clone()
            root_states[:, 7] = final_velocity[:, 0]  # world-frame x lin vel
            root_states[:, 8] = final_velocity[:, 1]  # world-frame y lin vel
            robot.write_root_velocity_to_sim(root_states[:, 7:], env_ids=push_env_ids.to(root_states.device))
            self._pending_push = None
    
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

