#!/usr/bin/env bash
## SLURM Submission Script for OOD Detection on Taguchi L12 Ablations
## Logic:
## 1. Scans mae_ablations_L12/
## 2. For each trained model, reads experiment_config.json (FLAT structure)
## 3. Extracts arguments directly.
## 4. Submits 3 inference jobs (Mean, Max, Both).

## --- Configuration ---
CONDA_ENV_PATH="/scratch3/mun127/conda_envs/env_isaaclab5"
CONDA_BASE_DIR="/apps/miniconda3/23.5.2"
SCRIPT_PATH="scripts/rsl_rl/play_with_fdm.py"
BASE_DIR="/scratch3/mun127/isaaclab_stack/unitree_rl_lab"
ABLATION_MODELS_DIR="$BASE_DIR/fdm_ablations_L12_BOTTLENECK"
OUTPUT_BASE_DIR="$BASE_DIR/ood_ablation_results_fdmrapt_BOTTLENECK"

# Common Experiment Params
NUM_ENVS=4096
CATEGORIES="all"
SUBMISSION_DELAY=2

# --- Episode Length Mapping ---
declare -A EPISODE_LENGTH_MAP
EPISODE_LENGTH_MAP["Unitree-G1-29dof-Throwing"]=100
EPISODE_LENGTH_MAP["Unitree-G1-29dof-Velocity"]=1000
EPISODE_LENGTH_MAP["Unitree-G1-29dof-Mimic-Gangnanm-Style"]=1500
EPISODE_LENGTH_MAP["Unitree-G1-29dof-Mimic-Dance-102"]=1500

# --- Functions ---

# Extract value from JSON root
get_json_value() {
    local json_file="$1"
    local key="$2"
    # Returns empty string if key not found
    python3 -c "import json; print(json.load(open('$json_file')).get('$key', ''))" 2>/dev/null
}

# Extract boolean flag from JSON root
get_json_bool() {
    local json_file="$1"
    local key="$2"
    local flag_name="$3"
    
    val=$(python3 -c "import json; print(str(json.load(open('$json_file')).get('$key', False)).lower())" 2>/dev/null)
    
    if [ "$val" == "true" ]; then
        echo "$flag_name"
    else
        echo ""
    fi
}

get_episode_length() {
    local task_name="$1"
    local val="${EPISODE_LENGTH_MAP[$task_name]}"
    if [ -z "$val" ]; then echo "1000"; else echo "$val"; fi
}

submit_job() {
    local metadata_file="$1"
    local checkpoint_file="$2"
    local task_dir_name="$3"
    local run_id="$4"
    local detect_type="$5"
    
    # 1. Parse Config from experiment_config.json (Flat structure)
    local task_name=$(get_json_value "$metadata_file" "task")
    local recon_type=$(get_json_value "$metadata_file" "reconstruction_type")
    local mask_ratio=$(get_json_value "$metadata_file" "mask_ratio")
    
    # Extract Run Name (Policy)
    # Priority 1: Check if 'load_run' exists in config
    local load_run=$(get_json_value "$metadata_file" "load_run")
    
    # Priority 2: If 'load_run' is empty/None, try to extract from 'checkpoint' path
    if [ -z "$load_run" ] || [ "$load_run" == "None" ]; then
        local policy_ckpt_path=$(get_json_value "$metadata_file" "checkpoint")
        if [ -n "$policy_ckpt_path" ] && [ "$policy_ckpt_path" != "None" ]; then
            # Extract parent directory name (Run Name)
            load_run=$(basename "$(dirname "$policy_ckpt_path")")
        else
            # Fallback for old configs that might not have captured args explicitly
            echo "  [WARN] Could not determine load_run from config. Skipping."
            return
        fi
    fi

    # Boolean Flags
    local flag_resid=$(get_json_bool "$metadata_file" "use_residual" "--use_residual")
    local flag_prob=$(get_json_bool "$metadata_file" "use_probabilistic" "--use_probabilistic")
    local flag_temp=$(get_json_bool "$metadata_file" "use_temporal" "--use_temporal")
    local flag_train=$(get_json_bool "$metadata_file" "train_dynamics" "--train_dynamics")
    
    local ep_len=$(get_episode_length "$task_name")
    
    # 2. Setup Paths
    local job_name="inf_${task_dir_name:0:10}_${run_id}_${detect_type}"
    local output_dir="${OUTPUT_BASE_DIR}/${task_dir_name}/${run_id}/${detect_type}"
    local job_file="job_${job_name}.sh"
    
    # Skip if done
    if [ -f "$output_dir/ood_results.json" ]; then
        echo "  [SKIP] Results exist for $job_name"
        return
    fi

    # 3. Create Script
    cat << EOF > "$job_file"
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --account=OD-235390
#SBATCH --output=slurm_logs/abl_inf/${job_name}_%j.out
#SBATCH --error=slurm_logs/abl_inf/${job_name}_%j.err

source $CONDA_BASE_DIR/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_PATH"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$BASE_DIR"

echo "=== Ablation Inference ==="
echo "Task: $task_name"
echo "Run ID: $run_id"
echo "Detect Type: $detect_type"
echo "Policy Run: $load_run"
echo "Config: $recon_type | Mask: $mask_ratio | $flag_resid $flag_prob $flag_temp $flag_train"

python -u "$SCRIPT_PATH" \\
    --task "$task_name" \\
    --load_run "$load_run" \\
    --mae_checkpoint "$checkpoint_file" \\
    --output_dir "$output_dir" \\
    --ood_detect_type "$detect_type" \\
    --reconstruction_type "$recon_type" \\
    --mask_ratio "$mask_ratio" \\
    $flag_resid \\
    $flag_prob \\
    $flag_temp \\
    --episode_length $ep_len \\
    --categories "$CATEGORIES" \\
    --num_envs $NUM_ENVS \\
    --headless \\
    --use_critic_multi

conda deactivate
EOF

    sbatch "$job_file" > /dev/null
    rm "$job_file"
    echo "  Submitted: $job_name"
    sleep $SUBMISSION_DELAY
}

# --- Main Execution ---
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p slurm_logs/abl_inf

echo "=================================================="
echo "Starting Ablation Inference Submission"
echo "=================================================="

# 1. Iterate Task Dirs
for task_dir in "$ABLATION_MODELS_DIR"/*/; do
    task_dir_name=$(basename "$task_dir")
    [ -d "$task_dir" ] || continue
    
    echo "Processing Task: $task_dir_name"
    
    # 2. Iterate Runs
    for run_dir in "$task_dir"/*/; do
        run_id=$(basename "$run_dir")
        [ -d "$run_dir" ] || continue
        
        # 3. Find the experiment folder
        exp_dir=$(find "$run_dir" -maxdepth 1 -mindepth 1 -type d | head -n 1)
        
        if [ -z "$exp_dir" ]; then
            echo "  [WARN] No experiment dir found in $run_dir"
            continue
        fi
        
        # CHANGED: Look for experiment_config.json
        metadata_file="$exp_dir/experiment_config.json"
        
        # Checkpoint logic
        if [ -f "$exp_dir/best_model.pth" ]; then
            checkpoint_file="$exp_dir/best_model.pth"
        elif [ -f "$exp_dir/final_model.pth" ]; then
            checkpoint_file="$exp_dir/final_model.pth"
            echo "  [INFO] best_model.pth not found, using final_model.pth for $run_id"
        else
            echo "  [WARN] Missing checkpoint in $run_id"
            continue
        fi

        if [ ! -f "$metadata_file" ]; then
            echo "  [WARN] Missing experiment_config.json in $run_id"
            continue
        fi
        
        # 4. Submit 3 Jobs (Mean, Max, Both)
        for dtype in "max"; do
            submit_job "$metadata_file" "$checkpoint_file" "$task_dir_name" "$run_id" "$dtype"
        done
        
    done
done

echo "Done."