#!/usr/bin/env bash
#
# SLURM Submission Script for MAE Training on All Trained Policies
#

# --- Configuration ---
CONDA_ENV_PATH="/scratch3/mun127/conda_envs/env_isaaclab5"
CONDA_BASE_DIR="/apps/miniconda3/23.5.2"
SCRIPT_PATH="scripts/rsl_rl/collect_and_train_fdm.py"
BASE_DIR="/scratch3/mun127/isaaclab_stack/unitree_rl_lab"
LOGS_DIR="$BASE_DIR/logs/rsl_rl"
OUTPUT_BASE_DIR="$BASE_DIR/fdm_rapt_models"

# FDM training parameters
COLLECTION_TIME=60
NUM_EPOCHS=100
NUM_ENVS=4096

# Delay between job submissions (seconds)
SUBMISSION_DELAY=3

# --- Task Name Mapping ---
declare -A TASK_MAP
TASK_MAP["throwing"]="Unitree-G1-29dof-Throwing"
TASK_MAP["unitree_g1_29dof_velocity"]="Unitree-G1-29dof-Velocity"
TASK_MAP["unitree_g1_29dof_mimic_gangnanm_style"]="Unitree-G1-29dof-Mimic-Gangnanm-Style"
TASK_MAP["unitree_g1_29dof_mimic_dance_102"]="Unitree-G1-29dof-Mimic-Dance-102"

# --- End Configuration ---

job_count=0

submit_fdm_job() {
    local task_name="$1"
    local run_name="$2"
    local task_dir_name="$3"
    
    local job_file_name="job_fdm_${task_dir_name}_${run_name}.sh"
    local job_name="fdm_${task_dir_name}_${run_name}"
    local output_dir="${OUTPUT_BASE_DIR}/${task_dir_name}/${run_name}"

    # --- LOGIC CHANGE START ---
    # Determine mask ratio based on specific task directory name
    
    # --- LOGIC CHANGE END ---
    
    job_name="${job_name:0:60}"

    cat << EOF > "$job_file_name"
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --account=OD-235390
#SBATCH --output=slurm_logs/fdm/${job_name}_%j.out
#SBATCH --error=slurm_logs/fdm/${job_name}_%j.err

# 1. Source the Conda initialization script
source $CONDA_BASE_DIR/etc/profile.d/conda.sh

# 2. Activate the target environment
conda activate "$CONDA_ENV_PATH"

# 3. Disable HDF5 file locking
export HDF5_USE_FILE_LOCKING=FALSE

# 4. Navigate to the base directory
cd "$BASE_DIR"

# 5. Print which run we're loading (for debugging)
echo "============================================"
echo "Loading policy from run: $run_name"
echo "Task: $task_name"
echo "Output dir: $output_dir"
echo "============================================"

# 6. Execute with explicit experiment name to avoid confusion
python -u "$SCRIPT_PATH" \\
    --task "$task_name" \\
    --collection_time $COLLECTION_TIME \\
    --num_epochs $NUM_EPOCHS \\
    --num_envs $NUM_ENVS \\
    --headless \\
    --use_critic_multi \\
    --load_run "$run_name" \\
    --experiment_name "$task_dir_name" \\
    --output_dir "$output_dir" \\

# 7. Deactivate
conda deactivate
EOF

    mkdir -p slurm_logs/fdm

    echo "Submitting FDM job:"
    echo "  Task: $task_name"
    echo "  Run: $run_name"
    echo "  Output: $output_dir"
    sbatch "$job_file_name"

    rm "$job_file_name"
    
    ((job_count++))
    
    # Stagger submissions to avoid race conditions
    sleep $SUBMISSION_DELAY
}

# --- Main Logic ---

echo "========================================"
echo "FDM Training Job Submission"
echo "========================================"
echo "Scanning for trained policies in: $LOGS_DIR"
echo "FDM outputs will be saved to: $OUTPUT_BASE_DIR"
echo "Delay between submissions: ${SUBMISSION_DELAY}s"
echo ""

if [ ! -d "$LOGS_DIR" ]; then
    echo "ERROR: Logs directory not found: $LOGS_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_BASE_DIR"

for task_dir in "$LOGS_DIR"/*/; do
    task_dir_name=$(basename "$task_dir")
    
    [ -d "$task_dir" ] || continue
    
    full_task_name="${TASK_MAP[$task_dir_name]}"
    
    if [ -z "$full_task_name" ]; then
        echo "WARNING: No task mapping found for directory: $task_dir_name (skipping)"
        continue
    fi
    
    echo "Found task: $task_dir_name -> $full_task_name"
    
    for run_dir in "$task_dir"/*/; do
        run_name=$(basename "$run_dir")
        # skip if "_456" not fund in run_name
        #if [[ "$run_name" != *"_456"* ]]; then
        #    echo "  Skipping $run_name (does not match seed filter)"
        #    continue
        #fi

        [ -d "$run_dir" ] || continue
        
        if [ -f "$run_dir/model_*.pt" ] || [ -d "$run_dir/checkpoints" ] || ls "$run_dir"/*.pt 1> /dev/null 2>&1; then
            submit_fdm_job "$full_task_name" "$run_name" "$task_dir_name"
        else
            echo "  Skipping $run_name (no checkpoint found)"
        fi
    done
    
    echo ""
done

echo "========================================"
echo "Submitted $job_count FDM training jobs."
echo "========================================"