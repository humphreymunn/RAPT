import os
import glob
import json
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = "/scratch3/mun127/isaaclab_stack/unitree_rl_lab"
RESULTS_DIR = os.path.join(BASE_DIR, "ood_ablation_results_fdmrapt")
MODELS_DIR = os.path.join(BASE_DIR, "fdm_ablations_L12")
OUTPUT_FILE = os.path.join(BASE_DIR, "collated_taguchi_results.csv")

# Hyperparameters to extract from config files
PARAMS_TO_EXTRACT = [
    "reconstruction_type", 
    "mask_ratio", 
    "use_residual", 
    "use_probabilistic", 
    "train_dynamics",
    "use_temporal", 
    "embed_dim", 
    "num_blocks", 
    "dropout",
    "lr"
]

def load_config(task_name, run_id):
    """
    Attempts to find and load configuration from experiment_config.json
    or collection_metadata.json for a specific run.
    """
    # Pattern: mae_ablations_L12/task_name/run_id/*/experiment_config.json
    search_path = os.path.join(MODELS_DIR, task_name, run_id, "*")
    
    # 1. Try experiment_config.json (Primary)
    config_matches = glob.glob(os.path.join(search_path, "experiment_config.json"))
    if config_matches:
        with open(config_matches[0], 'r') as f:
            return json.load(f)
            
    # 2. Try collection_metadata.json (Fallback)
    metadata_matches = glob.glob(os.path.join(search_path, "collection_metadata.json"))
    if metadata_matches:
        with open(metadata_matches[0], 'r') as f:
            data = json.load(f)
            return data.get("experiment_arguments", {})
            
    return None

def main():
    print(f"Scanning results in: {RESULTS_DIR}")
    print(f"Looking up models in: {MODELS_DIR}")
    
    records = []
    
    # Walk through: results/task/run_id/detect_type/ood_summary.csv
    # 1. Task Level
    task_dirs = glob.glob(os.path.join(RESULTS_DIR, "*"))
    
    for task_path in task_dirs:
        if not os.path.isdir(task_path): continue
        task_name = os.path.basename(task_path)
        
        # 2. Run Level (run_1, run_2, etc.)
        run_dirs = glob.glob(os.path.join(task_path, "*"))
        
        for run_path in run_dirs:
            if not os.path.isdir(run_path): continue
            run_id = os.path.basename(run_path)
            
            # --- Retrieve Hyperparameters for this Run ---
            config = load_config(task_name, run_id)
            
            if config is None:
                print(f"[WARN] No config found for {task_name}/{run_id}. Skipping parameters.")
                extracted_params = {k: None for k in PARAMS_TO_EXTRACT}
            else:
                extracted_params = {k: config.get(k, None) for k in PARAMS_TO_EXTRACT}

            # 3. Detect Type Level (max, mean, both)
            type_dirs = glob.glob(os.path.join(run_path, "*"))
            
            for type_path in type_dirs:
                if not os.path.isdir(type_path): continue
                detect_type = os.path.basename(type_path)
                
                csv_file = os.path.join(type_path, "ood_summary.csv")
                if not os.path.exists(csv_file):
                    continue
                
                try:
                    # Load the summary CSV
                    df = pd.read_csv(csv_file)
                    
                    # Process each row (Category)
                    for _, row in df.iterrows():
                        # Build the record
                        record = {
                            # Identifiers
                            "Task": task_name,
                            "Run_ID": run_id,
                            "Detect_Type": detect_type,
                            
                            # Standard Metrics
                            "Category": row.get("category", "unknown"),
                            "PADD": row.get("PADD", np.nan),
                            "FPR": row.get("FPR", np.nan),
                            "TPR": row.get("TPR", np.nan),
                            "F1": row.get("F1", np.nan),
                            "Accuracy": row.get("accuracy", np.nan),
                            
                            # --- NEW METRICS FROM YOUR CSV ---
                            "AUROC_Hybrid": row.get("AUROC_Hybrid", np.nan),
                            "AUROC_Model": row.get("AUROC_Model", np.nan),
                            "TPR@FPR0.5%_Hybrid": row.get("TPR@FPR0.5%_Hybrid", np.nan),
                            "TPR@FPR0.5%_Model": row.get("TPR@FPR0.5%_Model", np.nan),
                            "Avg_Latency_ms": row.get("Avg_Latency_ms", np.nan),
                            
                            # Counts
                            "Count_Range_Only": row.get("range_only", 0),
                            "Count_MAE_Only": row.get("mae_only", 0),
                            "Count_Both": row.get("both", 0)
                        }
                        
                        # Add Hyperparameters
                        record.update(extracted_params)
                        
                        records.append(record)
                        
                except Exception as e:
                    print(f"[ERR] Failed processing {csv_file}: {e}")

    # --- Save to CSV ---
    if records:
        final_df = pd.DataFrame(records)
        
        # Sort for easier reading
        final_df.sort_values(by=["Task", "Run_ID", "Detect_Type", "Category"], inplace=True)
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n[SUCCESS] Collated {len(final_df)} rows.")
        print(f"Results saved to: {OUTPUT_FILE}")
        
        # Print a quick preview
        print("\n--- Quick Stats ---")
        print(f"Unique Runs processed: {final_df['Run_ID'].nunique()}")
        print("Sample row:")
        print(final_df.iloc[0])
    else:
        print("\n[WARN] No results found to collate.")

if __name__ == "__main__":
    main()