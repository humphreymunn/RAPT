import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
BASE_DIR = "/scratch3/mun127/isaaclab_stack/unitree_rl_lab"
INPUT_FILE = os.path.join(BASE_DIR, "collated_taguchi_results.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "ranked_taguchi_models.csv")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    print(f"Reading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # 1. Clean Data for Analysis
    # We want Average TPR over OOD categories only.
    # Exclude 'control'/'none' which are for FPR calculation.
    ood_df = df[~df['Category'].str.lower().isin(['none', 'control'])].copy()

    # 2. Define Grouping Keys (Task + Unique Model Configuration + Inference Mode)
    group_cols = ['Task', 'Run_ID', 'Detect_Type']
    
    # 3. Define Hyperparameter Columns to preserve
    # We list potential params; the script will only use what exists in your CSV.
    potential_params = [
        "reconstruction_type", "mask_ratio", "use_residual", 
        "use_probabilistic", "use_temporal", "train_dynamics", "embed_dim", 
        "num_blocks", "dropout", "lr", "train_dynamics"
    ]
    available_params = [c for c in potential_params if c in df.columns]

    # 4. Aggregation Dictionary
    # Calculate Mean of metrics across all OOD categories
    # Note: We prioritize the Hybrid metrics now.
    metrics_to_agg = {
        'TPR@FPR0.5%_Hybrid': 'mean',  # Primary Safety Metric
        'AUROC_Hybrid': 'mean',        # Primary Performance Metric
        'TPR': 'mean',                 # Secondary (Legacy)
        'FPR': 'mean',
        'PADD': 'mean',
        'F1': 'mean',
        'Accuracy': 'mean',
        'Avg_Latency_ms': 'mean'       # Efficiency
    }
    
    # Only aggregate metrics that actually exist in the CSV
    agg_dict = {k: v for k, v in metrics_to_agg.items() if k in df.columns}

    # Add params to aggregation (taking the first value found per group)
    for p in available_params:
        agg_dict[p] = 'first'

    # 5. Perform Aggregation
    print("Calculating average metrics across OOD categories...")
    ranked_df = ood_df.groupby(group_cols).agg(agg_dict).reset_index()

    # 6. Sort and Rank
    # Primary Sort: Task (A-Z)
    # Secondary Sort: TPR@FPR0.5%_Hybrid (Descending) -> The "Hero Metric"
    # Tertiary Sort: AUROC_Hybrid (Descending)
    
    sort_keys = ['Task']
    ascending_vals = [True]
    
    if 'TPR@FPR0.5%_Hybrid' in ranked_df.columns:
        sort_keys.append('TPR@FPR0.5%_Hybrid')
        ascending_vals.append(False)
    
    if 'AUROC_Hybrid' in ranked_df.columns:
        sort_keys.append('AUROC_Hybrid')
        ascending_vals.append(False)

    ranked_df.sort_values(by=sort_keys, ascending=ascending_vals, inplace=True)

    # 7. Reorder columns for readability
    # Put the most important metrics right after the identifiers
    metric_cols_display = [c for c in ['TPR@FPR0.5%_Hybrid', 'AUROC_Hybrid', 'Avg_Latency_ms', 'TPR'] if c in ranked_df.columns]
    
    final_cols = ['Task', 'Run_ID', 'Detect_Type'] + metric_cols_display + available_params
    
    # Add any remaining metrics (like F1, PADD) at the end
    remaining_cols = [c for c in ranked_df.columns if c not in final_cols]
    final_cols = final_cols + remaining_cols
    
    ranked_df = ranked_df[final_cols]

    # 8. Formatting (Round floats)
    float_cols = ranked_df.select_dtypes(include=['float']).columns
    ranked_df[float_cols] = ranked_df[float_cols].round(4)

    # 9. Save
    ranked_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Ranking complete.")
    print(f"Output saved to: {OUTPUT_FILE}")
    
    # 10. Preview Best Models per Task
    print("\n=== TOP MODEL PER TASK (Based on TPR@FPR=0.5%) ===")
    for task in ranked_df['Task'].unique():
        task_df = ranked_df[ranked_df['Task'] == task]
        if task_df.empty: continue
        
        top_row = task_df.iloc[0]
        
        # Get score safely
        score = top_row.get('TPR@FPR0.5%_Hybrid', top_row.get('TPR', 0))
        
        print(f"\nTask: {task}")
        print(f"  Best Score: {score:.4f}")
        print(f"  Config:     {top_row['Detect_Type'].upper()} | {top_row['Run_ID']}")
        
        # Dynamic param printing
        params_str = " | ".join([f"{p}={top_row[p]}" for p in available_params])
        print(f"  Params:     {params_str}")

if __name__ == "__main__":
    main()