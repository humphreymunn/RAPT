import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
BASE_DIR = "/scratch3/mun127/isaaclab_stack/unitree_rl_lab"
INPUT_FILE = os.path.join(BASE_DIR, "collated_taguchi_results.csv")

# Features to analyze (The "Factors" of your experiment)
FEATURES = [
    "reconstruction_type", 
    "mask_ratio", 
    "use_residual", 
    "use_probabilistic", 
    "use_temporal",
    "train_dynamics",
    "Detect_Type" # Also checking if inference method matters
]

# --- MODIFIED TARGET ---
TARGET = "TPR@FPR0.5%_Hybrid" 

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    print(f"Reading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # 1. Preprocessing
    # Filter out Control/None categories (they have 0 TPR by definition usually)
    df_clean = df[~df['Category'].str.lower().isin(['none', 'control'])].copy()
    
    # Drop rows with missing target or features
    df_clean.dropna(subset=[TARGET] + FEATURES, inplace=True)
    
    print(f"Analyzing {len(df_clean)} samples across {df_clean['Task'].nunique()} tasks...")
    print(f"Target Metric: {TARGET}")

    # 2. Prepare Data for ML
    X = df_clean[FEATURES].copy()
    y = df_clean[TARGET]

    # Encode categorical/boolean features to numbers
    encoders = {}
    for col in X.columns:
        le = LabelEncoder()
        # Convert to string to handle mixed types (e.g. True/False vs 1/0)
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # 3. Train Random Forest
    # We use a Regressor to predict the hybrid TPR based on the config
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 4. Extract Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n" + "="*50)
    print("FEATURE IMPORTANCE RANKING (Impact on Hybrid TPR)")
    print("="*50)
    print(f"{'Rank':<5} {'Feature':<25} {'Importance':<10}")
    print("-" * 45)
    
    sorted_features = []
    for f in range(X.shape[1]):
        feature_name = FEATURES[indices[f]]
        score = importances[indices[f]]
        sorted_features.append(feature_name)
        print(f"{f+1:<5} {feature_name:<25} {score:.4f}")

    # 5. Marginal Means Analysis (The "Direction" of the effect)
    print("\n" + "="*50)
    print(f"MARGINAL MEANS (Which setting maximizes {TARGET}?)")
    print("="*50)
    
    for feature in sorted_features:
        print(f"\n[ {feature} ]")
        # Group by the original values (before encoding)
        stats = df_clean.groupby(feature)[TARGET].agg(['mean', 'std', 'count'])
        
        # Calculate delta from global mean
        global_mean = df_clean[TARGET].mean()
        stats['delta'] = stats['mean'] - global_mean
        
        # Sort by best performance (High TPR is better)
        stats.sort_values(by='mean', ascending=False, inplace=True)
        
        for setting, row in stats.iterrows():
            # Formatting for boolean vs string
            setting_str = str(setting)
            mean_val = row['mean']
            delta = row['delta']
            sign = "+" if delta >= 0 else ""
            
            print(f"  {setting_str:<15} -> {TARGET}: {mean_val:.4f} ({sign}{delta:.4f} vs avg)")

    # 6. Interaction Check (Bonus)
    # Check simple interaction: Temporal + Detect_Type
    print("\n" + "="*50)
    print(f"KEY INTERACTION CHECK: Temporal x Detection Type ({TARGET})")
    print("="*50)
    interaction = df_clean.groupby(['use_temporal', 'Detect_Type'])[TARGET].mean().reset_index()
    interaction.sort_values(by=TARGET, ascending=False, inplace=True)
    print(interaction)

    # 7. Task-Specific Dynamics Analysis (NEW SECTION)
    print("\n" + "="*60)
    print(f"TASK-SPECIFIC ANALYSIS: Dynamics vs. Reconstruction ({TARGET})")
    print("="*60)
    
    # Group by Task and train_dynamics to get the mean performance
    task_dynamics_stats = df_clean.groupby(['Task', 'train_dynamics'])[TARGET].mean().unstack()
    
    # Check if columns are booleans or strings and handle naming
    # Assuming 'train_dynamics' values are roughly True/False or 1/0
    cols = task_dynamics_stats.columns
    
    # Try to identify which column represents "Dynamics=True"
    # Usually it sorts as False, True or 0, 1. So the last column is likely True.
    dyn_col = cols[-1] 
    recon_col = cols[0]
    
    print(f"{'Task Name':<30} | {'Dynamics':<10} | {'Reconstruct':<10} | {'Delta':<10} | {'Winner'}")
    print("-" * 85)

    for task_name, row in task_dynamics_stats.iterrows():
        dyn_score = row[dyn_col]
        recon_score = row[recon_col]
        
        # Calculate improvement
        delta = dyn_score - recon_score
        winner = "DYNAMICS" if delta > 0 else "RECONST."
        
        print(f"{task_name:<30} | {dyn_score:.4f}     | {recon_score:.4f}      | {delta:+.4f}     | {winner}")

    print("-" * 85)


if __name__ == "__main__":
    main()