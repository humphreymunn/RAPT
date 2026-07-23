import torch
import torch.nn as nn
import json
import os
import argparse
import numpy as np

# ==================================================================================
# 1. MODEL DEFINITIONS
# ==================================================================================

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

class UniversalAutoencoder(nn.Module):
    # UPDATED: Now accepts separate input_dim and output_dim
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.args = args
        self.use_temporal = args.use_temporal
        self.embed_dim = args.embed_dim
        
        # --- Bottleneck Calculation ---
        if args.reconstruction_type == 'bottleneck':
            keep_ratio = 1.0 - args.mask_ratio
            self.latent_dim = max(4, int(args.embed_dim * keep_ratio))
        else:
            self.latent_dim = args.embed_dim

        # --- 1. Spatial Encoder (MLP) ---
        # Uses INPUT_DIM (e.g., 125 for FDM)
        encoder_layers = [nn.Linear(self.input_dim, self.embed_dim), nn.ReLU()]
        for _ in range(args.num_blocks):
            encoder_layers.append(ConfigurableBlock(self.embed_dim, self.embed_dim, args.use_residual, args.dropout))
        self.encoder_mlp = nn.Sequential(*encoder_layers)
        
        # --- 2. Temporal Core (GRU) ---
        if self.use_temporal:
            self.gru = nn.GRU(self.embed_dim, self.embed_dim, num_layers=1, batch_first=True)
            
        # --- 3. Bottleneck Projection ---
        if args.reconstruction_type == 'bottleneck':
            self.compress = nn.Sequential(
                nn.Linear(self.embed_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.ReLU()
            )
            self.decompress = nn.Sequential(
                nn.Linear(self.latent_dim, self.embed_dim),
                nn.ReLU()
            )
            
        # --- 4. Spatial Decoder (MLP) ---
        decoder_layers = []
        for _ in range(args.num_blocks):
            decoder_layers.append(ConfigurableBlock(self.embed_dim, self.embed_dim, args.use_residual, args.dropout))
        self.decoder_mlp = nn.Sequential(*decoder_layers)
        
        # --- 5. Output Head ---
        # Uses OUTPUT_DIM (e.g., 96 for Reconstruction)
        out_features = self.output_dim * 2 if args.use_probabilistic else self.output_dim
        self.head = nn.Linear(self.embed_dim, out_features)

    def forward(self, x, hidden=None, force_no_mask=False):
        # 1. Handle Dimensions
        is_sequence = x.dim() == 3
        if is_sequence:
            batch, seq, dim = x.shape
            x_flat = x.reshape(-1, dim)
        else:
            batch, dim = x.shape
            x_flat = x
            
        # 2. Input Masking (Ignored for ONNX export if force_no_mask is used)
        # For FDM, masking is usually disabled or applied differently, 
        # but for export we force full pass.
        x_in = x_flat

        # 3. Spatial Encode
        z = self.encoder_mlp(x_in)
        
        # 4. Temporal Core
        if self.use_temporal:
            if not is_sequence:
                z = z.unsqueeze(1) 
            else:
                z = z.view(batch, seq, -1)
            
            z, hidden = self.gru(z, hidden)
            z = z.reshape(-1, z.shape[-1])
            
        # 5. Latent Bottleneck
        if self.args.reconstruction_type == 'bottleneck':
            z = self.compress(z)
            z = self.decompress(z)
            
        # 6. Spatial Decode
        z = self.decoder_mlp(z)
        out = self.head(z)
        
        # 7. Restore Shape
        if is_sequence:
            out = out.view(batch, seq, -1)
            
        return out, None, hidden

# ==================================================================================
# 2. EXPORT HELPER CLASSES
# ==================================================================================

class ArgsNamespace:
    """Helper to convert dictionary to object for model init."""
    def __init__(self, **entries):
        self.__dict__.update(entries)

class OnnxWrapper(nn.Module):
    """
    Wraps the model to simplify inputs/outputs for C++ inference.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, hidden):
        # Pass 'hidden' into the model
        out, _, new_hidden = self.model(x, hidden=hidden, force_no_mask=True)
        
        # Return both the reconstruction and the new hidden state
        return out, new_hidden

# ==================================================================================
# 3. MAIN EXPORT LOGIC
# ==================================================================================

def export_model(model_dir):
    print(f"[INFO] Processing directory: {model_dir}")
    
    # 1. Load Configuration
    config_path = os.path.join(model_dir, "experiment_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    args = ArgsNamespace(**config_dict)
    print("[INFO] Configuration loaded.")

    # 2. Load Stats to determine Output Dimension (Observation Dim)
    stats_path = os.path.join(model_dir, "obs_stats.json")
    if not os.path.exists(stats_path):
        import h5py
        h5_path = os.path.join(model_dir, "obs_stats.h5")
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                 obs_dim = f['mean'].shape[0]
        else:
            raise FileNotFoundError("Could not find obs_stats.json")
    else:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            obs_dim = len(stats['mean'])
            
    print(f"[INFO] Detected observation dimension (output): {obs_dim}")

    # 3. Load Weights FIRST to detect Input Dimension (Auto-detect FDM vs Standard)
    weights_path = os.path.join(model_dir, "final_model.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "best_model.pth")
        
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
        
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # --- AUTO DETECTION LOGIC ---
    # Check the shape of the first layer weight: [embed_dim, input_dim]
    first_layer_weight = state_dict['encoder_mlp.0.weight']
    detected_input_dim = first_layer_weight.shape[1]
    
    print(f"[INFO] Weights loaded. First layer input shape: {detected_input_dim}")
    
    if detected_input_dim == obs_dim:
        print("[INFO] Mode: Standard Autoencoder (Reconstruction)")
    elif detected_input_dim > obs_dim:
        diff = detected_input_dim - obs_dim
        print(f"[INFO] Mode: Forward Dynamics Model (FDM) | Action Dim detected: {diff}")
    else:
        print(f"[WARNING] Input dim ({detected_input_dim}) < Obs dim ({obs_dim}). This is unusual.")

    # 4. Initialize Model with corrected dimensions
    model = UniversalAutoencoder(input_dim=detected_input_dim, output_dim=obs_dim, args=args)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Prepare for ONNX
    onnx_model = OnnxWrapper(model)
    
    dummy_input = torch.randn(1, detected_input_dim)
    
    # --- NEW: Create dummy hidden state ---
    # Shape: [Num_Layers, Batch, Hidden_Dim]
    # Assuming Num_Layers=1 (from your UniversalAutoencoder init) and Batch=1
    dummy_hidden = torch.zeros(1, 1, args.embed_dim) 

    # Output path
    onnx_path = os.path.join(model_dir, "rapt.onnx") 
    
    # 6. Export
    print("[INFO] Exporting to ONNX...")
    torch.onnx.export(
        onnx_model,
        (dummy_input, dummy_hidden),    # Pass TUPLE of inputs
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input', 'hidden_in'],           # Define Names
        output_names=['reconstruction', 'hidden_out'], # Define Names
        dynamic_axes={
            'input': {0: 'batch_size'},
            'hidden_in': {1: 'batch_size'},   # Note: PyTorch GRU hidden is [Layers, Batch, Dim]
            'reconstruction': {0: 'batch_size'},
            'hidden_out': {1: 'batch_size'}
        }
    )
    
    print(f"[SUCCESS] Model saved to: {onnx_path}")
    print(f"[INFO] Input Name: 'input', Shape: [Batch, {detected_input_dim}]")
    print(f"[INFO] Output Name: 'reconstruction', Shape: [Batch, {obs_dim * 2 if args.use_probabilistic else obs_dim}]")

if __name__ == "__main__":
    # Point this to your FDM model directory
    TARGET_DIR = "fdm_models_deploy/DYN_Unitree-G1-29dof-Velocity_20260325_125357"
    
    if os.path.exists(TARGET_DIR):
        export_model(TARGET_DIR)
    else:
        print(f"[ERROR] Directory not found: {TARGET_DIR}")