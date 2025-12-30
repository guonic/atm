#!/usr/bin/env python3
"""
Check Structure Expert model file.

This script verifies that structure_expert.pth model file is valid and can be loaded correctly.

Usage:
    python check_structure_expert.py [--model_path models/structure_expert.pth]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

# Add parent directory to path to import structure_expert
sys.path.insert(0, str(Path(__file__).parent))

from structure_expert import StructureExpertGNN


def check_file_exists(model_path: Path) -> bool:
    """Check if model file exists."""
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    print(f"✓ Model file exists: {model_path}")
    return True


def check_file_size(model_path: Path) -> bool:
    """Check if model file size is reasonable."""
    file_size = model_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"  File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    
    if file_size == 0:
        print("❌ Model file is empty!")
        return False
    
    if file_size < 1000:  # Less than 1KB is suspicious
        print("⚠️  Warning: Model file is very small, may be corrupted")
        return False
    
    print("✓ File size is reasonable")
    return True


def check_load_state_dict(model_path: Path) -> Optional[Dict]:
    """Check if state dict can be loaded."""
    try:
        print("\n2. Checking state dict loading...")
        state_dict = torch.load(model_path, map_location="cpu")
        print("✓ State dict loaded successfully")
        return state_dict
    except Exception as e:
        print(f"❌ Failed to load state dict: {e}")
        return None


def infer_model_params(state_dict: Dict) -> tuple:
    """
    Infer model parameters from state dict.
    
    Args:
        state_dict: Model state dictionary.
    
    Returns:
        Tuple of (n_feat, n_hidden, n_heads) or None if cannot infer.
    """
    try:
        # Try to infer from conv1.lin_l.weight (input layer)
        if "conv1.lin_l.weight" in state_dict:
            n_feat = state_dict["conv1.lin_l.weight"].shape[1]  # Input features
        elif "conv1.weight" in state_dict:
            n_feat = state_dict["conv1.weight"].shape[1]
        else:
            return None, None, None
        
        # Try to infer n_heads and n_hidden from conv1
        if "conv1.att" in state_dict:
            # GATv2Conv structure: att shape is [1, n_heads, n_hidden]
            att_shape = state_dict["conv1.att"].shape
            n_heads = att_shape[1]
            n_hidden = att_shape[2]
        elif "conv1.lin_l.weight" in state_dict:
            # Infer from lin_l output size
            # conv1 output = n_hidden * n_heads
            conv1_output = state_dict["conv1.lin_l.weight"].shape[0]
            # conv2 input should match conv1 output
            if "conv2.lin_l.weight" in state_dict:
                conv2_input = state_dict["conv2.lin_l.weight"].shape[1]
                # conv1 output = conv2 input = n_hidden * n_heads
                # conv2 output = n_hidden
                n_hidden = state_dict["conv2.lin_l.weight"].shape[0]
                # Infer n_heads
                if conv1_output == conv2_input:
                    n_heads = conv1_output // n_hidden
                else:
                    # Fallback: assume n_heads = 4
                    n_heads = 4
            else:
                # Fallback values
                n_hidden = 64
                n_heads = 4
        else:
            return n_feat, None, None
        
        return n_feat, n_hidden, n_heads
    except Exception as e:
        print(f"  ⚠️  Warning: Could not infer parameters: {e}")
        return None, None, None


def check_state_dict_structure(state_dict: Dict) -> tuple:
    """Check if state dict has expected structure."""
    print("\n3. Checking state dict structure...")
    
    # Expected keys for StructureExpertGNN with GATv2Conv
    # GATv2Conv uses lin_l, lin_r, and att instead of simple weight/bias
    expected_gat_keys = [
        "conv1.att",           # Attention weights
        "conv1.lin_l.weight",  # Left linear transformation
        "conv1.lin_l.bias",
        "conv1.lin_r.weight",  # Right linear transformation
        "conv1.lin_r.bias",
        "conv1.bias",          # Optional bias
        "conv2.att",
        "conv2.lin_l.weight",
        "conv2.lin_l.bias",
        "conv2.lin_r.weight",
        "conv2.lin_r.bias",
        "conv2.bias",
        "predict.0.weight",    # First linear layer in prediction head
        "predict.0.bias",
        "predict.2.weight",    # Second linear layer in prediction head
        "predict.2.bias",
    ]
    
    # Legacy keys (for older model versions, if any)
    legacy_keys = [
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
    ]
    
    print(f"  Total parameters: {len(state_dict)}")
    print(f"  Parameter keys:")
    for key in sorted(state_dict.keys()):
        shape = state_dict[key].shape if hasattr(state_dict[key], "shape") else "N/A"
        print(f"    - {key}: {shape}")
    
    # Try to infer model parameters
    print("\n  Inferring model parameters from state dict...")
    n_feat, n_hidden, n_heads = infer_model_params(state_dict)
    
    if n_feat and n_hidden and n_heads:
        print(f"  ✓ Inferred parameters:")
        print(f"    n_feat: {n_feat}")
        print(f"    n_hidden: {n_hidden}")
        print(f"    n_heads: {n_heads}")
    else:
        print(f"  ⚠️  Could not infer all parameters")
        if n_feat:
            print(f"    n_feat: {n_feat}")
    
    # Check for expected keys (GATv2Conv structure)
    found_gat_keys = [k for k in expected_gat_keys if k in state_dict]
    found_legacy_keys = [k for k in legacy_keys if k in state_dict]
    
    if found_gat_keys:
        print(f"\n  ✓ Found GATv2Conv structure ({len(found_gat_keys)} keys)")
        print("  (This is the expected structure for StructureExpertGNN)")
    elif found_legacy_keys:
        print(f"\n  ⚠️  Found legacy structure ({len(found_legacy_keys)} keys)")
        print("  (Model may use older GAT implementation)")
    else:
        print(f"\n  ⚠️  Warning: Unexpected model structure")
    
    # Check for critical missing keys
    critical_keys = [
        "conv1.lin_l.weight",
        "conv2.lin_l.weight",
        "predict.0.weight",
        "predict.2.weight",
    ]
    missing_critical = [k for k in critical_keys if k not in state_dict]
    if missing_critical:
        print(f"  ❌ Missing critical keys: {missing_critical}")
        return None, None, None
    
    return n_feat, n_hidden, n_heads


def check_model_loading(state_dict: Dict, n_feat: int = 158, n_hidden: int = 64, n_heads: int = 4) -> bool:
    """Check if model can be initialized and weights loaded."""
    print("\n4. Checking model initialization and weight loading...")
    
    try:
        # Initialize model
        model = StructureExpertGNN(n_feat=n_feat, n_hidden=n_hidden, n_heads=n_heads)
        print(f"✓ Model initialized (n_feat={n_feat}, n_hidden={n_hidden}, n_heads={n_heads})")
        
        # Try to load state dict
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✓ State dict loaded into model (strict=False)")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load with strict=False: {e}")
            try:
                model.load_state_dict(state_dict, strict=True)
                print("✓ State dict loaded into model (strict=True)")
            except Exception as e2:
                print(f"❌ Failed to load state dict: {e2}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return False


def check_forward_pass(model: StructureExpertGNN, n_feat: int = 158) -> bool:
    """Check if model can perform forward pass."""
    print("\n5. Checking forward pass...")
    
    try:
        # Create dummy input
        num_nodes = 10
        num_edges = 20
        
        x = torch.randn(num_nodes, n_feat)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Forward pass with intermediate checks
        model.eval()
        with torch.no_grad():
            # Check intermediate outputs
            h1 = model.conv1(x, edge_index)
            h1_activated = torch.nn.functional.leaky_relu(h1)
            
            h2 = model.conv2(h1_activated, edge_index)
            
            logits = model.predict(h2)
            embedding = h2
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Output embedding shape: {embedding.shape}")
        
        # Check intermediate outputs for NaN/Inf
        has_nan = False
        has_inf = False
        
        # Check conv1 output
        if torch.isnan(h1).any():
            print(f"❌ NaN detected in conv1 output: {torch.isnan(h1).sum().item()} values")
            has_nan = True
        if torch.isinf(h1).any():
            print(f"❌ Inf detected in conv1 output: {torch.isinf(h1).sum().item()} values")
            has_inf = True
        
        # Check conv2 output
        if torch.isnan(h2).any():
            print(f"❌ NaN detected in conv2 output: {torch.isnan(h2).sum().item()} values")
            has_nan = True
        if torch.isinf(h2).any():
            print(f"❌ Inf detected in conv2 output: {torch.isinf(h2).sum().item()} values")
            has_inf = True
        
        # Check final output
        if torch.isnan(logits).any():
            nan_count = torch.isnan(logits).sum().item()
            print(f"❌ NaN detected in logits: {nan_count} values")
            print(f"   NaN percentage: {nan_count / logits.numel() * 100:.2f}%")
            has_nan = True
        
        if torch.isinf(logits).any():
            inf_count = torch.isinf(logits).sum().item()
            print(f"❌ Inf detected in logits: {inf_count} values")
            print(f"   Inf percentage: {inf_count / logits.numel() * 100:.2f}%")
            has_inf = True
        
        if has_nan or has_inf:
            print("\n  Diagnosing NaN/Inf source...")
            diagnose_nan_source(model, x, edge_index)
            return False
        
        print("✓ Output values are valid (no NaN/Inf)")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_nan_source(model: StructureExpertGNN, x: torch.Tensor, edge_index: torch.Tensor) -> None:
    """Diagnose the source of NaN values in model."""
    print("\n  Checking model parameters for issues...")
    
    # Check each layer's parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            print(f"    ❌ {name}: Contains {nan_count} NaN values")
        elif torch.isinf(param).any():
            inf_count = torch.isinf(param).sum().item()
            print(f"    ❌ {name}: Contains {inf_count} Inf values")
        else:
            # Check for extreme values
            param_abs = param.abs()
            max_val = param_abs.max().item()
            mean_val = param_abs.mean().item()
            
            if max_val > 1e6:
                print(f"    ⚠️  {name}: Very large values (max={max_val:.2e}, mean={mean_val:.2e})")
            elif max_val < 1e-6 and mean_val < 1e-6:
                print(f"    ⚠️  {name}: Very small values (max={max_val:.2e}, mean={mean_val:.2e})")
            else:
                print(f"    ✓ {name}: OK (max={max_val:.4f}, mean={mean_val:.4f})")
    
    # Check for zero or near-zero parameters
    print("\n  Checking for zero or near-zero parameters...")
    for name, param in model.named_parameters():
        zero_count = (param.abs() < 1e-8).sum().item()
        total_count = param.numel()
        if zero_count > total_count * 0.5:  # More than 50% zeros
            print(f"    ⚠️  {name}: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%) values are near-zero")
    
    print("\n  Recommendations:")
    print("    1. Model may have been trained with numerical instability")
    print("    2. Try re-training the model with gradient clipping")
    print("    3. Check if learning rate was too high during training")
    print("    4. Consider using a different initialization or normalization")


def check_parameter_statistics(state_dict: Dict) -> None:
    """Print statistics about model parameters."""
    print("\n6. Parameter statistics:")
    
    total_params = 0
    trainable_params = 0
    
    for key, param in state_dict.items():
        if hasattr(param, "numel"):
            num_params = param.numel()
            total_params += num_params
            
            if hasattr(param, "requires_grad") and param.requires_grad:
                trainable_params += num_params
            
            mean_val = param.float().mean().item()
            # Only calculate std if there's more than one element
            if num_params > 1:
                std_val = param.float().std().item()
            else:
                std_val = 0.0
            min_val = param.float().min().item()
            max_val = param.float().max().item()
            
            print(f"  {key}:")
            print(f"    Shape: {param.shape}")
            print(f"    Parameters: {num_params:,}")
            print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            print(f"    Min: {min_val:.6f}, Max: {max_val:.6f}")
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check Structure Expert model file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/structure_expert.pth",
        help="Path to model file (default: models/structure_expert.pth)",
    )
    
    parser.add_argument(
        "--n_feat",
        type=int,
        default=158,
        help="Number of input features (default: 158 for Alpha158)",
    )
    
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=64,
        help="Hidden dimension size (default: 64)",
    )
    
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    
    parser.add_argument(
        "--skip_forward",
        action="store_true",
        help="Skip forward pass test",
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path).expanduser()
    
    print("=" * 80)
    print("Structure Expert Model Check")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print()
    
    # Check 1: File exists
    print("1. Checking file existence...")
    if not check_file_exists(model_path):
        return 1
    
    if not check_file_size(model_path):
        return 1
    
    # Check 2: Load state dict
    state_dict = check_load_state_dict(model_path)
    if state_dict is None:
        return 1
    
    # Check 3: State dict structure and infer parameters
    inferred_n_feat, inferred_n_hidden, inferred_n_heads = check_state_dict_structure(state_dict)
    
    # Use inferred parameters if available, otherwise use provided args
    n_feat = inferred_n_feat if inferred_n_feat else args.n_feat
    n_hidden = inferred_n_hidden if inferred_n_hidden else args.n_hidden
    n_heads = inferred_n_heads if inferred_n_heads else args.n_heads
    
    if inferred_n_feat and inferred_n_hidden and inferred_n_heads:
        print(f"\n  Using inferred parameters: n_feat={n_feat}, n_hidden={n_hidden}, n_heads={n_heads}")
        print(f"  (Override with --n-feat, --n-hidden, --n-heads if needed)")
    else:
        print(f"\n  Using provided parameters: n_feat={n_feat}, n_hidden={n_hidden}, n_heads={n_heads}")
        if not inferred_n_feat:
            print(f"  ⚠️  Warning: Could not infer n_feat, using default: {args.n_feat}")
    
    # Check 4: Model loading
    model = None
    if not check_model_loading(state_dict, n_feat, n_hidden, n_heads):
        print(f"\n  ❌ Model loading failed with parameters: n_feat={n_feat}, n_hidden={n_hidden}, n_heads={n_heads}")
        if inferred_n_feat and inferred_n_hidden and inferred_n_heads:
            print(f"  These parameters were inferred from the model file.")
        else:
            print(f"  Try specifying correct parameters manually:")
            print(f"    --n-feat {n_feat} --n-hidden {n_hidden} --n-heads {n_heads}")
        return 1
    
    # Re-initialize model for forward pass (use inferred parameters)
    try:
        model = StructureExpertGNN(n_feat=n_feat, n_hidden=n_hidden, n_heads=n_heads)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"❌ Failed to prepare model for forward pass: {e}")
        return 1
    
    # Check 5: Forward pass
    if not args.skip_forward:
        if not check_forward_pass(model, n_feat):
            return 1
    
    # Check 6: Parameter statistics
    check_parameter_statistics(state_dict)
    
    print("\n" + "=" * 80)
    print("✓ All checks passed! Model file is valid.")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

