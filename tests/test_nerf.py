"""
Test script to verify NeRF implementation.
"""

import torch
import yaml
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def get_config_path():
    """Get the path to the default config file."""
    # Look for config relative to project root
    config_path = os.path.join(project_root, 'configs', 'default.yaml')
    if os.path.exists(config_path):
        return config_path
    # If running from project root
    config_path = 'configs/default.yaml'
    if os.path.exists(config_path):
        return config_path
    raise FileNotFoundError("Could not find configs/default.yaml")

from src.models.nerf import create_nerf_model
from src.rendering.volume_renderer import create_volume_renderer
from src.utils.nerf_utils import get_rays, sample_rays


def test_model_creation():
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
    # Load config
    with open(get_config_path(), 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_nerf_model(config)
    print(f"Model created successfully")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 1000
    pts = torch.randn(batch_size, 3)
    dirs = torch.randn(batch_size, 3)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)  # Normalize directions
    
    with torch.no_grad():
        coarse_output, fine_output = model(pts, dirs, pts, dirs)
    
    print(f"Coarse output shape: {coarse_output.shape}")
    print(f"Fine output shape: {fine_output.shape}")
    
    assert coarse_output.shape == (batch_size, 4), f"Expected {(batch_size, 4)}, got {coarse_output.shape}"
    assert fine_output.shape == (batch_size, 4), f"Expected {(batch_size, 4)}, got {fine_output.shape}"
    
    print("✓ Model test passed!")
    return True


def test_ray_generation():
    """Test ray generation."""
    print("\nTesting ray generation...")
    
    H, W = 100, 100
    focal = 50.0
    c2w = torch.eye(4)[:3, :4]  # Identity camera pose
    
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    
    print(f"Ray origins shape: {rays_o.shape}")
    print(f"Ray directions shape: {rays_d.shape}")
    
    assert rays_o.shape == (H, W, 3)
    assert rays_d.shape == (H, W, 3)
    
    # Check that rays are normalized
    ray_norms = torch.norm(rays_d, dim=-1)
    print(f"Ray direction norms - min: {ray_norms.min():.4f}, max: {ray_norms.max():.4f}")
    
    print("✓ Ray generation test passed!")
    return True


def test_volume_renderer():
    """Test volume renderer."""
    print("\nTesting volume renderer...")
    
    # Load config
    with open(get_config_path(), 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    renderer = create_volume_renderer(config, device)
    
    # Create dummy model
    model = create_nerf_model(config)
    model.eval()
    
    # Generate test rays
    N_rays = 100
    rays_o = torch.randn(N_rays, 3)
    rays_d = torch.randn(N_rays, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Test rendering
    with torch.no_grad():
        outputs = renderer.render_rays(model, rays_o, rays_d)
    
    print(f"Render outputs keys: {list(outputs.keys())}")
    
    # Check output shapes
    for key, val in outputs.items():
        print(f"{key}: {val.shape}")
        if 'rgb' in key:
            assert val.shape == (N_rays, 3), f"RGB output should be ({N_rays}, 3), got {val.shape}"
        else:
            assert val.shape == (N_rays,), f"Scalar output should be ({N_rays},), got {val.shape}"
    
    print("✓ Volume renderer test passed!")
    return True


def test_full_pipeline():
    """Test the full pipeline with dummy data."""
    print("\nTesting full pipeline...")
    
    # Load config
    with open(get_config_path(), 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # Create model and renderer
    model = create_nerf_model(config).to(device)
    renderer = create_volume_renderer(config, device)
    model.eval()
    renderer.training = False
    
    # Test image rendering
    H, W = 50, 50  # Small image for testing
    focal = 25.0
    c2w = torch.eye(4, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = renderer.render_image(model, H, W, focal, c2w)
    
    print(f"Image render outputs: {list(outputs.keys())}")
    
    for key, val in outputs.items():
        print(f"{key}: {val.shape}")
        if 'rgb' in key:
            assert val.shape == (H, W, 3), f"RGB image should be ({H}, {W}, 3), got {val.shape}"
        else:
            assert val.shape == (H, W), f"Scalar image should be ({H}, {W}), got {val.shape}"
    
    print("✓ Full pipeline test passed!")
    return True


def main():
    """Run all tests."""
    print("Running NeRF implementation tests...\n")
    
    tests = [
        test_model_creation,
        test_ray_generation,
        test_volume_renderer,
        test_full_pipeline
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed!")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your NeRF implementation is ready.")
        print("\nNext steps:")
        print("1. Place your images in data/images/")
        print("2. Run: python prepare_data.py --images_dir path/to/your/images")
        print("3. Start training: python train.py")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == len(tests)


if __name__ == '__main__':
    main()
