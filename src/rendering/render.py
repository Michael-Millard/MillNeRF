"""
Rendering script for novel view synthesis.
"""

import torch
import yaml
import numpy as np
import imageio
import os
from tqdm import tqdm

from ..models.nerf import create_nerf_model
from .volume_renderer import create_volume_renderer
from ..utils.camera_utils import generate_render_path


def render_novel_views(model, renderer, config, output_dir, num_frames=60):
    """Render a 360-degree video of novel views."""
    print(f"Rendering {num_frames} novel views...")
    
    # Get device from renderer
    device = renderer.device
    
    # Create a simple circular camera path
    def create_circular_poses(num_frames, radius=3.0, height=0.0):
        """Create a simple circular camera path."""
        poses = []
        for i in range(num_frames):
            theta = 2 * np.pi * i / num_frames
            # Camera position
            cam_pos = np.array([
                radius * np.cos(theta),
                radius * np.sin(theta), 
                height
            ])
            
            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0., 0., 1.])
            right = np.cross(forward, up)
            up = np.cross(right, forward)
            
            # Create pose matrix
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward  # Negative because camera looks down -z
            pose[:3, 3] = cam_pos
            
            poses.append(pose)
        return poses
    
    poses = create_circular_poses(num_frames, radius=3.0, height=0.0)

    # Get image dimensions from config or use reasonable defaults
    data_config = config.get('data', {})
    H = data_config.get('image_height', 200)
    W = data_config.get('image_width', 200) 
    focal = data_config.get('focal_length', 200)
    
    os.makedirs(output_dir, exist_ok=True)
    images = []
    
    model.eval()
    with torch.no_grad():
        for i, pose in enumerate(tqdm(poses, desc="Rendering views")):
            pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)
            
            # Render image
            outputs = renderer.render_image(model, H, W, focal, pose_tensor)
            
            # Use fine network output if available, otherwise coarse
            if 'rgb_fine' in outputs:
                rgb = outputs['rgb_fine'].cpu().numpy()
            else:
                rgb = outputs['rgb_coarse'].cpu().numpy()
            
            # Convert to uint8
            rgb = (rgb * 255).astype(np.uint8)
            images.append(rgb)
            
            # Save individual frame
            imageio.imwrite(f"{output_dir}/frame_{i:04d}.png", rgb)
    
    # Save as video
    imageio.mimsave(f"{output_dir}/novel_views.mp4", images, fps=10)
    print(f"Saved novel view video to {output_dir}/novel_views.mp4")


def main(args=None):
    """Main rendering function."""
    import argparse
    
    if args is None:
        parser = argparse.ArgumentParser(description='Render NeRF novel views')
        parser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to model checkpoint')
        parser.add_argument('--config', type=str, default='configs/default.yaml',
                           help='Path to config file')
        parser.add_argument('--mode', type=str, default='novel',
                           choices=['novel', 'train', 'test'],
                           help='Rendering mode')
        parser.add_argument('--output_dir', type=str, default='build/renders',
                           help='Output directory')
        args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and load checkpoint
    model = create_nerf_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create renderer
    renderer = create_volume_renderer(config, device)
    
    if args.mode == 'novel':
        render_novel_views(model, renderer, config, args.output_dir)
    else:
        print(f"Rendering mode '{args.mode}' not implemented yet")


if __name__ == '__main__':
    main()