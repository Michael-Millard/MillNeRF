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
from ..utils.camera_utils import generate_circular_path


def render_novel_views(model, renderer, config, output_dir, num_frames=60):
    """Render a 360-degree video of novel views."""
    print(f"Rendering {num_frames} novel views...")
    
    # Generate circular camera path
    poses = generate_circular_path(num_frames, radius=3.0, height=0.0)
    
    H = config['data'].get('image_height', 400)
    W = config['data'].get('image_width', 400)
    focal = config['data'].get('focal_length', 400)
    
    os.makedirs(output_dir, exist_ok=True)
    images = []
    
    model.eval()
    with torch.no_grad():
        for i, pose in enumerate(tqdm(poses, desc="Rendering views")):
            pose_tensor = torch.tensor(pose, dtype=torch.float32)
            
            # Render image
            outputs = renderer.render_image(model, H, W, focal, pose_tensor)
            rgb = outputs['rgb_fine'].cpu().numpy()
            
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
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create renderer
    renderer = create_volume_renderer(config, device)
    
    if args.mode == 'novel':
        render_novel_views(model, renderer, config, args.output_dir)
    else:
        print(f"Rendering mode '{args.mode}' not implemented yet")


if __name__ == '__main__':
    main()