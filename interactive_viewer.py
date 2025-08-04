"""
Interactive 3D NeRF Viewer
Allows real-time exploration of a trained NeRF model with mouse controls.
"""

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
from pathlib import Path
import argparse

from src.models.nerf import create_nerf_model
from src.rendering.volume_renderer import create_volume_renderer


class InteractiveNeRFViewer:
    def __init__(self, checkpoint_path, config_path):
        """Initialize the interactive NeRF viewer."""
        print("Loading NeRF model...")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = create_nerf_model(self.config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create renderer
        self.renderer = create_volume_renderer(self.config, self.device)
        self.renderer.training = False
        
        # Camera parameters
        self.radius = 3.0
        self.theta = 0.0  # Horizontal angle
        self.phi = 0.0    # Vertical angle
        self.height = 0.0
        
        # Rendering parameters
        self.H = 200
        self.W = 200
        self.focal = 200
        
        # Setup interactive plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the interactive matplotlib plot."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle('Interactive NeRF Viewer\nUse sliders to move around the scene', fontsize=14)
        
        # Create initial image
        initial_image = self.render_view()
        self.im = self.ax.imshow(initial_image)
        self.ax.set_title(f'Theta: {self.theta:.1f}°, Phi: {self.phi:.1f}°, Radius: {self.radius:.1f}')
        self.ax.axis('off')
        
        # Create sliders
        plt.subplots_adjust(bottom=0.3)
        
        # Theta slider (horizontal rotation)
        ax_theta = plt.axes([0.1, 0.15, 0.8, 0.03])
        self.slider_theta = Slider(ax_theta, 'Horizontal', 0, 360, valinit=self.theta)
        self.slider_theta.on_changed(self.update_theta)
        
        # Phi slider (vertical angle)
        ax_phi = plt.axes([0.1, 0.1, 0.8, 0.03])
        self.slider_phi = Slider(ax_phi, 'Vertical', -60, 60, valinit=self.phi)
        self.slider_phi.on_changed(self.update_phi)
        
        # Radius slider (distance)
        ax_radius = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.slider_radius = Slider(ax_radius, 'Distance', 1.0, 6.0, valinit=self.radius)
        self.slider_radius.on_changed(self.update_radius)
        
        # Height slider
        ax_height = plt.axes([0.1, 0.2, 0.8, 0.03])
        self.slider_height = Slider(ax_height, 'Height', -2.0, 2.0, valinit=self.height)
        self.slider_height.on_changed(self.update_height)
        
    def create_camera_pose(self):
        """Create camera pose from current parameters."""
        # Convert angles to radians
        theta_rad = np.radians(self.theta)
        phi_rad = np.radians(self.phi)
        
        # Camera position
        cam_pos = np.array([
            self.radius * np.cos(phi_rad) * np.cos(theta_rad),
            self.radius * np.cos(phi_rad) * np.sin(theta_rad),
            self.radius * np.sin(phi_rad) + self.height
        ])
        
        # Look at origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        
        # Up vector (adjust based on camera angle)
        up = np.array([0., 0., 1.])
        
        # Create right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recalculate up vector
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward  # Camera looks down -z
        pose[:3, 3] = cam_pos
        
        return pose
        
    def render_view(self):
        """Render current view."""
        pose = self.create_camera_pose()
        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            outputs = self.renderer.render_image(self.model, self.H, self.W, self.focal, pose_tensor)
            
            # Use fine network output if available, otherwise coarse
            if 'rgb_fine' in outputs:
                rgb = outputs['rgb_fine'].cpu().numpy()
            else:
                rgb = outputs['rgb_coarse'].cpu().numpy()
            
            # Convert to uint8
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            
        return rgb
        
    def update_theta(self, val):
        """Update horizontal angle."""
        self.theta = val
        self.update_view()
        
    def update_phi(self, val):
        """Update vertical angle."""
        self.phi = val
        self.update_view()
        
    def update_radius(self, val):
        """Update camera distance."""
        self.radius = val
        self.update_view()
        
    def update_height(self, val):
        """Update camera height."""
        self.height = val
        self.update_view()
        
    def update_view(self):
        """Update the displayed image."""
        print(f"Rendering view: θ={self.theta:.1f}°, φ={self.phi:.1f}°, r={self.radius:.1f}, h={self.height:.1f}")
        
        # Render new view
        new_image = self.render_view()
        
        # Update display
        self.im.set_array(new_image)
        self.ax.set_title(f'θ: {self.theta:.1f}°, φ: {self.phi:.1f}°, r: {self.radius:.1f}, h: {self.height:.1f}')
        self.fig.canvas.draw_idle()
        
    def run(self):
        """Start the interactive viewer."""
        print("\n🎮 Interactive NeRF Viewer Controls:")
        print("• Horizontal: Rotate around the scene (0-360°)")
        print("• Vertical: Look up/down (-60° to +60°)")
        print("• Distance: Move closer/farther (1-6 units)")
        print("• Height: Move camera up/down (-2 to +2 units)")
        print("• Close window to exit")
        
        plt.show()


def create_web_viewer(checkpoint_path, config_path, output_dir="build/web_viewer"):
    """Create a web-based viewer using pre-rendered views."""
    import os
    
    print("Creating web-based viewer...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_nerf_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    renderer = create_volume_renderer(config, device)
    renderer.training = False
    
    # Pre-render views at different angles
    print("Pre-rendering views...")
    H, W, focal = 300, 300, 300
    
    theta_steps = 36  # Every 10 degrees
    phi_steps = 7     # Every 20 degrees from -60 to +60
    
    total_views = theta_steps * phi_steps
    view_count = 0
    
    for i, theta in enumerate(np.linspace(0, 360, theta_steps, endpoint=False)):
        for j, phi in enumerate(np.linspace(-60, 60, phi_steps)):
            # Create camera pose
            theta_rad = np.radians(theta)
            phi_rad = np.radians(phi)
            radius = 3.0
            
            cam_pos = np.array([
                radius * np.cos(phi_rad) * np.cos(theta_rad),
                radius * np.cos(phi_rad) * np.sin(theta_rad),
                radius * np.sin(phi_rad)
            ])
            
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0., 0., 1.])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward
            pose[:3, 3] = cam_pos
            
            pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)
            
            # Render
            with torch.no_grad():
                outputs = renderer.render_image(model, H, W, focal, pose_tensor)
                if 'rgb_fine' in outputs:
                    rgb = outputs['rgb_fine'].cpu().numpy()
                else:
                    rgb = outputs['rgb_coarse'].cpu().numpy()
                
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            
            # Save image
            filename = f"view_{i:02d}_{j:02d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            view_count += 1
            print(f"Rendered {view_count}/{total_views} views", end='\r')
    
    print(f"\nSaved {total_views} views to {output_dir}")
    print("You can create a web gallery or slideshow with these images!")


def main():
    parser = argparse.ArgumentParser(description='Interactive NeRF Viewer')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'web'],
                       help='Viewer mode')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        viewer = InteractiveNeRFViewer(args.checkpoint, args.config)
        viewer.run()
    else:
        create_web_viewer(args.checkpoint, args.config)


if __name__ == '__main__':
    main()
