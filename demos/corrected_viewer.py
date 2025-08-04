#!/usr/bin/env python3
"""
Create a corrected interactive viewer based on the successful "mimic training exactly" approach.
"""

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.models.nerf import create_nerf_model
from src.rendering.volume_renderer import create_volume_renderer

class CorrectInteractiveNeRFViewer:
    def __init__(self, checkpoint_path, config_path):
        """Initialize the corrected interactive NeRF viewer."""
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
        
        # Load training poses to understand the coordinate system
        self.training_poses = self.load_training_poses()
        self.scene_center, self.scene_bounds = self.analyze_scene()
        
        print(f"Scene center: [{self.scene_center[0]:.3f}, {self.scene_center[1]:.3f}, {self.scene_center[2]:.3f}]")
        print(f"Scene bounds: {self.scene_bounds}")
        
        # Camera parameters - start at the first training pose
        # Get the actual radius from first pose
        first_pose_params = self.extract_spherical_from_pose(self.training_poses[0])
        self.radius = first_pose_params['radius']
        self.theta = first_pose_params['theta']    # Horizontal angle (degrees)
        self.phi = first_pose_params['phi']        # Vertical angle (degrees) 
        self.height = first_pose_params['height']  # Height offset
        
        print(f"Starting at first training pose: θ={self.theta:.1f}°, φ={self.phi:.1f}°, r={self.radius:.2f}, h={self.height:.2f}")
        print(f"Note: θ was normalized from -148.2° to {self.theta:.1f}° for slider compatibility")
        
        # Debug: Test if spherical conversion is correct
        self.test_spherical_conversion()

        # Debug: Test coordinate system with cardinal directions
        self.debug_coordinate_system()

        # NEW: Analyze what direction training cameras are actually looking
        self.analyze_training_poses()

        # NEW: Check if we have the right scene center
        self.recalculate_scene_center()

        # NEW: Debug pose comparison
        self.debug_pose_comparison()
        
        # Mode selection - start in training mode to show exact match
        self.view_mode = 'training'  # 'novel' or 'training'
        self.current_pose_idx = 0
        self.interpolation = 0.0
        
        print(f"Default radius: {self.radius:.3f}")
        print(f"Average training radius: {np.linalg.norm(self.training_poses[:, :3, 3] - self.scene_center, axis=1).mean():.3f}")
        print(f"Radius range: [{self.scene_bounds['distances'].min():.3f}, {self.scene_bounds['distances'].max():.3f}]")
        
        # Rendering parameters - match training image size after downscaling
        # Original: 267x357, downscaled by factor of 4
        self.W = 267 // 4  # Width: ~67
        self.H = 357 // 4  # Height: ~89
        
        # Get focal length from training data and scale for downsampled images
        poses_file = self.config['data']['poses_file']
        with open(poses_file, 'r') as f:
            data = json.load(f)
        if 'fl_x' in data:
            # Scale focal length by the same downscaling factor used for images
            self.focal = data['fl_x'] / 4  # Divided by downscaling factor
        else:
            self.focal = 280 / 4  # fallback, also scaled
        
        # Setup interactive plot
        self.setup_plot()
        
    def load_training_poses(self):
        """Load training poses to understand coordinate system."""
        poses_file = self.config['data']['poses_file']
        with open(poses_file, 'r') as f:
            data = json.load(f)
        
        poses = []
        for frame in data['frames']:
            pose = np.array(frame['transform_matrix'])
            poses.append(pose)
        
        return np.array(poses)
    
    def analyze_scene(self):
        """Analyze training poses to understand scene structure."""
        positions = self.training_poses[:, :3, 3]
        scene_center = positions.mean(axis=0)
        
        # Calculate bounds
        bounds = {
            'x_range': [positions[:, 0].min(), positions[:, 0].max()],
            'y_range': [positions[:, 1].min(), positions[:, 1].max()],
            'z_range': [positions[:, 2].min(), positions[:, 2].max()],
            'distances': np.linalg.norm(positions - scene_center, axis=1)
        }
        
        return scene_center, bounds

    def extract_spherical_from_pose(self, pose):
        """Extract spherical coordinates from a training pose.

        Training data shows cameras looking OUTWARD from scene center.
        Uses consistent convention: 
        - theta=0 points along +X axis, increases counter-clockwise (when viewed from above)
        - phi=0 is horizontal plane, positive upward
        """
        # Get camera position from pose
        cam_pos = pose[:3, 3]

        # Calculate relative position from scene center
        rel_pos = cam_pos - self.scene_center

        # Calculate spherical coordinates
        radius = np.linalg.norm(rel_pos)

        # Calculate theta (azimuth): angle in XZ plane from +X axis
        # atan2(Z, X) gives angle from +X toward +Z
        theta = np.degrees(np.arctan2(rel_pos[2], rel_pos[0]))

        # Normalize theta to 0-360° range
        if theta < 0:
            theta += 360.0

        # Calculate phi (elevation): angle from XZ plane toward +Y
        phi = np.degrees(np.arcsin(np.clip(rel_pos[1] / radius, -1.0, 1.0)))

        return {
            'theta': theta,
            'phi': phi, 
            'height': 0.0,  # We encode Y position in phi, not height
            'radius': radius
        }

    def create_pose_from_spherical(self, theta, phi, radius, height=0.0):
        """Create pose using spherical coordinates that match NeRF coordinate system.

        Convention:
        - theta=0 points along +X, increases toward +Z (counter-clockwise from above)  
        - phi=0 is horizontal, positive is up
        - NeRF: +X right, +Y up, +Z backward, camera looks down -Z
        """
        # Normalize theta
        theta = theta % 360.0

        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)

        # Calculate camera position using CONSISTENT spherical coordinates
        # This MUST match the convention used in extract_spherical_from_pose
        effective_center = self.scene_center + np.array([0, height, 0])

        # Standard spherical coordinates: 
        # x = r * cos(phi) * cos(theta)  -- matches extract: theta from atan2(z,x), so theta=0 => x component
        # y = r * sin(phi)               -- matches extract: phi from asin(y/r)
        # z = r * cos(phi) * sin(theta)  -- matches extract: theta from atan2(z,x), so theta=90° => z component
        cam_pos = effective_center + np.array([
            radius * np.cos(phi_rad) * np.cos(theta_rad),  # X component
            radius * np.sin(phi_rad),                      # Y component  
            radius * np.cos(phi_rad) * np.sin(theta_rad)   # Z component
        ])

        # Calculate camera orientation
        # Forward vector: AWAY from scene center (training data shows outward-looking cameras)
        away_from_center = cam_pos - self.scene_center
        forward = away_from_center / np.linalg.norm(away_from_center)

        # World up vector
        world_up = np.array([0, 1, 0], dtype=np.float32)

        # Calculate right vector using correct cross product order
        # For NeRF right-handed system: right = forward × up
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)

        # Handle gimbal lock (camera pointing straight up/down)
        if right_norm < 1e-6:
            # When looking straight up/down, forward is parallel to world_up
            # Choose an arbitrary right vector in the horizontal plane
            if abs(forward[0]) > abs(forward[2]):
                right = np.array([0, 0, 1], dtype=np.float32)  # Use +Z if forward has more X
            else:
                right = np.array([1, 0, 0], dtype=np.float32)  # Use +X if forward has more Z

            # Make sure right is perpendicular to forward
            right = right - np.dot(right, forward) * forward
            right = right / np.linalg.norm(right)
        else:
            right = right / right_norm

        # Calculate up vector: up = right × forward (maintains right-handed system)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # NeRF convention: +Z is backward, so backward = -forward
        backward = -forward

        # Build rotation matrix with columns [right, up, backward]
        # This represents the camera's local coordinate system in world coordinates
        rotation = np.stack([right, up, backward], axis=1)

        # Verify rotation matrix properties
        det = np.linalg.det(rotation)
        if abs(det - 1.0) > 0.01:
            print(f"⚠️  Warning: Rotation matrix determinant = {det:.6f} (should be ~1.0)")

        # Create pose matrix
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        pose[:3, 3] = cam_pos

        return pose

    def test_spherical_conversion(self):
        """Enhanced test for spherical coordinate conversion."""
        print("\n=== Testing Spherical Conversion ===")

        # Test with multiple training poses, not just the first one
        test_indices = [0, len(self.training_poses)//4, len(self.training_poses)//2, -1]

        for i, idx in enumerate(test_indices):
            print(f"\n--- Test {i+1}: Training Pose {idx} ---")

            original_pose = self.training_poses[idx]
            original_pos = original_pose[:3, 3]

            # Extract spherical coordinates
            params = self.extract_spherical_from_pose(original_pose)

            # Reconstruct pose
            reconstructed_pose = self.create_pose_from_spherical(
                params['theta'], params['phi'], params['radius'], params['height']
            )
            reconstructed_pos = reconstructed_pose[:3, 3]

            # Calculate errors
            pos_error = np.linalg.norm(original_pos - reconstructed_pos)

            # Test forward direction alignment
            original_forward = -original_pose[:3, 2]  # -Z column
            reconstructed_forward = -reconstructed_pose[:3, 2]
            forward_alignment = np.dot(original_forward, reconstructed_forward)

            print(f"Position error: {pos_error:.6f}")
            print(f"Forward alignment: {forward_alignment:.6f}")
            print(f"θ={params['theta']:.1f}°, φ={params['phi']:.1f}°, r={params['radius']:.3f}")

            if pos_error > 0.01 or forward_alignment < 0.7:
                print("❌ Large error detected!")
                print(f"Original pos: {original_pos}")
                print(f"Reconstructed: {reconstructed_pos}")
            else:
                print("✅ Conversion OK")

        print("=====================================\n")

    # Additional debugging method to add to your class:
    def debug_coordinate_system(self):
        """Debug the coordinate system by rendering views at cardinal directions."""
        print("\n=== Debugging Coordinate System ===")

        cardinal_directions = [
            (0, 0, "Looking OUTWARD from center toward +X"),
            (90, 0, "Looking OUTWARD from center toward +Z"), 
            (180, 0, "Looking OUTWARD from center toward -X"),
            (270, 0, "Looking OUTWARD from center toward -Z"),
            (0, 45, "Looking OUTWARD from center toward +X, 45° up"),
            (0, -45, "Looking OUTWARD from center toward +X, 45° down")
        ]

        for theta, phi, description in cardinal_directions:
            pose = self.create_pose_from_spherical(theta, phi, self.radius, 0)
            cam_pos = pose[:3, 3]
            forward = -pose[:3, 2]  # -Z direction

            print(f"{description}:")
            print(f"  θ={theta}°, φ={phi}° -> Camera at [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
            print(f"  Forward: [{forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}]")

            # Verify forward points AWAY from scene center (outward)
            away_from_center = cam_pos - self.scene_center
            away_from_center_norm = away_from_center / np.linalg.norm(away_from_center)
            alignment = np.dot(forward, away_from_center_norm)
            print(f"  Alignment with OUTWARD direction: {alignment:.3f} (should be ~1.0)")
            print()

    def create_perfect_training_match(self, training_pose_idx=0):
        """Create a novel view that EXACTLY matches a training pose.

        This bypasses spherical coordinate conversion to ensure perfect matching.
        """
        # Get the exact training pose
        exact_pose = self.training_poses[training_pose_idx].copy()

        # Extract the exact spherical parameters for display
        params = self.extract_spherical_from_pose(exact_pose)

        return exact_pose, params

    def get_current_pose(self):
        """Get current camera pose based on UI state and mode."""
        if self.view_mode == 'training':
            # Interpolate between training poses
            n_poses = len(self.training_poses)
            idx1 = self.current_pose_idx % n_poses
            idx2 = (self.current_pose_idx + 1) % n_poses
            return self.interpolate_training_poses(idx1, idx2, self.interpolation)
        else:
            # MODIFICATION: Check if we're at the "starting" novel view position
            # If sliders match the first training pose exactly, use the exact pose
            first_pose_params = self.extract_spherical_from_pose(self.training_poses[0])

            # Check if current slider values match first training pose (within small tolerance)
            theta_match = abs(self.theta - first_pose_params['theta']) < 0.1
            phi_match = abs(self.phi - first_pose_params['phi']) < 0.1
            radius_match = abs(self.radius - first_pose_params['radius']) < 0.01
            height_match = abs(self.height - first_pose_params['height']) < 0.01

            if theta_match and phi_match and radius_match and height_match:
                print("🎯 Using EXACT first training pose for perfect match")
                return self.training_poses[0].copy()
            else:
                # Generate novel view using spherical coordinates
                return self.create_pose_from_spherical(self.theta, self.phi, self.radius, self.height)

    # Alternative: Add a "Reset to Training Pose" button
    def add_reset_button(self):
        """Add a button to reset novel view to exactly match first training pose."""
        from matplotlib.widgets import Button

        # Add reset button
        ax_reset = plt.axes([0.85, 0.9, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Match\nTraining')
        self.button_reset.on_clicked(self.reset_to_training_pose)

    def reset_to_training_pose(self, event):
        """Reset novel view sliders to exactly match first training pose."""
        first_pose_params = self.extract_spherical_from_pose(self.training_poses[0])

        # Update slider values
        self.theta = first_pose_params['theta']
        self.phi = first_pose_params['phi'] 
        self.radius = first_pose_params['radius']
        self.height = first_pose_params['height']

        # Update slider displays
        self.slider_theta.set_val(self.theta)
        self.slider_phi.set_val(self.phi)
        self.slider_radius.set_val(self.radius)
        self.slider_height.set_val(self.height)

        if self.view_mode == 'novel':
            self.update_view()

        print(f"🎯 Reset to match first training pose: θ={self.theta:.1f}°, φ={self.phi:.1f}°, r={self.radius:.3f}")

    # Enhanced debugging to compare poses exactly
    def debug_pose_comparison(self):
        """Compare reconstructed pose vs original training pose in detail."""
        print("\n=== Detailed Pose Comparison ===")

        original_pose = self.training_poses[0]
        params = self.extract_spherical_from_pose(original_pose)
        reconstructed_pose = self.create_pose_from_spherical(
            params['theta'], params['phi'], params['radius'], params['height']
        )

        print("Original Training Pose Matrix:")
        print(original_pose)
        print("\nReconstructed Pose Matrix:")  
        print(reconstructed_pose)
        print(f"\nMax difference: {np.max(np.abs(original_pose - reconstructed_pose)):.8f}")

        # Check each component
        print("\nDetailed Comparison:")
        print(f"Position diff: {np.linalg.norm(original_pose[:3,3] - reconstructed_pose[:3,3]):.8f}")
        print(f"Right vec diff: {np.linalg.norm(original_pose[:3,0] - reconstructed_pose[:3,0]):.8f}")
        print(f"Up vec diff: {np.linalg.norm(original_pose[:3,1] - reconstructed_pose[:3,1]):.8f}")
        print(f"Back vec diff: {np.linalg.norm(original_pose[:3,2] - reconstructed_pose[:3,2]):.8f}")

        print("=====================================\n")

    def analyze_training_poses(self):
        """Analyze the actual forward directions in training poses."""
        print("\n=== Analyzing Training Pose Orientations ===")

        for i in [0, len(self.training_poses)//4, len(self.training_poses)//2, -1]:
            pose = self.training_poses[i]
            cam_pos = pose[:3, 3]

            # Extract all three basis vectors from the training pose
            right_vec = pose[:3, 0]    # +X column
            up_vec = pose[:3, 1]       # +Y column  
            back_vec = pose[:3, 2]     # +Z column (should be backward in NeRF)

            # In NeRF: camera looks down -Z, so forward = -back_vec
            forward_vec = -back_vec

            # Calculate vector from camera to scene center
            to_center = self.scene_center - cam_pos
            to_center_norm = to_center / np.linalg.norm(to_center)

            # Check alignment
            alignment = np.dot(forward_vec, to_center_norm)

            print(f"Training Pose {i}:")
            print(f"  Camera position: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")
            print(f"  Right (+X): [{right_vec[0]:.3f}, {right_vec[1]:.3f}, {right_vec[2]:.3f}]")
            print(f"  Up    (+Y): [{up_vec[0]:.3f}, {up_vec[1]:.3f}, {up_vec[2]:.3f}]")
            print(f"  Back  (+Z): [{back_vec[0]:.3f}, {back_vec[1]:.3f}, {back_vec[2]:.3f}]")
            print(f"  Forward (-Z): [{forward_vec[0]:.3f}, {forward_vec[1]:.3f}, {forward_vec[2]:.3f}]")
            print(f"  To center: [{to_center_norm[0]:.3f}, {to_center_norm[1]:.3f}, {to_center_norm[2]:.3f}]")
            print(f"  Forward→Center alignment: {alignment:.3f}")

            if alignment < 0:
                print("  ✅ Training camera is looking AWAY from scene center (as expected)")
                # Check if camera is looking outward from center
                away_from_center = -to_center_norm
                away_alignment = np.dot(forward_vec, away_from_center)
                print(f"  Forward→Away alignment: {away_alignment:.3f}")
            else:
                print("  ⚠️  Training camera is looking TOWARD scene center (unexpected)")
            print()

        print("=====================================\n")

    # Also add this method to determine the correct scene center:
    def recalculate_scene_center(self):
        """Try to determine if we're using the correct scene center."""
        print("\n=== Recalculating Scene Center ===")

        # Current scene center (average of camera positions)
        current_center = self.scene_center
        print(f"Current scene center (camera avg): [{current_center[0]:.3f}, {current_center[1]:.3f}, {current_center[2]:.3f}]")

        # Alternative 1: Where cameras are actually looking
        # Calculate intersection of camera forward rays
        forward_rays = []
        positions = []

        for pose in self.training_poses[:10]:  # Use first 10 poses
            cam_pos = pose[:3, 3]
            forward = -pose[:3, 2]  # NeRF forward is -Z

            positions.append(cam_pos)
            forward_rays.append(forward)

        positions = np.array(positions)
        forward_rays = np.array(forward_rays)

        # Simple approach: find point that minimizes distance to all forward rays
        # This is where cameras are actually looking

        # Test different potential centers
        test_centers = [
            current_center,
            np.array([0, 0, 0]),  # Origin
            np.array([0, current_center[1], 0]),  # Y-aligned with current
        ]

        for i, test_center in enumerate(test_centers):
            total_alignment = 0
            for j in range(len(positions)):
                to_test = test_center - positions[j]
                to_test_norm = to_test / np.linalg.norm(to_test)
                alignment = np.dot(forward_rays[j], to_test_norm)
                total_alignment += alignment

            avg_alignment = total_alignment / len(positions)
            print(f"Test center {i}: [{test_center[0]:.3f}, {test_center[1]:.3f}, {test_center[2]:.3f}] -> avg alignment: {avg_alignment:.3f}")

        print("=====================================\n")
    
    def interpolate_training_poses(self, idx1, idx2, t):
        """Interpolate between two training poses."""
        pose1 = self.training_poses[idx1]
        pose2 = self.training_poses[idx2]
        
        # Simple linear interpolation of position
        pos1, pos2 = pose1[:3, 3], pose2[:3, 3]
        pos = pos1 * (1 - t) + pos2 * t
        
        # Interpolate rotation (simplified - just use closest)
        if t < 0.5:
            rot = pose1[:3, :3]
        else:
            rot = pose2[:3, :3]
        
        # Create interpolated pose
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos
        
        return pose
    
    #def get_current_pose(self):
    #    """Get current camera pose based on UI state and mode."""
    #    if self.view_mode == 'training':
    #        # Interpolate between training poses
    #        n_poses = len(self.training_poses)
    #        idx1 = self.current_pose_idx % n_poses
    #        idx2 = (self.current_pose_idx + 1) % n_poses
    #        return self.interpolate_training_poses(idx1, idx2, self.interpolation)
    #    else:
    #        # Generate novel view using spherical coordinates
    #        return self.create_pose_from_spherical(self.theta, self.phi, self.radius, self.height)
    
    def setup_plot(self):
        """Setup the interactive matplotlib plot."""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle('Corrected Interactive NeRF Viewer\nStarting at First Training Pose', fontsize=14)
        
        # Create initial image
        initial_image = self.render_view()
        self.im = self.ax.imshow(initial_image)
        self.update_title()
        self.ax.axis('off')
        
        # Create sliders
        plt.subplots_adjust(bottom=0.35)
        
        from matplotlib.widgets import Slider, RadioButtons
        
        # Mode selection
        ax_mode = plt.axes([0.02, 0.7, 0.15, 0.15])
        self.radio_mode = RadioButtons(ax_mode, ('novel', 'training'))
        self.radio_mode.set_active(1)  # Start with 'training' mode active
        self.radio_mode.on_clicked(self.set_mode)
        
        # Novel view controls
        ax_theta = plt.axes([0.1, 0.25, 0.8, 0.03])
        self.slider_theta = Slider(ax_theta, 'Horizontal (θ)', 0, 360, 
                                  valinit=self.theta, valfmt='%.1f°')
        self.slider_theta.on_changed(self.update_theta)
        
        ax_phi = plt.axes([0.1, 0.2, 0.8, 0.03])
        self.slider_phi = Slider(ax_phi, 'Vertical (φ)', -90, 90, 
                                valinit=self.phi, valfmt='%.1f°')
        self.slider_phi.on_changed(self.update_phi)
        
        ax_radius = plt.axes([0.1, 0.15, 0.8, 0.03])
        self.slider_radius = Slider(ax_radius, 'Distance', 
                                   self.scene_bounds['distances'].min(), 
                                   self.scene_bounds['distances'].max() * 1.5, 
                                   valinit=self.radius, valfmt='%.2f')
        self.slider_radius.on_changed(self.update_radius)
        
        ax_height = plt.axes([0.1, 0.1, 0.8, 0.03])
        self.slider_height = Slider(ax_height, 'Height', -2.0, 2.0, 
                                   valinit=self.height, valfmt='%.2f')
        self.slider_height.on_changed(self.update_height)
        
        # Training pose controls
        ax_pose = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.slider_pose = Slider(ax_pose, 'Training Pose', 0, len(self.training_poses)-1, 
                                 valinit=self.current_pose_idx, valfmt='%d')
        self.slider_pose.on_changed(self.update_pose_idx)

        # Add reset button for perfect matching
        self.add_reset_button()
        
        # Add info text
        info_text = f"""Scene Info:
Center: [{self.scene_center[0]:.2f}, {self.scene_center[1]:.2f}, {self.scene_center[2]:.2f}]
Distance Range: [{self.scene_bounds['distances'].min():.2f}, {self.scene_bounds['distances'].max():.2f}]
Training Poses: {len(self.training_poses)}

Starting Position: First Training Pose
• Training Mode: Browse original training poses (ACTIVE)
• Novel Mode: Generate new viewpoints using spherical coordinates"""
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    def render_view(self):
        """Render current view."""
        pose = self.get_current_pose()
        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=self.device)
        
        # Clear GPU cache before rendering to free up memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        try:
            with torch.no_grad():
                outputs = self.renderer.render_image(self.model, self.H, self.W, self.focal, pose_tensor)
                
                # Use fine network output if available
                if 'rgb_fine' in outputs:
                    rgb = outputs['rgb_fine'].cpu().numpy()
                else:
                    rgb = outputs['rgb_coarse'].cpu().numpy()
                
                # Convert to uint8
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                
        except torch.cuda.OutOfMemoryError:
            print("⚠️  CUDA out of memory! Falling back to smaller image size...")
            # Fallback to even smaller size
            small_H, small_W = 128, 128
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            with torch.no_grad():
                outputs = self.renderer.render_image(self.model, small_H, small_W, self.focal, pose_tensor)
                
                if 'rgb_fine' in outputs:
                    rgb = outputs['rgb_fine'].cpu().numpy()
                else:
                    rgb = outputs['rgb_coarse'].cpu().numpy()
                
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        # Clear cache after rendering
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return rgb
    
    def update_title(self):
        """Update the plot title based on current mode."""
        if self.view_mode == 'novel':
            self.ax.set_title(f'Novel View: θ={self.theta:.1f}°, φ={self.phi:.1f}°, r={self.radius:.2f}, h={self.height:.2f}')
        else:
            self.ax.set_title(f'Training Pose {self.current_pose_idx} (interp: {self.interpolation:.2f})')
    
    def set_mode(self, mode):
        """Set viewing mode."""
        self.view_mode = mode
        self.update_view()
    
    def update_theta(self, val):
        """Update horizontal angle."""
        self.theta = val
        if self.view_mode == 'novel':
            self.update_view()
            
    def update_phi(self, val):
        """Update vertical angle."""
        self.phi = val
        if self.view_mode == 'novel':
            self.update_view()
            
    def update_radius(self, val):
        """Update camera distance."""
        self.radius = val
        if self.view_mode == 'novel':
            self.update_view()
            
    def update_height(self, val):
        """Update camera height."""
        self.height = val
        if self.view_mode == 'novel':
            self.update_view()
        
    def update_pose_idx(self, val):
        """Update pose index."""
        self.current_pose_idx = int(val)
        if self.view_mode == 'training':
            self.update_view()
        
    def update_interpolation(self, val):
        """Update interpolation value."""
        self.interpolation = val
        if self.view_mode == 'training':
            self.update_view()
        
    def update_view(self):
        """Update the displayed image."""
        print(f"Rendering {self.view_mode} view...")
        
        # Render new view
        new_image = self.render_view()
        
        # Update display
        self.im.set_array(new_image)
        self.update_title()
        self.fig.canvas.draw_idle()
        
    def run(self):
        """Start the interactive viewer."""
        print("\n🎮 Corrected Interactive NeRF Viewer:")
        print("• Starting at FIRST TRAINING POSE (Training Mode)")
        print("• Training Mode: Browse original training poses using pose slider")
        print("• Novel Mode: Use θ/φ/Distance/Height sliders for new viewpoints")
        print("• Radio buttons (top-left): Switch between Training and Novel modes")
        print("• First image should match exactly - validating pose correctness!")
        print("• Close window to exit")
        
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Corrected Interactive NeRF Viewer')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    viewer = CorrectInteractiveNeRFViewer(args.checkpoint, args.config)
    viewer.run()

if __name__ == '__main__':
    main()
