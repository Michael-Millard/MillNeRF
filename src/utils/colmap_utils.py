"""
COLMAP integration utilities for camera poses.
"""

import os, subprocess, json
import numpy as np
from pathlib import Path

def run_colmap_sfm(images_dir, output_dir, camera_model='OPENCV'):
    """
    Run COLMAP Structure-from-Motion pipeline with robust settings.
    """
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    database_path = output_path / "database.db"
    sparse_path = output_path / "sparse"
    sparse_path.mkdir(exist_ok=True)
    
    print("Running COLMAP feature extraction with robust settings...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_path),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_image_size", "3200",
        "--SiftExtraction.max_num_features", "8192",
        "--SiftExtraction.first_octave", "-1",
        "--SiftExtraction.num_octaves", "4",
        "--SiftExtraction.octave_resolution", "3"
    ], check=True)
    
    print("Running COLMAP feature matching with relaxed settings...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.max_ratio", "0.8",
        "--SiftMatching.max_distance", "0.7",
        "--SiftMatching.cross_check", "1",
        "--SiftMatching.max_num_matches", "32768"
    ], check=True)
    
    print("Running COLMAP sparse reconstruction with VERY relaxed constraints...")
    try:
        subprocess.run([
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_path),
            "--output_path", str(sparse_path),
            # Very relaxed initialization
            "--Mapper.init_min_num_inliers", "15",          # Default: 100
            "--Mapper.init_max_forward_motion", "0.95",     # Default: 0.95
            "--Mapper.init_min_tri_angle", "4",             # Default: 16
            # Relaxed pose estimation
            "--Mapper.abs_pose_min_num_inliers", "10",      # Default: 30
            "--Mapper.abs_pose_min_inlier_ratio", "0.10",   # Default: 0.25
            # Relaxed model constraints
            "--Mapper.min_model_size", "3",                 # Default: 10
            "--Mapper.max_model_overlap", "50",             # Default: 20
            # More lenient triangulation
            "--Mapper.tri_min_angle", "1.5",               # Default: 1.5
            "--Mapper.tri_ignore_two_view_tracks", "0",    # Default: 1
            # Bundle adjustment settings
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "1",
            # Increase attempts
            "--Mapper.max_num_models", "5",                 # Try multiple models
            "--Mapper.init_max_reg_trials", "5"             # More initialization attempts
        ], check=True)
        
    except subprocess.CalledProcessError:
        print("Standard reconstruction failed. Trying EVEN MORE relaxed settings...")
        try:
            subprocess.run([
                "colmap", "mapper", 
                "--database_path", str(database_path),
                "--image_path", str(images_path),
                "--output_path", str(sparse_path),
                # Ultra-relaxed settings
                "--Mapper.init_min_num_inliers", "8",
                "--Mapper.abs_pose_min_num_inliers", "5",
                "--Mapper.abs_pose_min_inlier_ratio", "0.05",
                "--Mapper.min_model_size", "2",
                "--Mapper.init_min_tri_angle", "2",
                "--Mapper.tri_min_angle", "1.0",
                "--Mapper.max_num_models", "10"
            ], check=True)
        except subprocess.CalledProcessError:
            print("❌ COLMAP reconstruction failed even with ultra-relaxed settings")
            return None
    
    # Check if reconstruction was successful
    recon_dirs = list(sparse_path.glob("*"))
    recon_dirs = [d for d in recon_dirs if d.is_dir()]
    
    if not recon_dirs:
        print("❌ No reconstruction directories found")
        return None
    
    # Find the largest reconstruction
    best_recon = None
    max_images = 0
    
    for recon_dir in recon_dirs:
        images_file = recon_dir / "images.txt"
        if images_file.exists():
            with open(images_file, 'r') as f:
                lines = [l for l in f if not l.startswith('#') and l.strip()]
                num_images = len(lines) // 2  # Every other line is an image
                
            print(f"Reconstruction {recon_dir.name}: {num_images} images")
            if num_images > max_images:
                max_images = num_images
                best_recon = recon_dir
    
    if best_recon is None or max_images < 3:
        print(f"❌ Best reconstruction only has {max_images} images (need at least 3)")
        return None
    
    print(f"✅ COLMAP reconstruction successful: {max_images} images in {best_recon}")
    return best_recon

def colmap_to_nerf_format(colmap_dir, output_file, scale_factor=1.0):
    """
    Convert COLMAP output to NeRF format (transforms.json).
    
    Args:
        colmap_dir: Directory containing COLMAP sparse reconstruction
        output_file: Output JSON file path
        scale_factor: Scale factor for coordinates
    """
    import sqlite3
    
    # Find the reconstruction directory (usually '0')
    sparse_dir = Path(colmap_dir)
    recon_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
    
    if not recon_dirs:
        raise ValueError("No reconstruction found in COLMAP output")
    
    recon_dir = recon_dirs[0]  # Use first reconstruction
    
    # Read COLMAP files
    cameras = read_colmap_cameras(recon_dir / "cameras.txt")
    images = read_colmap_images(recon_dir / "images.txt")
    
    # Convert to NeRF format
    transforms = {
        "frames": []
    }
    
    # Get camera parameters (assume single camera)
    camera = list(cameras.values())[0]
    
    if camera['model'] == 'PINHOLE':
        fx, fy, cx, cy = camera['params']
        transforms["fl_x"] = fx
        transforms["fl_y"] = fy
        transforms["cx"] = cx
        transforms["cy"] = cy
        transforms["w"] = camera['width']
        transforms["h"] = camera['height']
        transforms["camera_angle_x"] = 2 * np.arctan(camera['width'] / (2 * fx))
        
    elif camera['model'] == 'SIMPLE_PINHOLE':
        fx, cx, cy = camera['params']
        transforms["fl_x"] = fx
        transforms["fl_y"] = fx  # Simple pinhole assumes fx == fy
        transforms["cx"] = cx
        transforms["cy"] = cy
        transforms["w"] = camera['width']
        transforms["h"] = camera['height']
        transforms["camera_angle_x"] = 2 * np.arctan(camera['width'] / (2 * fx))
        
    elif camera['model'] == 'OPENCV':
        fx, fy, cx, cy, k1, k2, p1, p2 = camera['params']
        transforms["fl_x"] = fx
        transforms["fl_y"] = fy  
        transforms["cx"] = cx
        transforms["cy"] = cy
        transforms["w"] = camera['width']
        transforms["h"] = camera['height']
        transforms["camera_angle_x"] = 2 * np.arctan(camera['width'] / (2 * fx))
        transforms["k1"] = k1
        transforms["k2"] = k2
        transforms["p1"] = p1
        transforms["p2"] = p2
    
    # Process each image
    for img_id, img_data in images.items():
        # COLMAP uses quaternion + translation
        quat = img_data['quat']  # [qw, qx, qy, qz]
        trans = img_data['trans']
        
        # Convert quaternion to rotation matrix
        R = quat_to_rotation_matrix(quat)
        
        # COLMAP uses world-to-camera, NeRF uses camera-to-world
        # So we need to invert: [R|t] -> [R^T | -R^T*t]
        R_inv = R.T
        t_inv = -R_inv @ trans
        
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_inv
        transform_matrix[:3, 3] = t_inv * scale_factor
        
        # Apply coordinate system conversion (COLMAP -> NeRF)
        # NeRF: +X right, +Y up, +Z backward
        # COLMAP: +X right, +Y down, +Z forward
        transform_matrix[1:3] *= -1  # Flip Y and Z
        
        frame = {
            "file_path": f"images/{img_data['name']}",
            "transform_matrix": transform_matrix.tolist()
        }
        transforms["frames"].append(frame)
    
    # Save transforms.json
    with open(output_file, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"Converted COLMAP poses to {output_file}")
    print(f"Found {len(transforms['frames'])} images")
    
    return transforms


def read_colmap_cameras(cameras_file):
    """Read COLMAP cameras.txt file."""
    cameras = {}
    
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
                
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    
    return cameras


def read_colmap_images(images_file):
    """Read COLMAP images.txt file."""
    images = {}
    
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    # Process every other line (image lines, skip point lines)
    for i in range(0, len(lines), 2):
        line = lines[i]
        if line.startswith('#') or not line.strip():
            continue
            
        parts = line.strip().split()
        image_id = int(parts[0])
        
        # Quaternion [qw, qx, qy, qz]
        quat = np.array([float(parts[1]), float(parts[2]), 
                        float(parts[3]), float(parts[4])])
        
        # Translation [tx, ty, tz]  
        trans = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        
        camera_id = int(parts[8])
        name = parts[9]
        
        images[image_id] = {
            'quat': quat,
            'trans': trans,
            'camera_id': camera_id,
            'name': name
        }
    
    return images


def quat_to_rotation_matrix(quat):
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = quat
    
    # Normalize quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    
    return R