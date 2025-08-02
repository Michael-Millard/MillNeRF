"""
Camera utilities for handling poses, intrinsics, and coordinate transformations.
"""

import numpy as np
import torch
import json
import os
from typing import Tuple, List, Dict, Any
import cv2


def load_poses_from_json(json_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load camera poses from a JSON file (NeRF format).
    
    Args:
        json_path: Path to transforms.json file
        
    Returns:
        poses: Camera poses [N, 4, 4]
        images: List of image filenames
        focal: Focal length
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    poses = []
    images = []
    
    for frame in data['frames']:
        poses.append(np.array(frame['transform_matrix']))
        images.append(frame['file_path'])
    
    poses = np.array(poses)
    
    # Extract focal length
    if 'fl_x' in data:
        focal = data['fl_x']
    elif 'camera_angle_x' in data:
        focal = 0.5 * data['w'] / np.tan(0.5 * data['camera_angle_x'])
    else:
        raise ValueError("No focal length information found in JSON")
    
    return poses, images, focal


def poses_avg(poses: np.ndarray) -> np.ndarray:
    """
    Compute average pose for centering camera positions.
    
    Args:
        poses: Camera poses [N, 4, 4]
        
    Returns:
        Average pose [4, 4]
    """
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), np.array([[0, 0, 0, 1.]])], 0)
    return c2w


def viewmatrix(z: np.ndarray, up: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Construct a camera view matrix.
    
    Args:
        z: Forward direction (negative z-axis)
        up: Up direction
        pos: Camera position
        
    Returns:
        View matrix [3, 4]
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return x / np.linalg.norm(x)


def recenter_poses(poses: np.ndarray) -> np.ndarray:
    """
    Recenter poses around the average pose.
    
    Args:
        poses: Camera poses [N, 4, 4]
        
    Returns:
        Recentered poses [N, 4, 4]
    """
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses: np.ndarray, bds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert poses for spherical coordinate system (360-degree scenes).
    
    Args:
        poses: Camera poses [N, 4, 4]  
        bds: Scene bounds [N, 2]
        
    Returns:
        Spherified poses and bounds
    """
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)
    
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]
    
    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist
    
    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    
    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])
    
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    
    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []
    
    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])
        
        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec0, vec2))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)
        
        new_poses.append(p)
    
    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)
    
    return poses_reset, new_poses, bds


def generate_render_path(c2w: np.ndarray, up: np.ndarray, rads: np.ndarray, 
                        focal: float, zdelta: float, zrate: float, N_rots: int = 2, N: int = 120) -> np.ndarray:
    """
    Generate a smooth camera path for rendering videos.
    
    Args:
        c2w: Base camera pose
        up: Up vector
        rads: Radii for camera path
        focal: Focal length
        zdelta: Z variation
        zrate: Rate of Z change
        N_rots: Number of rotations
        N: Number of frames
        
    Returns:
        Render poses [N, 4, 4]
    """
    render_poses = []
    
    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), np.array([[0, 0, 0, 1.]])], 0))
    
    return render_poses


def get_camera_intrinsics(H: int, W: int, focal: float) -> np.ndarray:
    """
    Get camera intrinsic matrix.
    
    Args:
        H: Image height
        W: Image width
        focal: Focal length
        
    Returns:
        Intrinsic matrix [3, 3]
    """
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    return K


def load_colmap_data(basedir: str, factor: int = 8) -> Tuple[np.ndarray, np.ndarray, List[str], int, int]:
    """
    Load COLMAP data (poses, images, intrinsics).
    
    Args:
        basedir: Base directory containing COLMAP output
        factor: Downsampling factor
        
    Returns:
        poses, bds, imgfiles, H, W
    """
    # This is a placeholder - you would implement actual COLMAP loading here
    # For now, return dummy data
    poses = np.eye(4)[None].repeat(10, axis=0)
    bds = np.array([[1.0, 10.0]] * 10)
    imgfiles = [f"image_{i:03d}.jpg" for i in range(10)]
    H, W = 480 // factor, 640 // factor
    
    return poses, bds, imgfiles, H, W
