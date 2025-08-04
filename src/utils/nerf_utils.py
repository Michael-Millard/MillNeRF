"""
Utility functions for NeRF implementation.
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional
import json
import os

def get_rays(H, W, focal, c2w, device=None):
    """
    Get ray origins and directions from camera pose.
    
    Args:
        H, W: Image height and width
        focal: Focal length
        c2w: Camera-to-world transformation matrix [4, 4]
        device: Device to put tensors on (if None, infer from c2w)
    
    Returns:
        rays_o: Ray origins [H, W, 3]
        rays_d: Ray directions [H, W, 3]
    """
    # Ensure c2w is a tensor and get its device
    if not isinstance(c2w, torch.Tensor):
        c2w = torch.tensor(c2w, dtype=torch.float32)
    
    if device is None:
        device = c2w.device
    
    # Move c2w to the specified device
    c2w = c2w.to(device)
    
    # Create coordinate grids
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=device),
        torch.linspace(0, H-1, H, device=device),
        indexing='ij'
    )
    i = i.t()  # Transpose to get [H, W]
    j = j.t()
    
    # Convert pixel coordinates to camera coordinates
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,  # Negative for camera convention
        -torch.ones_like(i, device=device)
    ], -1)  # [H, W, 3]
    
    # Transform ray directions from camera to world coordinates
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # [H, W, 3]
    
    # Ray origins are just the camera center
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # [H, W, 3]
    
    return rays_o, rays_d


def sample_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, 
                near: float, far: float, N_samples: int, 
                perturb: bool = False) -> torch.Tensor:
    """
    Sample points along rays.
    
    Args:
        rays_o: Ray origins [N_rays, 3]
        rays_d: Ray directions [N_rays, 3]
        near: Near bound
        far: Far bound
        N_samples: Number of samples per ray
        perturb: Whether to add noise to sample positions
        
    Returns:
        pts: Sampled 3D points [N_rays, N_samples, 3]
    """
    N_rays = rays_o.shape[0]
    device = rays_o.device
    
    # Sample linearly in disparity space
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    
    z_vals = z_vals.expand([N_rays, N_samples])
    
    if perturb:
        # Get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        
        # Stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand
    
    # Compute 3D points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    
    return pts, z_vals


def positional_encoding(x: torch.Tensor, L: int) -> torch.Tensor:
    """
    Apply positional encoding to input coordinates.
    
    Args:
        x: Input coordinates [..., D]
        L: Number of frequency levels
        
    Returns:
        Encoded coordinates [..., D * (2*L + 1)]
    """
    if L == 0:
        return x
        
    # Original coordinates
    encoded = [x]
    
    # Sinusoidal encodings
    for i in range(L):
        freq = 2.**i
        encoded.append(torch.sin(freq * x))
        encoded.append(torch.cos(freq * x))
    
    return torch.cat(encoded, -1)


def raw2outputs(raw: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor,
                raw_noise_std: float = 0., white_bkgd: bool = False) -> dict:
    """
    Transform model's raw output to rendered RGB and depth.
    
    Args:
        raw: Raw network output [N_rays, N_samples, 4] (RGB + density)
        z_vals: Sample distances along rays [N_rays, N_samples]
        rays_d: Ray directions [N_rays, 3]
        raw_noise_std: Standard deviation of noise added to densities
        white_bkgd: If True, assume white background
        
    Returns:
        Dictionary containing rgb_map, disp_map, acc_map, weights, depth_map
    """
    device = raw.device
    
    # Compute distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1)
    
    # Multiply each distance by the norm of its ray direction
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # Extract RGB and density
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    
    # Add noise to raw densities for regularization
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std
    
    # Compute alpha (opacity) from density
    alpha = 1. - torch.exp(-torch.relu(raw[..., 3] + noise) * dists)  # [N_rays, N_samples]
    
    # Compute transmittance (accumulated transparency)
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device),
                                           1. - alpha + 1e-10], -1), -1)[:, :-1]
    
    # Compute weights for volume rendering
    weights = alpha * transmittance  # [N_rays, N_samples]
    
    # Compute final RGB
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    
    # Compute depth map
    depth_map = torch.sum(weights * z_vals, -1)
    
    # Compute disparity map
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    
    # Compute accumulated opacity (alpha composite)
    acc_map = torch.sum(weights, -1)
    
    # Add white background if specified
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    
    return {
        'rgb_map': rgb_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'weights': weights,
        'depth_map': depth_map
    }


def img2mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss between images."""
    return torch.mean((x - y) ** 2)


def mse2psnr(x: torch.Tensor) -> torch.Tensor:
    """Convert MSE to PSNR."""
    return -10. * torch.log(x) / torch.log(torch.tensor(10.0, device=x.device))


def to8b(x) -> np.ndarray:
    """Convert tensor or numpy array to 8-bit numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)
