"""
Utility functions for NeRF implementation.
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional
import json
import os


def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ray origins and directions for all pixels in an image.
    
    Args:
        H: Image height
        W: Image width  
        focal: Focal length
        c2w: Camera-to-world transformation matrix [3, 4]
        
    Returns:
        rays_o: Ray origins [H, W, 3]
        rays_d: Ray directions [H, W, 3]
    """
    # Create meshgrid of pixel coordinates
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i = i.t()  # [H, W]
    j = j.t()  # [H, W]
    
    # Convert to normalized device coordinates
    dirs = torch.stack([(i - W * 0.5) / focal,
                       -(j - H * 0.5) / focal,
                       -torch.ones_like(i)], -1)  # [H, W, 3]
    
    # Transform ray directions from camera space to world space
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
    
    # Sample linearly in disparity space
    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    
    z_vals = z_vals.expand([N_rays, N_samples])
    
    if perturb:
        # Get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        
        # Stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
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
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)], -1)
    
    # Multiply each distance by the norm of its ray direction
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # Extract RGB and density
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    
    # Add noise to raw densities for regularization
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
    
    # Compute alpha (opacity) from density
    alpha = 1. - torch.exp(-torch.relu(raw[..., 3] + noise) * dists)  # [N_rays, N_samples]
    
    # Compute transmittance (accumulated transparency)
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device),
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
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to 8-bit numpy array."""
    return (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)
