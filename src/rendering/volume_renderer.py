"""
Volume rendering implementation for NeRF.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from ..utils.nerf_utils import sample_rays, raw2outputs, positional_encoding


class VolumeRenderer:
    """
    Volume renderer for NeRF that handles ray sampling and rendering.
    """
    
    def __init__(self, config: dict, device: torch.device):
        """
        Initialize volume renderer.
        
        Args:
            config: Configuration dictionary
            device: Device to run on
        """
        self.config = config
        self.device = device
        
        # Sampling configuration
        sampling_config = config.get('sampling', {})
        self.num_coarse_samples = sampling_config.get('num_coarse_samples', 64)
        self.num_fine_samples = sampling_config.get('num_fine_samples', 128)
        self.perturb = sampling_config.get('perturb', True)
        self.raw_noise_std = sampling_config.get('raw_noise_std', 1.0)
        
        # Data configuration
        data_config = config.get('data', {})
        self.near = data_config.get('near', 0.1)
        self.far = data_config.get('far', 10.0)
        self.white_background = data_config.get('white_background', False)
        
        # Rendering configuration
        rendering_config = config.get('rendering', {})
        self.chunk_size = rendering_config.get('chunk_size', 32768)
    
    def sample_coarse_points(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample coarse points along rays.
        
        Args:
            rays_o: Ray origins [N_rays, 3]
            rays_d: Ray directions [N_rays, 3]
            
        Returns:
            pts: Sampled points [N_rays, N_samples, 3]
            z_vals: Sample distances [N_rays, N_samples]
        """
        N_rays = rays_o.shape[0]
        
        # Sample points uniformly in depth
        t_vals = torch.linspace(0., 1., steps=self.num_coarse_samples, device=self.device)
        z_vals = self.near * (1. - t_vals) + self.far * t_vals
        z_vals = z_vals.expand([N_rays, self.num_coarse_samples])
        
        if self.perturb and self.training:
            # Add noise for regularization during training
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            
            # Stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=self.device)
            z_vals = lower + (upper - lower) * t_rand
        
        # Calculate 3D points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        
        return pts, z_vals
    
    def sample_fine_points(self, rays_o: torch.Tensor, rays_d: torch.Tensor, 
                          z_vals_coarse: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample fine points using importance sampling based on coarse weights.
        
        Args:
            rays_o: Ray origins [N_rays, 3]
            rays_d: Ray directions [N_rays, 3]
            z_vals_coarse: Coarse sample distances [N_rays, N_coarse]
            weights: Coarse rendering weights [N_rays, N_coarse]
            
        Returns:
            pts: Fine sample points [N_rays, N_fine, 3]
            z_vals: Combined sample distances [N_rays, N_coarse + N_fine]
        """
        N_rays = rays_o.shape[0]
        
        # Importance sampling based on coarse weights
        z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
        weights_mid = weights[..., 1:-1]  # Remove first and last weight
        
        # Add small epsilon to prevent NaN
        weights_mid = weights_mid + 1e-5
        
        # Create PDF
        pdf = weights_mid / torch.sum(weights_mid, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        
        # Sample from CDF
        u = torch.rand([N_rays, self.num_fine_samples], device=self.device)
        u = u.contiguous()
        
        # Invert CDF
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # [N_rays, N_fine, 2]
        
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(z_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)
        
        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        z_vals_fine = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        # Combine coarse and fine samples
        z_vals_combined, _ = torch.sort(torch.cat([z_vals_coarse, z_vals_fine], -1), -1)
        
        # Calculate fine sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
        
        return pts, z_vals_combined
    
    def render_rays(self, model, rays_o: torch.Tensor, rays_d: torch.Tensor, 
                   return_raw: bool = False) -> Dict[str, torch.Tensor]:
        """
        Render rays using the NeRF model.
        
        Args:
            model: NeRF model
            rays_o: Ray origins [N_rays, 3]
            rays_d: Ray directions [N_rays, 3]
            return_raw: Whether to return raw network outputs
            
        Returns:
            Dictionary containing rendered outputs
        """
        N_rays = rays_o.shape[0]
        results = {}
        
        # Sample coarse points
        pts_coarse, z_vals_coarse = self.sample_coarse_points(rays_o, rays_d)
        
        # Prepare directions for coarse network
        dirs_coarse = rays_d[..., None, :].expand(pts_coarse.shape)
        
        # Flatten for network forward pass
        pts_coarse_flat = pts_coarse.reshape(-1, 3)
        dirs_coarse_flat = dirs_coarse.reshape(-1, 3)
        
        # Forward pass through coarse network
        raw_coarse, _ = model(pts_coarse_flat, dirs_coarse_flat)
        raw_coarse = raw_coarse.reshape(N_rays, self.num_coarse_samples, 4)
        
        # Volume rendering for coarse samples
        outputs_coarse = raw2outputs(raw_coarse, z_vals_coarse, rays_d, 
                                   self.raw_noise_std if self.training else 0.,
                                   self.white_background)
        
        results['rgb_coarse'] = outputs_coarse['rgb_map']
        results['depth_coarse'] = outputs_coarse['depth_map']
        results['acc_coarse'] = outputs_coarse['acc_map']
        
        if return_raw:
            results['raw_coarse'] = raw_coarse
        
        # Fine sampling and rendering
        if self.num_fine_samples > 0:
            # Sample fine points using importance sampling
            pts_fine, z_vals_fine = self.sample_fine_points(
                rays_o, rays_d, z_vals_coarse, outputs_coarse['weights'].detach()
            )
            
            # Prepare directions for fine network
            dirs_fine = rays_d[..., None, :].expand(pts_fine.shape)
            
            # Flatten for network forward pass
            pts_fine_flat = pts_fine.reshape(-1, 3)
            dirs_fine_flat = dirs_fine.reshape(-1, 3)
            
            # Forward pass through fine network
            _, raw_fine = model(pts_coarse_flat, dirs_coarse_flat, pts_fine_flat, dirs_fine_flat)
            raw_fine = raw_fine.reshape(N_rays, self.num_coarse_samples + self.num_fine_samples, 4)
            
            # Volume rendering for fine samples
            outputs_fine = raw2outputs(raw_fine, z_vals_fine, rays_d,
                                     self.raw_noise_std if self.training else 0.,
                                     self.white_background)
            
            results['rgb_fine'] = outputs_fine['rgb_map']
            results['depth_fine'] = outputs_fine['depth_map']
            results['acc_fine'] = outputs_fine['acc_map']
            
            if return_raw:
                results['raw_fine'] = raw_fine
        
        return results
    
    def render_image(self, model, H: int, W: int, focal: float, c2w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Render a full image using the NeRF model.
        
        Args:
            model: NeRF model
            H: Image height
            W: Image width
            focal: Focal length
            c2w: Camera-to-world matrix [4, 4]
            
        Returns:
            Dictionary containing rendered image and depth
        """
        # Generate all rays for the image
        from ..utils.nerf_utils import get_rays
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.reshape(-1, 3)  # [H*W, 3]
        rays_d = rays_d.reshape(-1, 3)  # [H*W, 3]
        
        # Render in chunks to save memory
        all_results = {}
        
        for i in range(0, rays_o.shape[0], self.chunk_size):
            chunk_rays_o = rays_o[i:i + self.chunk_size]
            chunk_rays_d = rays_d[i:i + self.chunk_size]
            
            chunk_results = self.render_rays(model, chunk_rays_o, chunk_rays_d)
            
            # Store results
            for key, val in chunk_results.items():
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append(val)
        
        # Concatenate all chunks
        for key in all_results:
            all_results[key] = torch.cat(all_results[key], 0)
        
        # Reshape to image format
        results = {}
        for key, val in all_results.items():
            if 'rgb' in key:
                results[key] = val.reshape(H, W, 3)
            else:
                results[key] = val.reshape(H, W)
        
        return results
    
    def __init__(self, config: dict, device: torch.device):
        """
        Initialize volume renderer.
        
        Args:
            config: Configuration dictionary
            device: Device to run on
        """
        self.config = config
        self.device = device
        self.training = True  # Training mode flag
        
        # Sampling configuration
        sampling_config = config.get('sampling', {})
        self.num_coarse_samples = sampling_config.get('num_coarse_samples', 64)
        self.num_fine_samples = sampling_config.get('num_fine_samples', 128)
        self.perturb = sampling_config.get('perturb', True)
        self.raw_noise_std = sampling_config.get('raw_noise_std', 1.0)
        
        # Data configuration
        data_config = config.get('data', {})
        self.near = data_config.get('near', 0.1)
        self.far = data_config.get('far', 10.0)
        self.white_background = data_config.get('white_background', False)
        
        # Rendering configuration
        rendering_config = config.get('rendering', {})
        self.chunk_size = rendering_config.get('chunk_size', 32768)


def create_volume_renderer(config: dict, device: torch.device) -> VolumeRenderer:
    """
    Create volume renderer from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to run on
        
    Returns:
        Volume renderer
    """
    return VolumeRenderer(config, device)
