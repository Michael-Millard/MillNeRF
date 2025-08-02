"""
NeRF model implementation using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class NeRFModel(nn.Module):
    """
    Neural Radiance Field model.
    
    This implements the core NeRF architecture that maps 3D coordinates and viewing directions
    to RGB colors and volume densities.
    """
    
    def __init__(self, 
                 pos_enc_levels: int = 10,
                 dir_enc_levels: int = 4,
                 hidden_dim: int = 256,
                 hidden_layers: int = 8,
                 skip_connections: list = [4]):
        """
        Initialize NeRF model.
        
        Args:
            pos_enc_levels: Number of positional encoding levels for positions
            dir_enc_levels: Number of positional encoding levels for directions  
            hidden_dim: Hidden layer dimension
            hidden_layers: Number of hidden layers
            skip_connections: List of layer indices to add skip connections
        """
        super(NeRFModel, self).__init__()
        
        self.pos_enc_levels = pos_enc_levels
        self.dir_enc_levels = dir_enc_levels
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.skip_connections = skip_connections
        
        # Calculate input dimensions after positional encoding
        # Position: 3 coords -> 3 + 3*2*pos_enc_levels  
        # Direction: 3 coords -> 3 + 3*2*dir_enc_levels
        self.pos_input_dim = 3 + 3 * 2 * pos_enc_levels
        self.dir_input_dim = 3 + 3 * 2 * dir_enc_levels
        
        # Position encoding network (outputs density + feature vector)
        pos_layers = []
        input_dim = self.pos_input_dim
        
        for i in range(hidden_layers):
            if i in skip_connections:
                # Add skip connection from input
                input_dim = hidden_dim + self.pos_input_dim
            
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim)
            else:
                layer = nn.Linear(input_dim, hidden_dim)
            
            pos_layers.append(layer)
            input_dim = hidden_dim
        
        self.pos_layers = nn.ModuleList(pos_layers)
        
        # Density output layer
        self.density_layer = nn.Linear(hidden_dim, 1)
        
        # Feature layer (before direction-dependent color prediction)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Direction-dependent color network
        self.color_layer1 = nn.Linear(hidden_dim + self.dir_input_dim, hidden_dim // 2)
        self.color_layer2 = nn.Linear(hidden_dim // 2, 3)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def positional_encoding(self, x: torch.Tensor, L: int) -> torch.Tensor:
        """
        Apply positional encoding to input coordinates.
        
        Args:
            x: Input coordinates [..., D]
            L: Number of frequency levels
            
        Returns:
            Encoded coordinates
        """
        if L == 0:
            return x
            
        # Original coordinates
        encoded = [x]
        
        # Sinusoidal encodings at different frequencies
        for i in range(L):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * torch.pi * x))
            encoded.append(torch.cos(freq * torch.pi * x))
        
        return torch.cat(encoded, -1)
    
    def forward(self, pts: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through NeRF model.
        
        Args:
            pts: 3D points [..., 3]
            dirs: Viewing directions [..., 3]
            
        Returns:
            RGB colors and densities [..., 4]
        """
        # Positional encoding
        pts_encoded = self.positional_encoding(pts, self.pos_enc_levels)
        dirs_encoded = self.positional_encoding(dirs, self.dir_enc_levels)
        
        # Forward through position network
        x = pts_encoded
        input_pts = x
        
        for i, layer in enumerate(self.pos_layers):
            if i in self.skip_connections:
                # Add skip connection
                x = torch.cat([x, input_pts], -1)
            
            x = F.relu(layer(x))
        
        # Output density
        density = self.density_layer(x)
        
        # Extract features for color prediction
        features = self.feature_layer(x)
        
        # Combine features with viewing directions
        color_input = torch.cat([features, dirs_encoded], -1)
        
        # Predict RGB color
        color = F.relu(self.color_layer1(color_input))
        color = torch.sigmoid(self.color_layer2(color))
        
        # Combine density and color
        outputs = torch.cat([color, density], -1)
        
        return outputs


class HierarchicalNeRF(nn.Module):
    """
    Hierarchical NeRF with coarse and fine networks.
    
    This implements the two-stage sampling approach from the original NeRF paper.
    """
    
    def __init__(self, 
                 pos_enc_levels: int = 10,
                 dir_enc_levels: int = 4,
                 coarse_hidden_dim: int = 256,
                 coarse_hidden_layers: int = 8,
                 fine_hidden_dim: int = 256,
                 fine_hidden_layers: int = 8,
                 skip_connections: list = [4]):
        """
        Initialize hierarchical NeRF.
        
        Args:
            pos_enc_levels: Number of positional encoding levels
            dir_enc_levels: Number of direction encoding levels
            coarse_hidden_dim: Hidden dimension for coarse network
            coarse_hidden_layers: Number of layers for coarse network
            fine_hidden_dim: Hidden dimension for fine network  
            fine_hidden_layers: Number of layers for fine network
            skip_connections: Skip connection layer indices
        """
        super(HierarchicalNeRF, self).__init__()
        
        # Coarse network
        self.coarse_net = NeRFModel(
            pos_enc_levels=pos_enc_levels,
            dir_enc_levels=dir_enc_levels,
            hidden_dim=coarse_hidden_dim,
            hidden_layers=coarse_hidden_layers,
            skip_connections=skip_connections
        )
        
        # Fine network
        self.fine_net = NeRFModel(
            pos_enc_levels=pos_enc_levels,
            dir_enc_levels=dir_enc_levels,
            hidden_dim=fine_hidden_dim,
            hidden_layers=fine_hidden_layers,
            skip_connections=skip_connections
        )
    
    def forward(self, pts_coarse: torch.Tensor, dirs_coarse: torch.Tensor,
                pts_fine: Optional[torch.Tensor] = None, 
                dirs_fine: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through hierarchical NeRF.
        
        Args:
            pts_coarse: Coarse sample points [..., 3]
            dirs_coarse: Viewing directions for coarse samples [..., 3]
            pts_fine: Fine sample points [..., 3] (optional)
            dirs_fine: Viewing directions for fine samples [..., 3] (optional)
            
        Returns:
            Coarse outputs and fine outputs (if provided)
        """
        # Coarse network forward pass
        coarse_outputs = self.coarse_net(pts_coarse, dirs_coarse)
        
        # Fine network forward pass (if fine samples provided)
        fine_outputs = None
        if pts_fine is not None and dirs_fine is not None:
            fine_outputs = self.fine_net(pts_fine, dirs_fine)
        
        return coarse_outputs, fine_outputs


def create_nerf_model(config: dict) -> HierarchicalNeRF:
    """
    Create NeRF model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized NeRF model
    """
    model_config = config.get('model', {})
    
    return HierarchicalNeRF(
        pos_enc_levels=model_config.get('pos_enc_levels', 10),
        dir_enc_levels=model_config.get('dir_enc_levels', 4),
        coarse_hidden_dim=model_config.get('coarse_net', {}).get('hidden_dim', 256),
        coarse_hidden_layers=model_config.get('coarse_net', {}).get('hidden_layers', 8),
        fine_hidden_dim=model_config.get('fine_net', {}).get('hidden_dim', 256),
        fine_hidden_layers=model_config.get('fine_net', {}).get('hidden_layers', 8),
        skip_connections=model_config.get('coarse_net', {}).get('skip_connections', [4])
    )
