"""
Training script for NeRF model.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any
import json

from ..models.nerf import create_nerf_model
from ..data.dataset import create_dataset, create_dataloader
from ..rendering.volume_renderer import create_volume_renderer
from ..utils.nerf_utils import img2mse, mse2psnr, to8b


class NeRFTrainer:
    """
    Trainer class for NeRF model.
    """
    
    def __init__(self, config: dict, device: torch.device):
        """
        Initialize NeRF trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to train on
        """
        self.config = config
        self.device = device
        
        # Training configuration
        training_config = config.get('training', {})
        self.max_iterations = training_config.get('max_iterations', 1000000)
        self.learning_rate = training_config.get('learning_rate', 5e-4)
        self.lr_decay = training_config.get('lr_decay', 0.1)
        self.lr_decay_steps = training_config.get('lr_decay_steps', [250000, 500000])
        self.coarse_loss_weight = training_config.get('coarse_loss_weight', 1.0)
        self.fine_loss_weight = training_config.get('fine_loss_weight', 1.0)
        
        # Logging configuration
        logging_config = config.get('logging', {})
        self.log_every = logging_config.get('log_every', 1000)
        self.validate_every = training_config.get('validate_every', 10000)
        self.save_checkpoint_every = training_config.get('save_checkpoint_every', 50000)
        self.render_test_every = logging_config.get('render_test_every', 25000)
        
        # Create directories
        self.log_dir = logging_config.get('log_dir', 'build/logs')
        self.checkpoint_dir = logging_config.get('checkpoint_dir', 'build/checkpoints')
        self.render_dir = logging_config.get('render_dir', 'build/renders')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.render_dir, exist_ok=True)
        
        # Initialize model, data, and optimizer
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_renderer()
        
        # Training state
        self.iteration = 0
        self.train_losses = []
        self.val_psnrs = []
        
    def _setup_model(self):
        """Initialize the NeRF model."""
        print("Setting up NeRF model...")
        self.model = create_nerf_model(self.config).to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params:,} trainable parameters")
        
    def _setup_data(self):
        """Initialize datasets and data loaders."""
        print("Setting up datasets...")
        
        # Training dataset
        self.train_dataset = create_dataset(self.config, 'train', self.device)
        self.train_loader = create_dataloader(self.train_dataset, self.config, 'train')
        
        # Validation dataset (use a subset of training images)
        # For simplicity, we'll use the same dataset but different sampling
        self.val_dataset = create_dataset(self.config, 'val', self.device)
        
        print(f"Training dataset: {len(self.train_dataset)} rays")
        print(f"Validation dataset: {len(self.val_dataset)} images")
        
    def _setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        def lr_lambda(step):
            factor = 1.0
            for decay_step in self.lr_decay_steps:
                if step >= decay_step:
                    factor *= self.lr_decay
            return factor
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def _setup_renderer(self):
        """Initialize volume renderer."""
        self.renderer = create_volume_renderer(self.config, self.device)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch containing rays and target colors
            
        Returns:
            Dictionary containing loss values
        """
        self.model.train()
        self.renderer.training = True
        
        # Extract batch data
        rays_o = batch['rays_o']  # [batch_size, 3]
        rays_d = batch['rays_d']  # [batch_size, 3]
        target_rgb = batch['rgb']  # [batch_size, 3]
        
        # Render rays
        outputs = self.renderer.render_rays(self.model, rays_o, rays_d)
        
        # Compute losses
        loss_coarse = img2mse(outputs['rgb_coarse'], target_rgb)
        
        total_loss = self.coarse_loss_weight * loss_coarse
        losses = {'loss_coarse': loss_coarse.item()}
        
        if 'rgb_fine' in outputs:
            loss_fine = img2mse(outputs['rgb_fine'], target_rgb)
            total_loss += self.fine_loss_weight * loss_fine
            losses['loss_fine'] = loss_fine.item()
        
        losses['total_loss'] = total_loss.item()
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return losses
    
    def validate(self) -> float:
        """
        Validate the model on a subset of validation images.
        
        Returns:
            Average PSNR on validation set
        """
        print("Running validation...")
        self.model.eval()
        self.renderer.training = False
        
        psnrs = []
        
        with torch.no_grad():
            # Validate on a few images
            n_val_images = min(5, len(self.val_dataset))
            
            for i in range(n_val_images):
                sample = self.val_dataset[i]
                
                # Render image
                outputs = self.renderer.render_image(
                    self.model, 
                    sample['H'], 
                    sample['W'], 
                    sample['focal'], 
                    sample['pose']
                )
                
                # Use fine network output if available, otherwise coarse
                if 'rgb_fine' in outputs:
                    pred_rgb = outputs['rgb_fine']
                else:
                    pred_rgb = outputs['rgb_coarse']
                
                # Compute PSNR
                target_rgb = sample['image']
                mse = img2mse(pred_rgb, target_rgb)
                psnr = mse2psnr(mse)
                psnrs.append(psnr.item())
        
        avg_psnr = np.mean(psnrs)
        print(f"Validation PSNR: {avg_psnr:.2f}")
        
        return avg_psnr
    
    def render_test_image(self):
        """Render and save a test image."""
        print("Rendering test image...")
        self.model.eval()
        self.renderer.training = False
        
        with torch.no_grad():
            # Use first validation image
            sample = self.val_dataset[0]
            
            # Render image
            outputs = self.renderer.render_image(
                self.model,
                sample['H'],
                sample['W'], 
                sample['focal'],
                sample['pose']
            )
            
            # Use fine network output if available
            if 'rgb_fine' in outputs:
                pred_rgb = outputs['rgb_fine'].cpu().numpy()
                depth = outputs['depth_fine'].cpu().numpy()
            else:
                pred_rgb = outputs['rgb_coarse'].cpu().numpy()
                depth = outputs['depth_coarse'].cpu().numpy()
            
            target_rgb = sample['image'].cpu().numpy()
            
            # Convert to 8-bit
            pred_rgb_8b = to8b(pred_rgb)
            target_rgb_8b = to8b(target_rgb)
            
            # Save images
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(target_rgb_8b)
            plt.title('Target')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(pred_rgb_8b)
            plt.title('Predicted')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(depth, cmap='viridis')
            plt.title('Depth')
            plt.axis('off')
            plt.colorbar()
            
            # Save plot
            save_path = os.path.join(self.render_dir, f'test_render_{self.iteration:06d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved test render to {save_path}")
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_psnrs': self.val_psnrs
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{self.iteration:06d}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.iteration = checkpoint['iteration']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_psnrs = checkpoint.get('val_psnrs', [])
        
        print(f"Resumed training from iteration {self.iteration}")
    
    def train(self):
        """Main training loop."""
        print("Starting NeRF training...")
        print(f"Training for {self.max_iterations} iterations")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        # Create infinite data iterator
        train_iter = iter(self.train_loader)
        
        while self.iteration < self.max_iterations:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart data loader
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            losses = self.train_step(batch)
            self.train_losses.append(losses)
            
            self.iteration += 1
            
            # Logging
            if self.iteration % self.log_every == 0:
                elapsed = time.time() - start_time
                print(f"Iter {self.iteration:6d} | "
                      f"Loss: {losses['total_loss']:.4f} | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                      f"Time: {elapsed:.1f}s")
            
            # Validation
            if self.iteration % self.validate_every == 0:
                psnr = self.validate()
                self.val_psnrs.append((self.iteration, psnr))
            
            # Render test image
            if self.iteration % self.render_test_every == 0:
                self.render_test_image()
            
            # Save checkpoint
            if self.iteration % self.save_checkpoint_every == 0:
                self.save_checkpoint()
        
        print("Training completed!")
        self.save_checkpoint()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train NeRF model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = NeRFTrainer(config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
