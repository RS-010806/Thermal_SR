"""
Training script for thermal super-resolution model.

Features:
- Multi-stage training (initial, refinement, adversarial)
- Physics-informed loss functions
- Curriculum learning
- Mixed precision training
- Tensorboard logging
"""

import os
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.models import ThermalSRWithGAN, ThermalSuperResolutionNet
from src.data_loader import create_dataloaders
from src.losses import ThermalSRLoss, DiscriminatorLoss
from src.metrics import compute_metrics
from src.utils import (
    save_checkpoint, load_checkpoint, 
    visualize_results, set_random_seed
)


class Trainer:
    """
    Trainer class for thermal super-resolution model.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed for reproducibility
        set_random_seed(config['seed'])
        
        # Create model
        self.model = ThermalSRWithGAN(scale_factor=config['scale_factor'])
        self.model = self.model.to(self.device)
        
        # Create optimizers
        self.optimizer_g = optim.AdamW(
            self.model.generator.parameters(),
            lr=config['lr_generator'],
            betas=(0.9, 0.99),
            weight_decay=config['weight_decay']
        )
        
        self.optimizer_d = optim.AdamW(
            self.model.discriminator.parameters(),
            lr=config['lr_discriminator'],
            betas=(0.9, 0.99),
            weight_decay=config['weight_decay']
        )
        
        # Create learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g,
            T_max=config['num_epochs'],
            eta_min=config['lr_min']
        )
        
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d,
            T_max=config['num_epochs'],
            eta_min=config['lr_min']
        )
        
        # Create loss functions
        self.criterion_g = ThermalSRLoss(
            lambda_pixel=config['lambda_pixel'],
            lambda_perceptual=config['lambda_perceptual'],
            lambda_physics=config['lambda_physics'],
            lambda_spectral=config['lambda_spectral'],
            lambda_edge=config['lambda_edge'],
            lambda_adversarial=config['lambda_adversarial']
        )
        
        self.criterion_d = DiscriminatorLoss(loss_type=config['gan_loss_type'])
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            patch_size=config['patch_size'],
            scale_factor=config['scale_factor'],
            num_workers=config['num_workers']
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config['use_amp'] else None
        
        # Logging
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.global_step = 0
        self.epoch = 0
        
        # Training phase management
        self.current_phase = 'initial'
        self.phase_epochs = {
            'initial': config['initial_epochs'],
            'refinement': config['refinement_epochs'],
            'adversarial': config['adversarial_epochs']
        }
        
        # Best model tracking
        self.best_psnr = 0.0
        self.best_epoch = 0
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'g_total': 0.0,
            'd_total': 0.0,
            'pixel': 0.0,
            'perceptual': 0.0,
            'physics': 0.0,
            'spectral': 0.0,
            'edge': 0.0,
            'adversarial': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} [{self.current_phase}]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            thermal_lr = batch['thermal_lr'].to(self.device)
            thermal_hr = batch['thermal_hr'].to(self.device)
            optical = batch['optical'].to(self.device)
            
            # Generator training
            self.optimizer_g.zero_grad()
            
            if self.scaler:
                with autocast():
                    # Forward pass
                    thermal_sr, emissivity, physics_outputs = self.model(thermal_lr, optical)
                    
                    # Compute generator loss
                    if self.current_phase == 'adversarial':
                        # Get discriminator predictions for adversarial loss
                        with torch.no_grad():
                            disc_pred_fake = self.model.discriminator(thermal_sr)
                    else:
                        disc_pred_fake = None
                        
                    g_loss, g_loss_dict = self.criterion_g(
                        pred=thermal_sr,
                        target=thermal_hr,
                        lr_input=thermal_lr,
                        emissivity=emissivity,
                        optical_features=optical,
                        discriminator_pred=disc_pred_fake,
                        training_phase=self.current_phase
                    )
                    
                # Backward pass with mixed precision
                self.scaler.scale(g_loss).backward()
                self.scaler.step(self.optimizer_g)
                self.scaler.update()
            else:
                # Forward pass
                thermal_sr, emissivity, physics_outputs = self.model(thermal_lr, optical)
                
                # Compute generator loss
                if self.current_phase == 'adversarial':
                    disc_pred_fake = self.model.discriminator(thermal_sr)
                else:
                    disc_pred_fake = None
                    
                g_loss, g_loss_dict = self.criterion_g(
                    pred=thermal_sr,
                    target=thermal_hr,
                    lr_input=thermal_lr,
                    emissivity=emissivity,
                    optical_features=optical,
                    discriminator_pred=disc_pred_fake,
                    training_phase=self.current_phase
                )
                
                # Backward pass
                g_loss.backward()
                self.optimizer_g.step()
                
            # Discriminator training (only in adversarial phase)
            if self.current_phase == 'adversarial':
                self.optimizer_d.zero_grad()
                
                if self.scaler:
                    with autocast():
                        # Discriminator predictions
                        disc_pred_real = self.model.discriminator(thermal_hr)
                        disc_pred_fake = self.model.discriminator(thermal_sr.detach())
                        
                        # Compute discriminator loss
                        d_loss, d_loss_dict = self.criterion_d(disc_pred_real, disc_pred_fake)
                        
                    # Backward pass
                    self.scaler.scale(d_loss).backward()
                    self.scaler.step(self.optimizer_d)
                    self.scaler.update()
                else:
                    # Discriminator predictions
                    disc_pred_real = self.model.discriminator(thermal_hr)
                    disc_pred_fake = self.model.discriminator(thermal_sr.detach())
                    
                    # Compute discriminator loss
                    d_loss, d_loss_dict = self.criterion_d(disc_pred_real, disc_pred_fake)
                    
                    # Backward pass
                    d_loss.backward()
                    self.optimizer_d.step()
                    
                epoch_losses['d_total'] += d_loss.item()
            else:
                d_loss_dict = {'total': 0.0}
                
            # Update epoch losses
            epoch_losses['g_total'] += g_loss.item()
            epoch_losses['pixel'] += g_loss_dict.get('pixel', 0.0).item()
            epoch_losses['perceptual'] += g_loss_dict.get('perceptual', 0.0).item()
            epoch_losses['physics'] += g_loss_dict.get('physics_total', 0.0).item()
            epoch_losses['spectral'] += g_loss_dict.get('spectral', 0.0).item()
            epoch_losses['edge'] += g_loss_dict.get('edge', 0.0).item()
            epoch_losses['adversarial'] += g_loss_dict.get('adversarial', 0.0).item()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{g_loss.item():.4f}",
                'D': f"{d_loss_dict['total']:.4f}",
                'Pixel': f"{g_loss_dict.get('pixel', 0.0).item():.4f}",
                'Physics': f"{g_loss_dict.get('physics_total', 0.0).item():.4f}"
            })
            
            # Log to tensorboard
            if self.global_step % self.config['log_interval'] == 0:
                self.log_training_step(g_loss_dict, d_loss_dict, physics_outputs)
                
            # Visualize results
            if self.global_step % self.config['vis_interval'] == 0:
                self.visualize_training_batch(
                    thermal_lr, thermal_hr, thermal_sr, 
                    optical, emissivity
                )
                
            self.global_step += 1
            
        # Average epoch losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'pixel': 0.0,
            'physics': 0.0
        }
        
        val_metrics = {
            'psnr': 0.0,
            'ssim': 0.0,
            'rmse_kelvin': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                thermal_lr = batch['thermal_lr'].to(self.device)
                thermal_hr = batch['thermal_hr'].to(self.device)
                optical = batch['optical'].to(self.device)
                
                # Forward pass
                thermal_sr, emissivity, physics_outputs = self.model(thermal_lr, optical)
                
                # Compute loss
                val_loss, val_loss_dict = self.criterion_g(
                    pred=thermal_sr,
                    target=thermal_hr,
                    lr_input=thermal_lr,
                    emissivity=emissivity,
                    optical_features=optical,
                    training_phase='refinement'  # Use refinement phase for validation
                )
                
                # Update losses
                val_losses['total'] += val_loss.item()
                val_losses['pixel'] += val_loss_dict.get('pixel', 0.0).item()
                val_losses['physics'] += val_loss_dict.get('physics_total', 0.0).item()
                
                # Compute metrics
                metrics = compute_metrics(
                    thermal_sr, thermal_hr,
                    data_range=1.0,  # Assuming normalized data
                    compute_kelvin=True,
                    thermal_mean=self.train_loader.dataset.thermal_mean,
                    thermal_std=self.train_loader.dataset.thermal_std
                )
                
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
                    
        # Average validation results
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches
            
        return val_losses, val_metrics
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Total epochs: {sum(self.phase_epochs.values())}")
        
        # Load checkpoint if resuming
        if self.config['resume']:
            self.load_checkpoint()
            
        # Training loop
        for phase, num_epochs in self.phase_epochs.items():
            self.current_phase = phase
            print(f"\n{'='*50}")
            print(f"Training Phase: {phase.upper()} ({num_epochs} epochs)")
            print(f"{'='*50}")
            
            for epoch_in_phase in range(num_epochs):
                self.epoch += 1
                
                # Train for one epoch
                train_losses = self.train_epoch()
                
                # Validate
                val_losses, val_metrics = self.validate()
                
                # Update learning rates
                self.scheduler_g.step()
                if self.current_phase == 'adversarial':
                    self.scheduler_d.step()
                    
                # Log epoch results
                self.log_epoch_results(train_losses, val_losses, val_metrics)
                
                # Save checkpoint
                if self.epoch % self.config['save_interval'] == 0:
                    self.save_checkpoint()
                    
                # Save best model
                if val_metrics['psnr'] > self.best_psnr:
                    self.best_psnr = val_metrics['psnr']
                    self.best_epoch = self.epoch
                    self.save_checkpoint(is_best=True)
                    
                print(f"Epoch {self.epoch}: "
                      f"Train Loss: {train_losses['g_total']:.4f}, "
                      f"Val PSNR: {val_metrics['psnr']:.2f} dB, "
                      f"Val SSIM: {val_metrics['ssim']:.4f}, "
                      f"Val RMSE: {val_metrics['rmse_kelvin']:.2f} K")
                
        print(f"\nTraining completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB at epoch {self.best_epoch}")
        
        # Final evaluation on test set
        self.evaluate_test_set()
        
    def evaluate_test_set(self):
        """Evaluate on test set."""
        print("\nEvaluating on test set...")
        
        # Load best model
        best_checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.model.eval()
        
        test_metrics = {
            'psnr': [],
            'ssim': [],
            'rmse_kelvin': []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Test'):
                thermal_lr = batch['thermal_lr'].to(self.device)
                thermal_hr = batch['thermal_hr'].to(self.device)
                optical = batch['optical'].to(self.device)
                
                # Forward pass
                thermal_sr, _, _ = self.model(thermal_lr, optical)
                
                # Compute metrics
                metrics = compute_metrics(
                    thermal_sr, thermal_hr,
                    data_range=1.0,
                    compute_kelvin=True,
                    thermal_mean=self.train_loader.dataset.thermal_mean,
                    thermal_std=self.train_loader.dataset.thermal_std
                )
                
                for key in test_metrics:
                    test_metrics[key].append(metrics[key])
                    
        # Compute statistics
        print("\nTest Set Results:")
        for key in test_metrics:
            values = np.array(test_metrics[key])
            mean = values.mean()
            std = values.std()
            print(f"{key.upper()}: {mean:.4f} ± {std:.4f}")
            
    def log_training_step(self, g_loss_dict, d_loss_dict, physics_outputs):
        """Log training step to tensorboard."""
        # Generator losses
        for key, value in g_loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f'Train/G_{key}', value.item(), self.global_step)
                
        # Discriminator losses
        if self.current_phase == 'adversarial':
            for key, value in d_loss_dict.items():
                if isinstance(value, torch.Tensor):
                    self.writer.add_scalar(f'Train/D_{key}', value.item(), self.global_step)
                    
        # Physics outputs
        for key, value in physics_outputs.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(f'Train/Physics_{key}', value.item(), self.global_step)
                
    def log_epoch_results(self, train_losses, val_losses, val_metrics):
        """Log epoch results to tensorboard."""
        # Training losses
        for key, value in train_losses.items():
            self.writer.add_scalar(f'Epoch/Train_{key}', value, self.epoch)
            
        # Validation losses
        for key, value in val_losses.items():
            self.writer.add_scalar(f'Epoch/Val_{key}', value, self.epoch)
            
        # Validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Epoch/Val_{key}', value, self.epoch)
            
        # Learning rates
        self.writer.add_scalar('Epoch/LR_G', self.optimizer_g.param_groups[0]['lr'], self.epoch)
        if self.current_phase == 'adversarial':
            self.writer.add_scalar('Epoch/LR_D', self.optimizer_d.param_groups[0]['lr'], self.epoch)
            
    def visualize_training_batch(self, thermal_lr, thermal_hr, thermal_sr, optical, emissivity):
        """Visualize training batch results."""
        # Take first sample from batch
        idx = 0
        
        # Denormalize for visualization
        thermal_lr_vis = self.train_loader.dataset.denormalize_thermal(thermal_lr[idx])
        thermal_hr_vis = self.train_loader.dataset.denormalize_thermal(thermal_hr[idx])
        thermal_sr_vis = self.train_loader.dataset.denormalize_thermal(thermal_sr[idx])
        
        # Create visualization
        vis_dict = {
            'thermal_lr': thermal_lr_vis,
            'thermal_hr': thermal_hr_vis,
            'thermal_sr': thermal_sr_vis,
            'optical_rgb': optical[idx, [3, 2, 1]],  # RGB bands
            'emissivity': emissivity[idx]
        }
        
        fig = visualize_results(vis_dict)
        
        # Log to tensorboard
        self.writer.add_figure('Visualization', fig, self.global_step)
        
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_psnr': self.best_psnr,
            'best_epoch': self.best_epoch,
            'current_phase': self.current_phase,
            'config': self.config
        }
        
        # Save checkpoint
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            checkpoint_path = checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
    def load_checkpoint(self):
        """Load model checkpoint."""
        checkpoint_path = Path(self.config['resume_path'])
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return
            
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint['best_psnr']
        self.best_epoch = checkpoint['best_epoch']
        self.current_phase = checkpoint['current_phase']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")


def main():
    parser = argparse.ArgumentParser(description='Train Thermal Super-Resolution Model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Update config with command line arguments
    config['resume'] = args.resume
    if args.resume_path:
        config['resume_path'] = args.resume_path
        
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Train model
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
