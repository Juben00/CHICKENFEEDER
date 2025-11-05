#!/usr/bin/env python3
"""
Optimized training script for MCNN pellet counting with maximum accuracy focus
- Enhanced loss functions
- Advanced learning rate scheduling 
- Better regularization
- Comprehensive validation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our enhanced models and dataloader
try:
    from enhanced_mcnn_model import EnhancedMCNNForPellets
    ENHANCED_AVAILABLE = True
    print("✅ Using enhanced models and dataloader")
except ImportError as e:
    print(f"⚠️ Enhanced model import failed: {e}")
    ENHANCED_AVAILABLE = False
    print("⚠️ Using fallback model")

# Import dataloader (use original for compatibility)
from my_dataloader import CrowdDataset as EnhancedPelletDataset

# Optional advanced losses
try:
    from pytorch_msssim import ssim, SSIM
    SSIM_AVAILABLE = True
    print("✅ SSIM loss available")
except ImportError:
    SSIM_AVAILABLE = False
    print("⚠️ SSIM loss not available")


class OptimizedTrainingConfig:
    """Optimized configuration for pellet counting"""
    def __init__(self):
        # Core settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 200
        self.batch_size = 4  # Increased for better gradient estimates
        self.num_workers = 4
        self.pin_memory = True
        
        # Learning rate and optimization
        self.initial_lr = 1e-4  # Conservative start
        self.weight_decay = 1e-4
        self.gradient_clip_norm = 1.0
        
        # Advanced scheduling
        self.warmup_epochs = 10
        self.cosine_restarts = True
        self.t_max = 50  # Cosine annealing period
        
        # Early stopping
        self.patience = 50
        self.min_delta = 0.1
        
        # Model saving
        self.save_dir = './checkpoints'
        self.log_csv = os.path.join(self.save_dir, 'training_log_optimized.csv')
        
        # Loss weights
        self.mse_weight = 1.0
        self.mae_weight = 0.5
        self.ssim_weight = 0.3 if SSIM_AVAILABLE else 0.0
        self.count_loss_weight = 2.0  # Focus on accurate counting
        
        # Mixed precision
        self.use_mixed_precision = torch.cuda.is_available()
        
        # Data settings
        self.target_size = 512
        self.gt_downsample = 4  # 128x128 output


class CombinedLoss(nn.Module):
    """
    Advanced loss combining multiple objectives for better pellet counting
    """
    def __init__(self, mse_weight=1.0, mae_weight=0.5, ssim_weight=0.3, count_weight=2.0):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.ssim_weight = ssim_weight
        self.count_weight = count_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        if SSIM_AVAILABLE and ssim_weight > 0:
            self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)
        else:
            self.ssim_loss = None
    
    def forward(self, pred, target):
        # Basic pixel-wise losses
        mse = self.mse_loss(pred, target)
        mae = self.mae_loss(pred, target)
        
        # Count-based loss (most important for accuracy)
        pred_count = torch.sum(pred, dim=(2, 3))
        target_count = torch.sum(target, dim=(2, 3))
        count_loss = nn.functional.mse_loss(pred_count, target_count)
        
        # Combine losses
        total_loss = (self.mse_weight * mse + 
                     self.mae_weight * mae + 
                     self.count_weight * count_loss)
        
        # Add SSIM if available
        if self.ssim_loss is not None and self.ssim_weight > 0:
            # SSIM expects values in [0,1], so normalize
            pred_norm = torch.clamp(pred / (pred.max() + 1e-8), 0, 1)
            target_norm = torch.clamp(target / (target.max() + 1e-8), 0, 1)
            ssim_val = self.ssim_loss(pred_norm, target_norm)
            ssim_loss = 1.0 - ssim_val
            total_loss += self.ssim_weight * ssim_loss
            
            return total_loss, {
                'mse': mse.item(),
                'mae': mae.item(), 
                'count_loss': count_loss.item(),
                'ssim_loss': ssim_loss.item(),
                'total': total_loss.item()
            }
        
        return total_loss, {
            'mse': mse.item(),
            'mae': mae.item(),
            'count_loss': count_loss.item(), 
            'total': total_loss.item()
        }


def compute_metrics(model, dataloader, device, criterion):
    """Comprehensive metric computation"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)
            
            # Count-based metrics
            pred_counts = torch.sum(predictions, dim=(2, 3))
            target_counts = torch.sum(targets, dim=(2, 3))
            
            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            
            # Accumulate errors
            for i in range(batch_size):
                pred_count = pred_counts[i].item()
                target_count = target_counts[i].item()
                error = abs(pred_count - target_count)
                total_mae += error
                total_mse += error * error
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples
    avg_rmse = np.sqrt(avg_mse)
    
    return avg_loss, avg_mae, avg_mse, avg_rmse


def train_optimized():
    """Main optimized training function"""
    config = OptimizedTrainingConfig()
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    print(f"🚀 Starting Optimized Pellet Counting Training")
    print(f"📱 Device: {config.device}")
    print(f"🎯 Target: Maximum accuracy for small dense pellets")
    print(f"📊 Enhanced features: {'✅' if ENHANCED_AVAILABLE else '❌'}")
    print(f"🔧 SSIM loss: {'✅' if SSIM_AVAILABLE else '❌'}")
    
    # Initialize model
    model = EnhancedMCNNForPellets().to(config.device)
    model_name = "EnhancedMCNN" if ENHANCED_AVAILABLE else "ImprovedMCNN"
    print(f"✅ Using {model_name}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss function
    criterion = CombinedLoss(
        mse_weight=config.mse_weight,
        mae_weight=config.mae_weight,
        ssim_weight=config.ssim_weight,
        count_weight=config.count_loss_weight
    ).to(config.device)
    
    # Optimizer with advanced settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup + cosine annealing
    if config.cosine_restarts:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.t_max, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
    
    # Mixed precision scaler
    if config.use_mixed_precision:
        try:
            scaler = torch.amp.GradScaler('cuda')  # Updated API
        except:
            scaler = torch.cuda.amp.GradScaler()  # Fallback for older versions
    else:
        scaler = None
    
    # Datasets - use standard dataloader for compatibility
    train_dataset = EnhancedPelletDataset(
        './data/train_data/images',
        './data/train_data/densitymaps',
        gt_downsample=config.gt_downsample,
        augment=True
    )
    
    val_dataset = EnhancedPelletDataset(
        './data/test_data/images', 
        './data/test_data/densitymaps',
        gt_downsample=config.gt_downsample,
        augment=False
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Training batches: {len(train_loader)}")
    
    # Initialize CSV logging with robust fallback for permission issues
    def _open_log_csv(path, mode='w'):
        try:
            f = open(path, mode, newline='')
            return f, path
        except PermissionError:
            # Fallback to a unique filename to avoid permission issues
            import time as _time, os as _os
            pid = _os.getpid()
            ts = int(_time.time())
            fallback = os.path.join(config.save_dir, f"training_log_optimized_{pid}_{ts}.csv")
            f = open(fallback, mode, newline='')
            return f, fallback

    # Ensure directory exists
    os.makedirs(config.save_dir, exist_ok=True)
    log_file_obj, used_log_path = _open_log_csv(config.log_csv, 'w')
    with log_file_obj as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'train_mae', 'val_loss', 'val_mae', 'val_rmse', 'time'])
    
    # Attempt to resume from the latest checkpoint if available
    def _find_latest_checkpoint(checkpoint_dir):
        files = []
        for name in os.listdir(checkpoint_dir):
            if name.endswith('.pth') and ('optimized' in name or 'best' in name):
                try:
                    # try to extract epoch number
                    import re
                    m = re.search(r"epoch_(\d+)", name)
                    epoch_no = int(m.group(1)) if m else 0
                except Exception:
                    epoch_no = 0
                files.append((epoch_no, os.path.join(checkpoint_dir, name)))
        if not files:
            return None, 0
        files.sort(key=lambda x: x[0])
        return files[-1][1], files[-1][0]

    latest_ckpt_path, latest_epoch = None, 0
    if os.path.isdir(config.save_dir):
        latest_ckpt_path, latest_epoch = _find_latest_checkpoint(config.save_dir)

    best_mae = float('inf')
    best_epoch = -1
    patience_counter = 0

    start_epoch = 0
    if latest_ckpt_path:
        try:
            print(f"🔁 Found checkpoint to resume: {latest_ckpt_path} (epoch {latest_epoch})")
            ck = torch.load(latest_ckpt_path, map_location=config.device)
            model.load_state_dict(ck.get('model_state_dict', model.state_dict()))
            optimizer.load_state_dict(ck.get('optimizer_state_dict', optimizer.state_dict()))
            try:
                scheduler.load_state_dict(ck.get('scheduler_state_dict', scheduler.state_dict()))
            except Exception:
                pass
            best_mae = ck.get('best_mae', best_mae)
            start_epoch = ck.get('epoch', latest_epoch)
            best_epoch = start_epoch - 1
            print(f"✅ Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️ Failed to resume from checkpoint: {e}")

    # Training loop
    for epoch in range(start_epoch, config.epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = model(images)
                    loss, loss_dict = criterion(predictions, targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(images)
                loss, loss_dict = criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            
            # Update metrics
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Count Loss': f"{loss_dict.get('count_loss', 0):.6f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate training metrics
        avg_train_loss = train_loss / train_samples
        
        # Validation phase
        val_loss, val_mae, val_mse, val_rmse = compute_metrics(model, val_loader, config.device, criterion)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Timing
        epoch_time = time.time() - start_time
        
        # Logging
        print(f"\n📈 Epoch {epoch+1}/{config.epochs} | LR: {current_lr:.2e}")
        print(f"   Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"   Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f}")
        print(f"   Time: {epoch_time:.1f}s")
        
        # Save to CSV (use robust opener that falls back on permission errors)
        try:
            # Prefer the previously used log path if available (it may be a fallback)
            log_target = used_log_path if 'used_log_path' in locals() and used_log_path else config.log_csv
            f_obj, actual_path = _open_log_csv(log_target, 'a')
            with f_obj as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, current_lr, avg_train_loss, 0, val_loss, val_mae, val_rmse, epoch_time
                ])
            # remember the actual file we successfully wrote to
            used_log_path = actual_path
        except Exception as e:
            # Don't crash training for logging problems; report and continue
            print(f"⚠️ Failed to write training log to CSV: {e}")
        
        # Model saving and early stopping
        if val_mae < best_mae - config.min_delta:
            best_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae,
                'config': config.__dict__
            }, os.path.join(config.save_dir, f'best_optimized_epoch_{epoch+1}.pth'))
            
            print(f"✅ New best model saved with MAE={val_mae:.2f}")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= config.patience:
            print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        print("-" * 80)
    
    # Training completion
    print(f"\n🏁 Training completed!")
    print(f"🏆 Best MAE: {best_mae:.2f} at epoch {best_epoch + 1}")
    print(f"💾 Best model: best_optimized_epoch_{best_epoch + 1}.pth")
    
    return best_mae, best_epoch + 1


if __name__ == "__main__":
    try:
        best_mae, best_epoch = train_optimized()
        print(f"\n✅ Training successful!")
        print(f"📊 Final results: MAE={best_mae:.2f} at epoch {best_epoch}")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()