#%%
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import numpy as np
import cv2

# Import models with fallback
try:
    from enhanced_mcnn_model import EnhancedMCNNForPellets as MCNN
    print("✅ Using Enhanced MCNN model")
except ImportError:
    from mcnn_model import ImprovedMCNN as MCNN
    print("⚠️ Using fallback ImprovedMCNN model")

from my_dataloader import CrowdDataset


def load_model_smart(model_param_path, device):
    '''
    Smart model loading that handles different checkpoint formats and model types
    '''
    # Load checkpoint
    checkpoint = torch.load(model_param_path, map_location=device)
    
    # Try to determine model type from checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            state_dict = checkpoint
        
        # Check if this is an enhanced model or original model based on keys
        if any('fusion' in key for key in state_dict.keys()):
            # This is likely an Enhanced model
            try:
                from enhanced_mcnn_model import EnhancedMCNNForPellets
                model = EnhancedMCNNForPellets().to(device)
                model.load_state_dict(state_dict)
                print("✅ Loaded Enhanced MCNN model")
                return model
            except Exception as e:
                print(f"⚠️ Failed to load as Enhanced model: {e}")
        
        # Try original ImprovedMCNN
        try:
            from mcnn_model import ImprovedMCNN
            model = ImprovedMCNN().to(device)
            model.load_state_dict(state_dict)
            print("✅ Loaded Improved MCNN model")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load as Improved model: {e}")
        
        # Try original MCNN
        try:
            from mcnn_model import MCNN
            model = MCNN().to(device)
            model.load_state_dict(state_dict)
            print("✅ Loaded Original MCNN model")
            return model
        except Exception as e:
            print(f"❌ Failed to load as Original model: {e}")
    
    raise Exception("Could not determine model type or load checkpoint")


def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE, MSE, and RMSE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Smart model loading
    mcnn = load_model_smart(model_param_path, device)
    
    # Get list of image files
    img_names = [f for f in os.listdir(img_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    dataset=CrowdDataset(img_root,gt_dmap_root,img_names,gt_downsample=4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False)
    mcnn.eval()
    mae, mse = 0.0, 0.0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            # Calculate counts
            pred_count = et_dmap.data.sum().item()
            gt_count = gt_dmap.data.sum().item()
            diff = pred_count - gt_count
            mae += abs(diff)
            mse += diff * diff
            del img,gt_dmap,et_dmap

    mae /= len(dataloader)
    mse /= len(dataloader)
    rmse = np.sqrt(mse)
    
    print(f"model_param_path: {model_param_path}")
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

def compare_predictions(img_root, gt_dmap_root, model_param_path, index):
    '''
    Show comprehensive comparison: input image, ground truth, and prediction.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Smart model loading
    try:
        mcnn = load_model_smart(model_param_path, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get image files
    img_names = [f for f in os.listdir(img_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if index >= len(img_names):
        print(f"❌ Index {index} out of range. Available images: {len(img_names)}")
        return
    
    img_name = img_names[index]
    print(f"📷 Processing image: {img_name}")
    
    # Load original image
    original_img_path = os.path.join(img_root, img_name)
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"❌ Cannot load image: {original_img_path}")
        return
    
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load dataset for processed image and ground truth
    dataset = CrowdDataset(img_root, gt_dmap_root, img_names, gt_downsample=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    mcnn.eval()
    
    # Get the specific sample
    for i, (img_tensor, gt_dmap) in enumerate(dataloader):
        if i == index:
            img_tensor = img_tensor.to(device)
            gt_dmap = gt_dmap.to(device)
            
            # Forward pass
            with torch.no_grad():
                pred_dmap = mcnn(img_tensor).detach()
            
            # Convert to numpy for visualization
            img_display = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            
            # Denormalize if needed (assuming normalization was applied)
            if img_display.min() < 0:  # Likely normalized
                img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            
            img_display = np.clip(img_display, 0, 1)
            
            pred_display = pred_dmap.squeeze().cpu().numpy()
            gt_display = gt_dmap.squeeze().cpu().numpy()
            
            # Calculate metrics
            pred_count = pred_display.sum()
            gt_count = gt_display.sum()
            error = abs(pred_count - gt_count)
            error_percent = (error / gt_count * 100) if gt_count > 0 else 0
            
            print(f"📊 Ground Truth Count: {gt_count:.1f}")
            print(f"🔮 Predicted Count: {pred_count:.1f}")
            print(f"📈 Absolute Error: {error:.1f} ({error_percent:.1f}%)")
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(20, 12))
            
            # Original high-resolution image
            ax1 = plt.subplot(2, 3, 1)
            ax1.imshow(original_img_rgb)
            ax1.set_title(f'Original Image\n{original_img_rgb.shape[0]}×{original_img_rgb.shape[1]}', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Processed input image  
            ax2 = plt.subplot(2, 3, 2)
            ax2.imshow(img_display)
            ax2.set_title(f'Processed Input\n512×512 (Model Input)', fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            # Ground truth density map
            ax3 = plt.subplot(2, 3, 3)
            im3 = ax3.imshow(gt_display, cmap='jet', vmin=0, vmax=max(gt_display.max(), pred_display.max()))
            ax3.set_title(f'Ground Truth Density\nCount: {gt_count:.1f}', fontsize=14, fontweight='bold', color='green')
            ax3.axis('off')
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('Density', rotation=270, labelpad=15)
            
            # Predicted density map
            ax4 = plt.subplot(2, 3, 4) 
            im4 = ax4.imshow(pred_display, cmap='jet', vmin=0, vmax=max(gt_display.max(), pred_display.max()))
            ax4.set_title(f'Predicted Density\nCount: {pred_count:.1f}', fontsize=14, fontweight='bold', color='blue')
            ax4.axis('off')
            cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.set_label('Density', rotation=270, labelpad=15)
            
            # Difference map
            ax5 = plt.subplot(2, 3, 5)
            diff_map = pred_display - gt_display
            im5 = ax5.imshow(diff_map, cmap='RdBu_r', vmin=-diff_map.max(), vmax=diff_map.max())
            ax5.set_title(f'Difference Map\n(Pred - GT)', fontsize=14, fontweight='bold', color='red')
            ax5.axis('off')
            cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
            cbar5.set_label('Difference', rotation=270, labelpad=15)
            
            # Metrics summary
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            metrics_text = f"""
            📊 EVALUATION METRICS
            
            Image: {img_name}
            Original Size: {original_img_rgb.shape[0]}×{original_img_rgb.shape[1]}
            Model Input: 512×512
            Density Map: {pred_display.shape[0]}×{pred_display.shape[1]}
            
            Ground Truth Count: {gt_count:.1f}
            Predicted Count: {pred_count:.1f}
            
            Absolute Error: {error:.1f}
            Relative Error: {error_percent:.1f}%
            
            Density Map Stats:
            GT Max: {gt_display.max():.4f}
            GT Min: {gt_display.min():.4f}
            
            Pred Max: {pred_display.max():.4f} 
            Pred Min: {pred_display.min():.4f}
            """
            
            ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.suptitle(f'MCNN Pellet Counting - Prediction Analysis\nModel: {os.path.basename(model_param_path)}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save the comparison
            save_path = f'prediction_comparison_{index}_{img_name.split(".")[0]}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comparison saved as: {save_path}")
            
            plt.show()
            break

def estimate_density_map(img_root, gt_dmap_root, model_param_path, index):
    '''
    Legacy function for backward compatibility - calls the new comparison function
    '''
    compare_predictions(img_root, gt_dmap_root, model_param_path, index)


def compare_multiple_samples(img_root, gt_dmap_root, model_param_path, num_samples=6):
    '''
    Compare predictions for multiple samples in a grid layout
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Smart model loading
    try:
        mcnn = load_model_smart(model_param_path, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get image files
    img_names = [f for f in os.listdir(img_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_samples = min(num_samples, len(img_names))
    
    # Load dataset
    dataset = CrowdDataset(img_root, gt_dmap_root, img_names, gt_downsample=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    mcnn.eval()
    
    # Create grid visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    errors = []
    
    for i, (img_tensor, gt_dmap) in enumerate(dataloader):
        if sample_count >= num_samples:
            break
            
        img_tensor = img_tensor.to(device)
        gt_dmap = gt_dmap.to(device)
        
        # Forward pass
        with torch.no_grad():
            pred_dmap = mcnn(img_tensor).detach()
        
        # Convert to numpy
        img_display = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        if img_display.min() < 0:  # Denormalize if needed
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_display = np.clip(img_display, 0, 1)
        
        pred_display = pred_dmap.squeeze().cpu().numpy()
        gt_display = gt_dmap.squeeze().cpu().numpy()
        
        # Calculate error
        pred_count = pred_display.sum()
        gt_count = gt_display.sum()
        error = abs(pred_count - gt_count)
        errors.append(error)
        
        # Plot input image
        axes[sample_count, 0].imshow(img_display)
        axes[sample_count, 0].set_title(f'Input Image\n{img_names[i]}', fontsize=10)
        axes[sample_count, 0].axis('off')
        
        # Plot ground truth
        im_gt = axes[sample_count, 1].imshow(gt_display, cmap='jet')
        axes[sample_count, 1].set_title(f'Ground Truth\nCount: {gt_count:.1f}', fontsize=10, color='green')
        axes[sample_count, 1].axis('off')
        
        # Plot prediction
        im_pred = axes[sample_count, 2].imshow(pred_display, cmap='jet')
        color = 'red' if error > 10 else 'orange' if error > 5 else 'blue'
        axes[sample_count, 2].set_title(f'Prediction\nCount: {pred_count:.1f}\nError: {error:.1f}', 
                                       fontsize=10, color=color)
        axes[sample_count, 2].axis('off')
        
        sample_count += 1
    
    plt.suptitle(f'Multiple Sample Comparison\nAvg Error: {np.mean(errors):.1f} ± {np.std(errors):.1f}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison
    save_path = f'multiple_samples_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Multiple samples comparison saved as: {save_path}")
    
    plt.show()


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    img_root='./data/test_data/images'
    gt_dmap_root='./data/test_data/densitymaps'
    
    # Force use of specific checkpoint
    model_param_path = './checkpoints_optimized/best_optimized_epoch_26.pth'
    
    if not os.path.exists(model_param_path):
        print(f"❌ Specified checkpoint not found: {model_param_path}")
        print("Available checkpoints:")
        checkpoint_dirs = ['./checkpoints_optimized', './checkpoints']
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                for checkpoint in checkpoints:
                    print(f"  - {os.path.join(checkpoint_dir, checkpoint)}")
    else:
        print(f"🔧 Using forced model: {model_param_path}")
        
        # Calculate overall metrics
        print("\n📊 Calculating overall metrics...")
        cal_mae(img_root, gt_dmap_root, model_param_path)
        
        print("\n🖼️ Creating detailed comparison for sample image...")
        # Show detailed comparison for one image
        compare_predictions(img_root, gt_dmap_root, model_param_path, 3)
        
        print("\n📋 Creating multiple samples comparison...")
        # Show comparison for multiple samples
        compare_multiple_samples(img_root, gt_dmap_root, model_param_path, 6) 