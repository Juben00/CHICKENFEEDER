import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale features"""
    def __init__(self, in_channels, out_channels):
        super(ASPPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),  # Use bias instead of BatchNorm for 1x1 features
            nn.ReLU(inplace=True)
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn6 = nn.BatchNorm2d(out_channels)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.bn18 = nn.BatchNorm2d(out_channels)
        
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.final_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        size = x.shape[2:]
        
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv6 = F.relu(self.bn6(self.conv6(x)))
        conv12 = F.relu(self.bn12(self.conv12(x)))
        conv18 = F.relu(self.bn18(self.conv18(x)))
        global_pool = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        
        concat = torch.cat([conv1, conv6, conv12, conv18, global_pool], dim=1)
        out = F.relu(self.final_bn(self.final_conv(concat)))
        
        return out


class EnhancedMCNNForPellets(nn.Module):
    """
    Optimized Multi-Column CNN specifically designed for feed pellet counting
    - Enhanced for small, dense objects
    - Better feature extraction for texture patterns
    - Improved spatial resolution preservation
    - Simplified architecture to avoid BatchNorm issues
    """
    
    def __init__(self, load_weights=False):
        super(EnhancedMCNNForPellets, self).__init__()
        
        # Branch 1: Fine-grained features for small pellets (3x3, 5x5 convs)
        self.branch1 = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Light pooling to preserve small details
            nn.MaxPool2d(2, stride=2),  # 512->256
            
            # Enhanced feature extraction
            nn.Conv2d(32, 48, 5, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Branch 2: Medium-scale features (5x5, 7x7 convs)
        self.branch2 = nn.Sequential(
            # Slightly larger receptive field
            nn.Conv2d(3, 24, 5, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(24, 48, 7, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(48, 36, 5, padding=2, bias=False),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        
        # Branch 3: Large-scale context (7x7, 9x9 convs)
        self.branch3 = nn.Sequential(
            # Large receptive field for context
            nn.Conv2d(3, 20, 7, padding=3, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(20, 40, 9, padding=4, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(40, 30, 7, padding=3, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
            nn.Conv2d(30, 20, 5, padding=2, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
        )
        
        # Enhanced fusion decoder (32+24+20=76 channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(76, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Additional downsampling to match gt_downsample=4 (256->128)
            nn.MaxPool2d(2, stride=2),  # 256->128
            
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final density map prediction
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True)
        )
        
        # Initialize weights
        if not load_weights:
            self._initialize_weights()
    
    def forward(self, x):
        # Extract multi-scale features
        x1 = self.branch1(x)  # Fine details
        x2 = self.branch2(x)  # Medium features  
        x3 = self.branch3(x)  # Context
        
        # Ensure spatial alignment
        target_size = x1.shape[2:]
        if x2.shape[2:] != target_size:
            x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        if x3.shape[2:] != target_size:
            x3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate multi-scale features
        features = torch.cat([x1, x2, x3], dim=1)
        
        # Generate density map with fusion
        density_map = self.fusion(features)
        
        # Ensure non-negative outputs for counting
        density_map = F.relu(density_map)
        
        return density_map
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UltraEnhancedMCNN(nn.Module):
    """
    Ultra-enhanced version with additional improvements for maximum accuracy
    """
    
    def __init__(self, load_weights=False):
        super(UltraEnhancedMCNN, self).__init__()
        
        # Shared stem for efficient computation
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale branches with residual connections
        self.branch1 = self._make_branch(16, [32, 32, 32], [3, 3, 1], name="fine")
        self.branch2 = self._make_branch(16, [24, 48, 24], [5, 5, 3], name="medium") 
        self.branch3 = self._make_branch(16, [20, 40, 20], [7, 7, 3], name="coarse")
        
        # Feature pyramid network for better multi-scale fusion
        self.fpn = self._make_fpn([32, 24, 20], 64)
        
        # Counting head with spatial attention
        self.counting_head = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Spatial attention
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final prediction
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=True)
        )
        
        if not load_weights:
            self._initialize_weights()
    
    def _make_branch(self, in_channels, channels, kernel_sizes, name):
        layers = []
        prev_channels = in_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            padding = kernel_size // 2
            
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            
            # Add pooling after first conv in each branch
            if i == 0:
                layers.append(nn.MaxPool2d(2, stride=2))
            
            # Add SE block after last conv
            elif i == len(channels) - 1:
                layers.append(SEBlock(out_channels))
                
            prev_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def _make_fpn(self, in_channels_list, out_channels):
        # Simple FPN implementation
        return nn.Sequential(
            nn.Conv2d(sum(in_channels_list), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Shared feature extraction
        stem_features = self.stem(x)
        
        # Multi-scale feature extraction
        x1 = self.branch1(stem_features)
        x2 = self.branch2(stem_features)
        x3 = self.branch3(stem_features)
        
        # Spatial alignment
        target_size = x1.shape[2:]
        if x2.shape[2:] != target_size:
            x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        if x3.shape[2:] != target_size:
            x3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        
        # Feature fusion
        multi_scale_features = torch.cat([x1, x2, x3], dim=1)
        fused_features = self.fpn(multi_scale_features)
        
        # Generate density map
        density_map = self.counting_head(fused_features)
        
        return F.relu(density_map)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Alias for easy import
PelletMCNN = EnhancedMCNNForPellets