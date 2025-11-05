import torch
import torch.nn as nn
import torch.nn.functional as F

class MCNN(nn.Module):
    '''
    Original Implementation of Multi-column CNN for crowd counting
    '''
    def __init__(self, load_weights=False):
        super(MCNN, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 7, padding=3),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, 5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(nn.Conv2d(30, 1, 1, padding=0))

        if not load_weights:
            self._initialize_weights()

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)
        x2 = self.branch2(img_tensor)
        x3 = self.branch3(img_tensor)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ImprovedMCNN(nn.Module):
    '''
    Enhanced Multi-column CNN for crowd counting with improved accuracy
    '''
    def __init__(self, load_weights=False):
        super(ImprovedMCNN, self).__init__()

        # Enhanced Branch 1 - Large receptive field with BatchNorm
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 7, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # Enhanced Branch 2 - Medium receptive field
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, 5, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
        )

        # Enhanced Branch 3 - Small receptive field
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )

        # Enhanced fusion with proper dimensions
        self.fuse = nn.Sequential(
            nn.Conv2d(30, 16, 3, padding=1),  # 8+10+12=30 channels
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )

        if not load_weights:
            self._initialize_weights()

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)
        x2 = self.branch2(img_tensor)
        x3 = self.branch3(img_tensor)
        
        # Ensure all tensors have the same spatial dimensions
        target_h, target_w = x1.shape[2], x1.shape[3]
        if x2.shape[2:] != (target_h, target_w):
            x2 = F.interpolate(x2, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if x3.shape[2:] != (target_h, target_w):
            x3 = F.interpolate(x3, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Concatenate features
        x = torch.cat((x1, x2, x3), 1)
        
        # Final fusion
        x = self.fuse(x)
        
        # Ensure positive outputs for counting
        x = F.relu(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# Fixed version of MCNNPlusPlus with proper spatial alignment
class MCNNPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c, k, p, d=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, dilation=d, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Branch 1 (large receptive field) - Fixed padding
        self.branch1 = nn.Sequential(
            conv_block(3, 16, 9, 4),
            nn.MaxPool2d(2),
            conv_block(16, 32, 7, 3),
            conv_block(32, 16, 7, 3),  # Removed dilation for consistent size
        )

        # Branch 2 (medium receptive field) - Fixed padding
        self.branch2 = nn.Sequential(
            conv_block(3, 20, 7, 3),
            nn.MaxPool2d(2),
            conv_block(20, 40, 5, 2),
            conv_block(40, 20, 5, 2),  # Removed dilation for consistent size
        )

        # Branch 3 (small receptive field) - Fixed padding
        self.branch3 = nn.Sequential(
            conv_block(3, 24, 5, 2),
            nn.MaxPool2d(2),
            conv_block(24, 48, 3, 1),
            conv_block(48, 24, 3, 1),  # Removed dilation for consistent size
        )

        # Fusion with proper input channels (16+20+24=60)
        self.fuse = nn.Sequential(
            nn.Conv2d(60, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        # Ensure all branches have the same spatial dimensions
        target_h, target_w = x1.shape[2], x1.shape[3]
        if x2.shape[2:] != (target_h, target_w):
            x2 = F.interpolate(x2, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if x3.shape[2:] != (target_h, target_w):
            x3 = F.interpolate(x3, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.fuse(x)
        return F.relu(out)
