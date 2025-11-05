from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random
import torchvision.transforms.functional as TF


class CrowdDataset(Dataset):
    def __init__(self, img_root, gt_dmap_root, img_names=None, gt_downsample=1, augment=False):
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.gt_downsample = gt_downsample
        self.augment = augment  # Enable augmentation

        # Get image names
        if img_names is not None:
            self.img_names = img_names
        else:
            self.img_names = [
                filename for filename in os.listdir(img_root)
                if os.path.isfile(os.path.join(img_root, filename))
            ]
        self.n_samples = len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = plt.imread(os.path.join(self.img_root, img_name))

        # Convert grayscale to RGB
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)

        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        # Load corresponding density map
        gt_path = os.path.join(self.gt_dmap_root, img_name.replace('.jpg', '.npy'))
        gt_dmap = np.load(gt_path).astype(np.float32)

        target_size = 512
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Resize + rescale density map based on gt_downsample
        orig_sum = np.sum(gt_dmap)
        if self.gt_downsample > 1:
            # Downsample the density map to match model output size
            ds_size = target_size // self.gt_downsample
            gt_resized = cv2.resize(gt_dmap, (ds_size, ds_size), interpolation=cv2.INTER_LINEAR)
        else:
            # Keep same size as input image
            gt_resized = cv2.resize(gt_dmap, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Rescale to preserve total count
        resized_sum = np.sum(gt_resized)
        if resized_sum > 0 and orig_sum > 0:
            gt_resized *= (orig_sum / resized_sum)
        gt_dmap = gt_resized[np.newaxis, :, :]

        # --- 🔁 DATA AUGMENTATION ---
        if self.augment:
            # Convert to tensor format for torchvision transforms
            img_tensor = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32)
            gt_tensor = torch.tensor(gt_dmap, dtype=torch.float32)

            # Random horizontal flip
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                gt_tensor = TF.hflip(gt_tensor)

            # Random vertical flip
            if random.random() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                gt_tensor = TF.vflip(gt_tensor)

            # Random rotation (-15° to +15°)
            angle = random.uniform(-15, 15)
            img_tensor = TF.rotate(img_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
            gt_tensor = TF.rotate(gt_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)

            # Optional: Random brightness jitter
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img_tensor = TF.adjust_brightness(img_tensor, factor)

            return img_tensor, gt_tensor

        # --- Default (no augmentation) ---
        img = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32)
        gt_dmap = torch.tensor(gt_dmap, dtype=torch.float32)
        return img, gt_dmap


# --- Test code ---
if __name__ == "__main__":
    img_root = "./data/train_data/images"
    gt_dmap_root = "./data/train_data/densitymaps"
    dataset = CrowdDataset(img_root, gt_dmap_root, gt_downsample=4, augment=True)
    
    for i, (img, gt_dmap) in enumerate(dataset):
        print(img.shape, gt_dmap.shape)
        if i > 5:
            break
