import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Dataset
# ---------------------------
class RockLetterDataset(Dataset):
    def __init__(self, img_path, mask_path, patch_size=256, transform=None):
        self.patch_size = patch_size
        self.transform = transform
        
        with rasterio.open(img_path) as src:
            self.img = src.read([1,2,3])  # RGB bands
        with rasterio.open(mask_path) as src:
            self.mask = src.read(1)
        
        # Normalize image to 0-1
        self.img = self.img.astype(np.float32) / 255.0
        # Binary mask
        self.mask = (self.mask > 0).astype(np.float32)
        
        # Generate patches
        self.patches = self.create_patches()

    def create_patches(self):
        patches = []
        c, h, w = self.img.shape
        ps = self.patch_size
        for i in range(0, h - ps + 1, ps):
            for j in range(0, w - ps + 1, ps):
                img_patch = self.img[:, i:i+ps, j:j+ps]
                mask_patch = self.mask[i:i+ps, j:j+ps]
                patches.append((img_patch, mask_patch))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img, mask = self.patches[idx]
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dim
        return img, mask

# ---------------------------
# Simple UNet
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(n_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.pool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.upsample(conv2)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        return x

# ---------------------------
# Paths
# ---------------------------
IMG_PATH = "m_3911926_sw_11_060_20220614.tif"
MASK_PATH = "rock_letters_mask.tif"

# ---------------------------
# Dataset & Loader
# ---------------------------
dataset = RockLetterDataset(IMG_PATH, MASK_PATH, patch_size=256)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# ---------------------------
# Model, Loss, Optimizer
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# Training Loop
# ---------------------------
EPOCHS = 5  # keep small for testing
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

# ---------------------------
# Test on first patch
# ---------------------------
model.eval()
with torch.no_grad():
    img, mask = dataset[0]
    img = img.unsqueeze(0).to(device)
    pred = torch.sigmoid(model(img)).cpu().squeeze().numpy()

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Input")
plt.imshow(np.moveaxis(img.cpu().squeeze().numpy(), 0, -1))
plt.subplot(1,3,2)
plt.title("Mask")
plt.imshow(mask.squeeze(), cmap="gray")
plt.subplot(1,3,3)
plt.title("Predicted")
plt.imshow(pred, cmap="gray")
plt.show()