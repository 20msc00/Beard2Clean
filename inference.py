import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # For tabular display


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        # Encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)
        # Decoder
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)

def load_model(checkpoint_path, device):
    model = GeneratorUNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    print("Model loaded from:", checkpoint_path)
    return model

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # Normalization to [-1, 1] as expected by the generator (Tanh output)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def denormalize(tensor):
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1)

def infer_and_display(model, bearded_img_path, clean_img_path, device):
    bearded_img = Image.open(bearded_img_path).convert("RGB")
    clean_img = Image.open(clean_img_path).convert("RGB")
    
    img_tensor = transform(bearded_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(img_tensor)
    output_tensor = denormalize(output_tensor.squeeze(0).cpu())
    output_img = transforms.ToPILImage()(output_tensor)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(bearded_img)
    plt.title("Bearded")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(output_img)
    plt.title("Inference")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(clean_img)
    plt.title("Clean (Ground Truth)")
    plt.axis("off")
    plt.show()

def run_inference(bearded_dir, clean_dir, checkpoint_path, num_examples=20, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    bearded_images = sorted(os.listdir(bearded_dir))[:num_examples]
    for img_name in bearded_images:
        bearded_img_path = os.path.join(bearded_dir, img_name)
        clean_img_name = img_name.replace("bearded", "clean")
        clean_img_path = os.path.join(clean_dir, clean_img_name)
        infer_and_display(model, bearded_img_path, clean_img_path, device)
