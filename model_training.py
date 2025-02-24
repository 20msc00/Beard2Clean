import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = '/content/drive/MyDrive/synthetic_data_training'
BATCH_SIZE = 2
NUM_EPOCHS = 750
LEARNING_RATE = 0.0002
L1_LAMBDA = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_DIR = "./samples"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/model_checkpoints'
os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)

class PairedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.beard_dir = os.path.join(root_dir, "bearded")
        self.clean_dir = os.path.join(root_dir, "clean")
        beard_files = sorted(os.listdir(self.beard_dir))
        self.pairs = []
        for beard_file in beard_files:
            beard_base = beard_file.split("_bearded")[0]
            clean_file = beard_base + "_clean.png"
            if os.path.exists(os.path.join(self.clean_dir, clean_file)):
                self.pairs.append((beard_file, clean_file))
        if len(self.pairs) == 0:
            raise ValueError("No paired images found! Verify your dataset structure.")
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        beard_file, clean_file = self.pairs[idx]
        beard_path = os.path.join(self.beard_dir, beard_file)
        clean_path = os.path.join(self.clean_dir, clean_file)
        beard_img = Image.open(beard_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")
        if self.transform:
            beard_img = self.transform(beard_img)
            clean_img = self.transform(clean_img)
        return {"beard": beard_img, "clean": clean_img}

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = PairedDataset(DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)
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

class Discriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )
    def forward(self, img_A, img_B):
        x = torch.cat((img_A, img_B), 1)
        return self.model(x)

generator = GeneratorUNet().to(DEVICE)
discriminator = Discriminator().to(DEVICE)
criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

def real_labels(size):
    return torch.ones(size, device=DEVICE)

def fake_labels(size):
    return torch.zeros(size, device=DEVICE)

def save_sample(generator, sample_data, epoch, batch_idx):
    generator.eval()
    with torch.no_grad():
        beard = sample_data["beard"].to(DEVICE)
        real_clean = sample_data["clean"].to(DEVICE)
        fake_clean = generator(beard)
        def denorm(x):
            return (x + 1) / 2
        comparison = torch.cat([denorm(beard), denorm(fake_clean), denorm(real_clean)], dim=0)
        save_path = os.path.join(SAMPLE_DIR, f"epoch{epoch:03d}_batch{batch_idx:03d}.png")
        utils.save_image(comparison, save_path, nrow=BATCH_SIZE)
    generator.train()

def train_model():
    print("Starting Training ...")
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        for batch_idx, batch in enumerate(dataloader):
            beard = batch["beard"].to(DEVICE)
            clean = batch["clean"].to(DEVICE)
            optimizer_D.zero_grad()
            pred_real = discriminator(beard, clean)
            loss_D_real = criterion_GAN(pred_real, real_labels(pred_real.shape))
            fake_clean = generator(beard)
            pred_fake = discriminator(beard, fake_clean.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake_labels(pred_fake.shape))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            pred_fake = discriminator(beard, fake_clean)
            loss_G_GAN = criterion_GAN(pred_fake, real_labels(pred_fake.shape))
            loss_G_L1 = criterion_L1(fake_clean, clean)
            loss_G = loss_G_GAN + L1_LAMBDA * loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")
            if batch_idx % 100 == 0:
                save_sample(generator, batch, epoch, batch_idx)
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed/60:.2f} minutes.")
    final_checkpoint = {
        'epoch': NUM_EPOCHS,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    final_checkpoint_path = os.path.join(DRIVE_CHECKPOINT_DIR, "final_checkpoint.pth")
    torch.save(final_checkpoint, final_checkpoint_path)
    print("Final checkpoint saved to", final_checkpoint_path)

def test_model(num_examples=5):
    generator.eval()
    samples = []
    with torch.no_grad():
        for idx in range(min(num_examples, len(dataset))):
            sample = dataset[idx]
            beard = sample["beard"].unsqueeze(0).to(DEVICE)
            fake_clean = generator(beard)
            def denorm(x):
                return (x + 1) / 2
            beard_img = denorm(beard.squeeze(0).cpu())
            fake_img = denorm(fake_clean.squeeze(0).cpu())
            real_img = denorm(sample["clean"])
            comparison = torch.cat([beard_img, fake_img, real_img], dim=2)
            samples.append(comparison)
    fig, axes = plt.subplots(len(samples), 1, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]
    for ax, img in zip(axes, samples):
        ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    generator.train()
