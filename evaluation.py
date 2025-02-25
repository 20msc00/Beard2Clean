import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import matplotlib.pyplot as plt

# Reuse the same transformation as in inference.py
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def denormalize(tensor):
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1)

def evaluate_model(bearded_dir, clean_dir, checkpoint_path, num_examples=20, device="cuda"):
    """
    Evaluate the model performance on the test dataset using SSIM, PSNR, and LPIPS.
    Displays a table of results for each image pair.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Import load_model from inference.py for model loading.
    from inference import load_model
    model = load_model(checkpoint_path, device)
    
    # Initialize LPIPS using the AlexNet backbone.
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    bearded_images = sorted(os.listdir(bearded_dir))[:num_examples]
    results = []
    
    for img_name in bearded_images:
        bearded_img_path = os.path.join(bearded_dir, img_name)
        clean_img_name = img_name.replace("bearded", "clean")
        clean_img_path = os.path.join(clean_dir, clean_img_name)
        
        # Load images using PIL.
        bearded_img = Image.open(bearded_img_path).convert("RGB")
        clean_img = Image.open(clean_img_path).convert("RGB")
        
        # Run inference on the bearded image.
        input_tensor = transform(bearded_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        output_tensor = denormalize(output_tensor.squeeze(0).cpu())
        output_img = transforms.ToPILImage()(output_tensor)
        
        # Convert images to numpy arrays for SSIM and PSNR evaluation.
        output_np = np.array(output_img)
        clean_np = np.array(clean_img)
        
        # Compute SSIM with explicit win_size=7 and channel_axis=2.
        ssim_val = ssim(clean_np, output_np, win_size=7, channel_axis=2)
        psnr_val = psnr(clean_np, output_np, data_range=output_np.max() - output_np.min())
        
        # Compute LPIPS using torch tensors (inputs are expected in the range [-1, 1]).
        clean_tensor = transform(clean_img).unsqueeze(0).to(device)
        lpips_val = loss_fn_alex(output_tensor.unsqueeze(0).to(device), clean_tensor).item()
        
        results.append({
            "Image": img_name,
            "SSIM": ssim_val,
            "PSNR": psnr_val,
            "LPIPS": lpips_val
        })
    
    df = pd.DataFrame(results)
    print(df)
