import os
import torch
import random
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch import autocast

class PairedFaceDatasetGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        self.model_id = model_id
        self.txt2img_pipe = self._setup_txt2img_pipeline()
        self.img2img_pipe = self._setup_img2img_pipeline()

    def _setup_txt2img_pipeline(self):
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        ).to(self.device)
        return pipeline

    def _setup_img2img_pipeline(self):
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        ).to(self.device)
        return pipeline

    def generate_base_face(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        prompt = ("a portrait photo of one adult male, the exact same face, the exact same hair, "
                  "clean-shaven, no beard, no mustache, smooth skin, "
                  "full head visible, front facing, centered, neutral expression, "
                  "high-resolution, uniform lighting, color photo.")
        negative_prompt = ("any facial hair, beard, mustache, stubble, shadow, goatee, patchy hair, "
                           "different face, different hair, changed features, multiple faces, multiple people, "
                           "blurry, distorted, cartoon, female, child, long hair, occlusions, sunglasses, masks, "
                           "makeup, accessories, vintage, hat, black and white.")
        with autocast(self.device):
            image = self.txt2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=40,
                guidance_scale=12.0,
                height=512,
                width=512
            ).images[0]
        return image

    def add_beard(self, base_image, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        prompt = ("a portrait photo of same adult male, same exact face, same exact hair, "
                  "now with full very very thick dense strong beard covering jawline, cheeks, chin, "
                  "strong beard with no gaps, well-groomed, "
                  "realistic, full head visible, front facing, centered, high-resolution.")
        negative_prompt = ("clean shaven, no beard, only mustache, thin beard, patchy beard, "
                           "gaps in beard, different face, different hair, changed features, "
                           "multiple people, multiple faces, cropped forehead, blurry, "
                           "distorted, cartoon, female, child, occlusions, sunglasses, "
                           "masks, makeup, accessories, hat, vintage, black and white.")
        with autocast(self.device):
            image = self.img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_image,
                strength=0.63,
                num_inference_steps=50,
                guidance_scale=13.0
            ).images[0]
        return image

    def generate_face_pair(self, seed=None):
        if seed is None:
            seed = random.randint(0, 999999)
        clean_shaven = self.generate_base_face(seed)
        bearded = self.add_beard(clean_shaven, seed)
        return clean_shaven, bearded

    def generate_dataset(self, num_pairs, output_dir="face_dataset"):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "bearded"), exist_ok=True)
        for i in tqdm(range(num_pairs)):
            seed = random.randint(0, 999999)
            clean, bearded = self.generate_face_pair(seed)
            clean.save(os.path.join(output_dir, "clean", f"face_{i:04d}_clean.png"))
            bearded.save(os.path.join(output_dir, "bearded", f"face_{i:04d}_bearded.png"))
            with open(os.path.join(output_dir, f"face_{i:04d}_metadata.txt"), "w") as f:
                f.write(f"Seed: {seed}\n")

    def preview_pairs(self, num_pairs=5):
        fig, axes = plt.subplots(num_pairs, 2, figsize=(10, num_pairs * 4))
        for i in range(num_pairs):
            seed = random.randint(0, 999999)
            clean, bearded = self.generate_face_pair(seed)
            axes[i, 0].imshow(clean)
            axes[i, 0].set_title(f"Pair {i+1}: Clean-shaven")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(bearded)
            axes[i, 1].set_title(f"Pair {i+1}: Bearded")
            axes[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

def preview_dataset(num_pairs=10, device="cuda"):
    generator = PairedFaceDatasetGenerator(device=device)
    generator.preview_pairs(num_pairs=num_pairs)

def generate_dataset(num_pairs=300, output_dir="synthetic_data_training", device="cuda"):
    generator = PairedFaceDatasetGenerator(device=device)
    generator.generate_dataset(num_pairs=num_pairs, output_dir=output_dir)

