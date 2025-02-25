# Beard2Clean

Beard2Clean is a project that leverages generative AI to create a synthetic paired dataset of faces—each pair consisting of a bearded version and a clean-shaven version of the same individual. The project is divided into two main stages:

Synthetic Dataset Creation:
Using Stable Diffusion, the system generates paired images with a consistent facial identity, differing only in the presence or absence of a beard. The challenge here was to achieve high similarity between the paired images despite the inherent randomness of generative models. Extensive prompt engineering and careful tuning of the parameters were employed to overcome issues such as multiple faces, incomplete features, or inconsistent accessories.

Image-to-Image Model Training:
A lightweight image-to-image model (based on a U-Net architecture) is trained to transform bearded images into clean-shaven ones. The model is optimized for limited computational resources, and its performance is evaluated using SSIM, PSNR, and LPIPS metrics to ensure both structural fidelity and perceptual quality.

For further explanations on model evaluation, encountered challenges, and future improvements, please refer to the corresponding sections in the Google Colab Notebook.

## Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fg2Y9crEt025oFEJmai923WUO49pq0WR?usp=sharing)


## Setup and Installation (Local)

If you prefer to run the project locally (e.g., using VS Code or Jupyter Notebook), follow these steps:

1. **Clone the Repository:**

   Open a terminal and run:

   ```bash
   git clone https://github.com/<your_username>/Beard2Clean.git
   cd Beard2Clean

2. **Install Dependencies:**

   You have two options to set up the environment:

   - **Using pip:**

     Install all required Python packages with:

     ```bash
     pip install -r requirements.txt
     ```

   - **Using Conda:**

     If you prefer using Conda for environment management, create a new environment using the provided `environment.yml` file:

     ```bash
     conda env create -f environment.yml
     conda activate Beard2Clean
     ```

3. **Open the Notebook::**
   
   You can download the Beard2Clean.ipynb notebook using the Google Colab link above, place it in the repository folder, and then run the following command to launch it locally:

   ```bash
   jupyter notebook Beard2Clean.ipynb
