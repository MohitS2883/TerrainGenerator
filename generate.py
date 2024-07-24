import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from PIL import Image
import uuid


def gen():
    latent_size = 128
    generator = nn.Sequential(
        # in: latent_size x 1 x 1
        nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(True),
        # out: 1024 x 4 x 4

        nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        # out: 512 x 8 x 8

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        # out: 256 x 16 x 16

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        # out: 128 x 32 x 32

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        # out: 64 x 64 x 64

        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
        # out: 3 x 128 x 128
    )

    # Load the generator model
    generator.load_state_dict(torch.load('allG.pth'))
    generator.to('cuda')  # Move the model to GPU

    # Set the model to evaluation mode
    generator.eval()

    # Generate an image
    latent_size = 128  # Define the latent size if not already defined
    latent_vector = torch.randn(1, latent_size, 1, 1, device='cuda')  # Generate a random noise vector on GPU
    generated_image = generator(latent_vector)  # Generate an image

    # Convert the generated image tensor to a NumPy array
    generated_image = generated_image.cpu().detach().numpy()  # Move the tensor to CPU and detach

    # Reshape and scale the image from [-1, 1] to [0, 1]
    generated_image = (generated_image + 1) / 2.0
    generated_image = np.transpose(generated_image[0], (1, 2, 0))  # Convert to HWC format
    os.makedirs('static/terraingenerate', exist_ok=True)
    image_filename = f"terraingenerate_{uuid.uuid4().hex}.png"
    image_path = os.path.join('static/terraingenerate', image_filename)

        # Save the image
    image = Image.fromarray((generated_image * 255).astype(np.uint8))
    image_filename = f"terraingenerate_{uuid.uuid4().hex}.png"
    image_path = os.path.join('static/terraingenerate', image_filename)
    image.save(image_path)
    return image_path
