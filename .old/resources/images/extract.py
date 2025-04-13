import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Define transformations for input image
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # Resize input image to 32x32
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # Normalize image tensors
    ]
)


# Function to convert image to tensor and save the flattened tensor as a text file
def convert_to_tensor(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    # Flatten the image tensor to a 1D array
    flattened_tensor = image_tensor.view(-1).numpy()
    # Generate output filename
    output_filename = (
        "flattened_" + os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    )
    # Save the flattened tensor as a text file
    np.savetxt(output_filename, flattened_tensor)


# Example usage
image_path = "../images/image_1.png"  # Path to your image
convert_to_tensor(image_path)
print(
    f"Flattened tensor saved as flattened_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
)
