import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageMode
from torchvision import transforms as T

# Load a sample image
image_path = "./dataset/test/Anglo_Arms_95_Fixed_Blade.png"
image = Image.open(image_path)

# Define a list of data augmentation transformations
augmentations = [
    T.ColorJitter(brightness=0.4, contrast=0, saturation=0.2, hue=0),
    T.RandomRotation(degrees=(0, 180)),
    T.RandomVerticalFlip(p=1),
    T.RandomHorizontalFlip(p=1),
]

# Visualize original image
plt.figure(figsize=(10, 10))
plt.subplot(3, 3, 1)
plt.imshow(image)
plt.title("Original Image")

# Apply and visualize each augmentation
for i, augmentation in enumerate(augmentations):
    augmented_image = augmentation(image)
    plt.subplot(3, 3, i + 2)
    plt.imshow(augmented_image)
    plt.title(f"{augmentation.__class__.__name__}")

plt.tight_layout()
plt.show()
