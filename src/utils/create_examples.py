# create_examples.py
import torch
import torchvision
import os
from PIL import Image

def create_example_images():
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    
    # Load MNIST test set
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True
    )
    
    # Save one example for digits 0-9
    for digit in range(10):
        # Find first occurrence of each digit
        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            if label == digit:
                img.save(f"examples/digit_{digit}.jpg")
                break

# Run if you want to create example images
if __name__ == "__main__":
    create_example_images()