import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Generate a Gaussian kernel
def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-np.power((kernel_1D[i]) / sigma, 2) / 2)

    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D /= kernel_2D.sum()

    return kernel_2D

# Apply a filter to an image using convolution.
def apply_filter(image, kernel):
    image_blurred = cv2.filter2D(image, -1, kernel)
    return image_blurred

def gaussian_blur(image, mask, kernel_size=5, sigma=1):
    """Apply Gaussian blur to the areas of the image specified by the mask."""
    # Ensure the kernel size is positive and odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Generate Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Create a blurred version of the image
    blurred_image = apply_filter(image, kernel)

    # Apply the mask to blend the blurred and original images
    result = np.where(mask[..., None] == 255, blurred_image, image)

    return result