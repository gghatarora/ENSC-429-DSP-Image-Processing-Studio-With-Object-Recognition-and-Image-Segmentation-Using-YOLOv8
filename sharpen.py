import cv2
import numpy as np

def sharpen_image(image):
    """
    Apply sharpening to the input image.
    
    Parameters:
    - image: The input image to be sharpened.
    
    Returns:
    - The sharpened image.
    """
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened
