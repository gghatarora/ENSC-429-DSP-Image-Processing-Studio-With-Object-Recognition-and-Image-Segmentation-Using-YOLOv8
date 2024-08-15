import cv2
import numpy as np

def edge_detection(image, mask=None):
    """
    Apply Canny edge detection to the image.
    
    Parameters:
    - image: The input image to detect edges.
    - mask: Optional mask to apply edge detection only to the masked region.
    
    Returns:
    - Edge-detected image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    
    if mask is not None:
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_detected_image = np.where(mask[..., None] == 255, edges_colored, image)
    else:
        edge_detected_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edge_detected_image
