import cv2
import numpy as np

def histogram_equalization(image, mask=None):
    """
    Apply histogram equalization to the image.
    
    Parameters:
    - image: The input image to be equalized.
    - mask: Optional mask to apply equalization only to the masked region.
    
    Returns:
    - Equalized image.
    """
    if mask is not None:
        # Create a copy of the image to avoid modifying the original
        equalized_image = image.copy()
        # Convert to YUV color space
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Equalize the Y channel
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
        equalized_yuv = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
        equalized_image = np.where(mask[..., None] == 255, equalized_yuv, image)
    else:
        # Convert to YUV color space
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Equalize the Y channel
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
        equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    
    return equalized_image
