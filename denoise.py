import cv2
import numpy as np

def adaptive_denoise_image(image, mask=None, h_base=1, templateWindowSize=7, searchWindowSize=21):
    """
    Apply Non-Local Means Denoising to the image with adaptive filtering.
    
    Parameters:
    - image: The input image to be denoised.
    - mask: Optional mask to apply denoising only to the masked region.
    - h_base: Base parameter regulating filter strength. The actual 'h' will be scaled adaptively.
    - templateWindowSize: Size in pixels of the window used to compute weighted average for given pixel.
    - searchWindowSize: Size in pixels of the window used to search for pixels with similar intensity.
    
    Returns:
    - Denoised image.
    """
    # Convert image to grayscale for variance calculation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate local variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate adaptive h parameter based on variance
    h_adaptive = h_base * (1.0 + variance / 1000.0)
    
    if mask is not None:
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8) * 255
        # Apply Non-Local Means denoising only to the masked region with adaptive h
        denoised_masked_region = cv2.fastNlMeansDenoisingColored(image, None, h_adaptive, h_adaptive, templateWindowSize, searchWindowSize)
        denoised_image = np.where(mask[..., None] == 255, denoised_masked_region, image)
    else:
        # Apply Non-Local Means denoising to the entire image with adaptive h
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h_adaptive, h_adaptive, templateWindowSize, searchWindowSize)
    
    return denoised_image



# import cv2
# import numpy as np

# def denoise_image(image, mask=None, h=10, templateWindowSize=7, searchWindowSize=21):
#     """
#     Apply Non-Local Means Denoising to the image.
    
#     Parameters:
#     - image: The input image to be denoised.
#     - mask: Optional mask to apply denoising only to the masked region.
#     - h: Parameter regulating filter strength. A higher value removes noise better but also removes image details.
#     - templateWindowSize: Size in pixels of the window used to compute weighted average for given pixel.
#     - searchWindowSize: Size in pixels of the window used to search for pixels with similar intensity.
    
#     Returns:
#     - Denoised image.
#     """
#     if mask is not None:
#         # Create a copy of the image to avoid modifying the original
#         denoised_image = image.copy()
#         # Apply Non-Local Means denoising only to the masked region
#         denoised_masked_region = cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)
#         denoised_image = np.where(mask[..., None] == 255, denoised_masked_region, image)
#     else:
#         # Apply Non-Local Means denoising to the entire image
#         denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)
    
#     return denoised_image
