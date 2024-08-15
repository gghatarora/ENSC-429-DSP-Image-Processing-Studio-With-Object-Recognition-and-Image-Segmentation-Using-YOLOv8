# DSP Image Processing Studio

## Overview

DSP Image Processing Studio is an advanced image processing application that leverages the YOLOv8 model for object recognition and segmentation. The application allows users to capture or upload images, identify and segment objects, and apply various transformations to the selected objects. The user interface is built with custom-tkinter, providing a minimal, intuitive, and easy-to-use layout.

## Features

- **Object Recognition and Segmentation**: Uses the YOLOv8 model to detect and segment objects within an image.
- **Morphological Operations**: Refines segmentation masks using erosion and dilation for improved accuracy.
- **Targeted Transformations**: Allows selective application of transformations such as Gaussian blur, sharpening, denoising, histogram equalization, and edge detection.
- **User Interface**: Provides a clear distinction between the original image with bounding boxes and the modified image after transformations.
- **Save Functionality**: Users can save the transformed images for later use.

## Installation

1. **Install Dependencies**:
    Make sure you have Python 3.9 or higher installed. Install the required packages using pip:
    ```
    pip install -r requirements.txt
    ```

2. **Ensure OpenCV and YOLOv8 are Installed**:
    Install OpenCV:
    ```
    pip install opencv-python
    ```
    Install Ultralytics YOLO:
    ```
    pip install ultralytics
    ```

## Usage

1. **Run the Application**:
    ```
    python main.py
    ```

2. **Capture or Open an Image**:
    - Use the "Capture Image" button to capture an image from your webcam.
    - Use the "Open Image" button to load an image from your local files.

3. **View Detected Objects**:
    - The application will display the original image with bounding boxes and labels for each detected object.
    - A list of detected items will be shown in the sidebar with checkboxes for selection.

4. **Apply Transformations**:
    - Select the desired objects from the list.
    - Choose a transformation from the options provided (Gaussian Blur, Sharpen, Denoise, Histogram Equalization, Edge Detection).
    - Click "Apply Transformation" to apply the selected transformation to the chosen objects.

5. **Save the Transformed Image**:
    - Use the "Save Image" button to save the modified image to your local files.
