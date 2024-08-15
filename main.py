import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
from gblur import gaussian_blur
from sharpen import sharpen_image
from denoise import adaptive_denoise_image
from histogram_equalization import histogram_equalization
from edge_detection import edge_detection

# How many items you want to detect (can be configured)
MAX_ITEMS = 15

# Load YOLOv8 segmentation model
model = YOLO('yolov8m-seg')

# Initialize the main window
root = ctk.CTk()
root.title("Image Viewer")
root.geometry("1400x800")

current_image_path = None  # Variable to store the path of the currently loaded image
cv_image = None  # Variable to store the OpenCV image
masks = {}  # Dictionary to store masks of detected items
pil_image_with_boxes = None  # Global variable for original image with boxes
pil_image_modified = None  # Global variable for modified image

# Function to capture an image using OpenCV
def capture_image():
    global current_image_path
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cap.release()
        current_image_path = "captured_image.jpg"
        cv2.imwrite(current_image_path, frame)  # Save the captured frame as is
        process_and_display_image(frame)  # Pass the frame directly

# Function to open and display an image from local files
def open_image():
    global current_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        current_image_path = file_path
        img = cv2.imread(file_path)
        process_and_display_image(img)

# Function to process and display the image
def process_and_display_image(image):
    global cv_image, masks, pil_image_with_boxes, pil_image_modified
    cv_image = image

    # Convert the image to RGB format for PIL
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image_rgb)

    # Run segmentation using YOLOv8
    results = model(cv_image_rgb)
    detected_items = ["Entire Image", "Background"]
    masks = {}

    # Create a mask for the whole image
    height, width = cv_image.shape[:2]
    full_mask = np.ones((height, width), dtype=np.uint8) * 255 

    # Process YOLOv8 segmentation results if available
    if results[0].masks is not None:
        for i, (box, mask) in enumerate(zip(results[0].boxes, results[0].masks.data)):
            label = f"Item {i+1} - {results[0].names[int(box.cls)]}"
            detected_items.append(label)
            if len(detected_items) - 1 >= MAX_ITEMS:
                break

            # Resize mask to match the image dimensions
            item_mask = cv2.resize(mask.cpu().numpy().astype(np.uint8), (width, height)) * 255

            # Refine mask using erosion and dilation operations
            kernel = np.ones((3, 3), np.uint8)
            refined_item_mask = cv2.erode(item_mask, kernel, iterations=2)
            refined_item_mask = cv2.dilate(refined_item_mask, kernel, iterations=2)

            # Subtract the refined item mask from the full mask to exclude detected items from the background
            full_mask = cv2.subtract(full_mask, refined_item_mask)

            # Store the refined mask
            masks[label] = refined_item_mask

        # Store the background mask
        masks["Background"] = full_mask

        # Draw the bounding boxes and labels on the original image
        pil_image_with_boxes = pil_image.copy()
        draw = ImageDraw.Draw(pil_image_with_boxes)
        for i, (box, mask) in enumerate(zip(results[0].boxes, results[0].masks.data)):
            label = f"Item {i+1} - {results[0].names[int(box.cls)]}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
            draw.text((x1, y1 - 10), label, fill="green")

    else:
        pil_image_with_boxes = pil_image.copy()

    # initialize a copy of the modified image.
    pil_image_modified = pil_image.copy()

    # Convert back to PIL for displaying
    pil_image_with_boxes = resize_image_to_fit(pil_image_with_boxes, 400, 400)
    pil_image_modified = resize_image_to_fit(pil_image_modified, 400, 400)

    img_ctk_with_boxes = ctk.CTkImage(light_image=pil_image_with_boxes, dark_image=pil_image_with_boxes, size=pil_image_with_boxes.size)
    img_ctk_modified = ctk.CTkImage(light_image=pil_image_modified, dark_image=pil_image_modified, size=pil_image_modified.size)

    image_label_with_boxes.configure(image=img_ctk_with_boxes)
    image_label_with_boxes.image = img_ctk_with_boxes

    image_label_modified.configure(image=img_ctk_modified)
    image_label_modified.image = img_ctk_modified

    # Update the scroll list with detected items
    for widget in scroll_frame.winfo_children():
        widget.destroy()
    item_vars.clear()
    for item in detected_items:
        item_var = ctk.BooleanVar()
        item_vars[item] = item_var
        item_checkbox = ctk.CTkCheckBox(scroll_frame, text=item, variable=item_var)
        item_checkbox.pack(padx=10, pady=5, anchor="w")

# Function to resize image to fit the window
def resize_image_to_fit(image, max_width, max_height):
    original_width, original_height = image.size
    ratio = min(max_width / original_width, max_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Function to apply a transformation to the selected items
def apply_transformation():
    global pil_image_modified, cv_image
    if cv_image is None:
        print("No image loaded.")
        return

    selected_transformation = selected_transformation_var.get()
    selected_items = [item for item in item_vars if item_vars[item].get()]

    if not selected_items:
        return

    for item in selected_items:
        print(f"Applying {selected_transformation} to {item}")
        if item == "Entire Image":
            mask = np.ones(cv_image.shape[:2], dtype=np.uint8) * 255
        elif item in masks:
            mask = masks[item]
        else:
            continue

        if selected_transformation == "Gaussian Blur":
            blurred_image = gaussian_blur(cv_image, mask, kernel_size=15, sigma=5)
            cv_image = np.where(mask[..., None] == 255, blurred_image, cv_image)

        if selected_transformation == "Sharpen":
            sharpened_image = sharpen_image(cv_image)
            cv_image = np.where(mask[..., None] == 255, sharpened_image, cv_image)

        if selected_transformation == "Denoise":
            denoised_image = adaptive_denoise_image(cv_image, mask)
            cv_image = np.where(mask[..., None] == 255, denoised_image, cv_image)
            
        if selected_transformation == "Histogram Equalization":
            equalized_image = histogram_equalization(cv_image, mask)
            cv_image = np.where(mask[..., None] == 255, equalized_image, cv_image)
            
        if selected_transformation == "Edge Detection":
            edge_detected_image = edge_detection(cv_image, mask)
            cv_image = np.where(mask[..., None] == 255, edge_detected_image, cv_image)

    # Convert back to PIL for displaying
    cv_image_rgb_modified = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image_modified = Image.fromarray(cv_image_rgb_modified)
    pil_image_modified = resize_image_to_fit(pil_image_modified, 400, 400)

    img_ctk_modified = ctk.CTkImage(light_image=pil_image_modified, dark_image=pil_image_modified, size=pil_image_modified.size)
    image_label_modified.configure(image=img_ctk_modified)
    image_label_modified.image = img_ctk_modified

# Function to save the transformed image
def save_image():
    global pil_image_modified
    if pil_image_modified is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        if save_path:
            pil_image_modified.save(save_path)
            print(f"Image saved to {save_path}")

# Create the left frame for displaying the original image with bounding boxes
left_frame = ctk.CTkFrame(root, width=500, height=800, border_width=2)
left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

# Create a label for the original image with boxes
original_image_label = ctk.CTkLabel(left_frame, text="Original Image with Labels", font=("Helvetica", 16, "bold"), padx=10, pady=10)
original_image_label.pack(fill="x", pady=(10, 0))

# Create a label to display the original image with boxes
image_label_with_boxes = ctk.CTkLabel(left_frame, text="")
image_label_with_boxes.pack(expand=True, fill="both", padx=10, pady=10)

# Create the middle frame for displaying the modified image
middle_frame = ctk.CTkFrame(root, width=500, height=800, border_width=2)
middle_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

# Create a label for the modified image
modified_image_label = ctk.CTkLabel(middle_frame, text="Modified Image", font=("Helvetica", 16, "bold"), padx=10, pady=10)
modified_image_label.pack(fill="x", pady=(10, 0))

# Create a label to display the modified image
image_label_modified = ctk.CTkLabel(middle_frame, text="")
image_label_modified.pack(expand=True, fill="both", padx=10, pady=10)

# Create the right frame for the sidebar
right_frame = ctk.CTkFrame(root, width=300, height=800, border_width=2)
right_frame.pack(side="right", fill="y", padx=10, pady=10)

# Create buttons on the sidebar
capture_button = ctk.CTkButton(right_frame, text="Capture Image", command=capture_image)
capture_button.pack(padx=10, pady=10)

open_button = ctk.CTkButton(right_frame, text="Open Image", command=open_image)
open_button.pack(padx=10, pady=10)

# Scroll list of items with checkboxes
scroll_frame = ctk.CTkScrollableFrame(right_frame, width=300, height=200)
scroll_frame.pack(padx=10, pady=10, fill="both", expand=True)

item_vars = {}

# Scroll list of transformations with radio buttons
transformation_frame = ctk.CTkScrollableFrame(right_frame, width=300, height=100)
transformation_frame.pack(padx=10, pady=10, fill="both", expand=True)

selected_transformation_var = ctk.StringVar()
transformations = ["Gaussian Blur", "Sharpen", "Denoise", "Histogram Equalization", "Edge Detection"]
for transformation in transformations:
    transformation_radio = ctk.CTkRadioButton(transformation_frame, text=transformation, variable=selected_transformation_var, value=transformation)
    transformation_radio.pack(padx=10, pady=5, anchor="w")

# Create buttons to apply transformations
transform_button = ctk.CTkButton(right_frame, text="Apply Transformation", command=apply_transformation)
transform_button.pack(padx=10, pady=10)

# Create a button to save the transformed image
save_button = ctk.CTkButton(right_frame, text="Save Image", command=save_image)
save_button.pack(padx=10, pady=10)

# Start the main loop
root.mainloop()
