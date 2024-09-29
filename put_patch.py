import cv2
import os
import numpy as np

# Load the patch image
patch_path = "images/patches/rtdetr.jpg"  # Update with your patch image path
patch_image = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if present

# Check if the patch is loaded correctly
if patch_image is None:
    raise ValueError(f"Failed to load patch image from {patch_path}")

print(f"Patch image loaded with shape: {patch_image.shape}")


# Set your directories
images_folder = "images/inria/exp/images"  # Update with your images folder path
labels_folder = "images/inria/exp/labels"  # Update with your labels folder path
output_folder = "images/inria/adv/atk_images/rtdetr_4th"  # Update with your output folder path
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

# Get all image and label filenames
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('.txt')])

if len(image_files) != len(label_files):
    raise ValueError("Mismatch between number of images and labels")


# Function to extract bounding boxes from a label file
def extract_bounding_boxes(label_file):
    with open(label_file, 'r') as f:
        boxes = []
        for line in f:
            values = line.strip().split()
            # Extract bounding box coordinates from label (assuming normalized format)
            class_id = int(values[0])
            cx, cy, w, h = map(float, values[1:])  # Normalize format: (class_id, cx, cy, w, h)
            boxes.append((cx, cy, w, h))
        return boxes


# Function to rescale and place patch in the center of the bounding box
def place_patch_in_bbox(image, patch, bbox, image_size):
    img_h, img_w = image_size
    cx, cy, bw, bh = bbox

    # Convert normalized coordinates to pixel values
    cx = int(cx * img_w)
    cy = int(cy * img_h)
    bw = int(bw * img_w)
    bh = int(bh * img_h)

    # Rescale the patch to 1/4 * height of the bounding box
    patch_size = int(bh / 4)
    resized_patch = cv2.resize(patch, (patch_size, patch_size))

    # Calculate the top-left corner of the patch placement
    start_x = max(0, cx - patch_size // 2)
    start_y = max(0, cy - patch_size // 2)

    # Ensure the patch fits within the image boundaries
    end_x = min(start_x + patch_size, img_w)
    end_y = min(start_y + patch_size, img_h)

    # Ensure patch dimensions align with bbox placement
    patch_w = end_x - start_x
    patch_h = end_y - start_y
    resized_patch = cv2.resize(patch, (patch_w, patch_h))

    # Overlay the patch on the image
    image[start_y:end_y, start_x:end_x] = resized_patch

    return image


# Iterate over images and labels, apply the patch and save the new images
for img_file, lbl_file in zip(image_files, label_files):
    # Load image and corresponding label
    img_path = os.path.join(images_folder, img_file)
    label_path = os.path.join(labels_folder, lbl_file)

    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image {img_path}")
        continue

    # Get image size
    img_h, img_w, _ = image.shape

    # Extract bounding boxes
    bounding_boxes = extract_bounding_boxes(label_path)

    # Place the patch on each bounding box
    for bbox in bounding_boxes:
        image = place_patch_in_bbox(image, patch_image, bbox, (img_h, img_w))

    # Save the modified image
    output_img_path = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_patched.png")
    cv2.imwrite(output_img_path, image)
    print(f"Saved patched image to {output_img_path}")
