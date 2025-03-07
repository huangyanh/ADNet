import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"runs/train/mobilenetV4/weights/best.pt")

# Define input and output directories
input_folder = r"F:\Datasets\segmentation\final\images\test"
output_dir = r"F:\Datasets\segmentation\final\images\mobilenetV4"
os.makedirs(output_dir, exist_ok=True)

# Perform inference for all images in the input folder
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    results = model.predict(img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

    for idx, result in enumerate(results):  # Handle cases where results are a list
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            masks = result.masks
            confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []

            # Specify the classes to process
            selected_classes = [2]  # Class 2 (ship)

            filtered_boxes = []
            filtered_confidences = []
            filtered_class_ids = []

            # Filter boxes, confidences, and class IDs
            for i, cls_id in enumerate(class_ids):
                if cls_id in selected_classes:
                    filtered_boxes.append(boxes.xyxy[i].cpu().numpy())
                    filtered_confidences.append(confidences[i])
                    filtered_class_ids.append(cls_id)

            # Visualization
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(image)  # Display the full image

            if masks:
                # Define a mapping of class IDs to colors (RGB values)
                class_color_map = {
                    4: (255, 0, 0),      # Red   obstacle
                    2: (0, 255, 0),      # Green   ship
                    3: (0, 0, 200),      # Dark Blue   land
                    0: (128, 0, 128),    # Purple    sky 75, 0, 130
                    5: (255, 165, 0),    # Orange
                    8: (128, 128, 128),  # Grey
                    1: (100, 255, 255),  # Pink   water
                    7: (0, 255, 255),    # Cyan
                    6: (255, 255, 255),  # White
                    9: (169, 169, 169)   # Dark Grey

                }

                transparency = 0.9  # Set transparency to a fixed value

                # Create an overlay to accumulate masks
                overlay = np.zeros_like(image, dtype=np.float32)

                for i, mask in enumerate(masks.data.cpu().numpy()):
                    if np.any(mask):  # Ensure the mask contains non-zero values
                        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask_resized = (mask_resized > 0).astype(np.uint8)  # Convert to binary mask

                        # Retrieve class color, defaulting to gray if unknown
                        class_id = class_ids[i] if i < len(class_ids) else max(class_color_map.keys())
                        color = class_color_map.get(class_id, (128, 128, 128))  # Default to Grey if class not found

                        # Expand dimensions to match image shape (H, W, 3)
                        colored_mask = np.zeros_like(image, dtype=np.uint8)
                        for c in range(3):  # Apply color per channel
                            colored_mask[:, :, c] = mask_resized * color[c]

                        # Add mask with transparency
                        overlay = cv2.addWeighted(overlay, 1.0, colored_mask.astype(np.float32), transparency, 0)

                # Prevent color overflow by clamping values
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)

                # Blend overlay with original image
                blended_image = cv2.addWeighted(image, 1, overlay, transparency, 0)

                # Display the blended image
                ax.imshow(blended_image)

            # Overlay bounding boxes for selected classes
            for box, conf, cls_id in zip(filtered_boxes, filtered_confidences, filtered_class_ids):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', fill=False, linewidth=4)
                ax.add_patch(rect)

                # Add class ID and confidence text
                class_name = result.names[cls_id] if hasattr(result, 'names') and cls_id in result.names else str(cls_id)
                label = f"{class_name}: {conf:.2f}"
                ax.text(
                    x1, y1 - 10,
                    label,
                    color='white',
                    fontsize=20,
                    bbox=dict(facecolor='red', edgecolor='red', alpha=0.5, boxstyle='round,pad=0.3')
                )

            # Remove axes for better visualization
            ax.axis('off')
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)

            # Save the image
            output_path = os.path.join(output_dir, filename)
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Close the figure to release memory

            print(f"Image saved to {output_path}")
