import cv2
import numpy as np

from shape_detector import detect_shapes, draw_shapes

# Load the image
image_path = "images/PennAir 2024 App Static.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")
shapes = detect_shapes(image)
print(f"Found {len(shapes)} shapes.")
for i, shape in enumerate(shapes, start=1):
    print(f"  Shape {i}: center={shape['center']}, area={shape['area']:.0f}px")

output = draw_shapes(image, shapes)