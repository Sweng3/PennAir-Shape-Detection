"""
STATIC IMAGE RESULTS REPORT

Approach: Shapes are detected using local pixel-to-pixel variance
rather than color. Every shape is rendered as a smooth surface (local
variance is near zero) as compared to the textured background (local 
variance is high). The shapes' contours are then drawn, smoothed out, and
filtered. 

Challenges: The first working version used color, but wasn't good since 
color distance was sensitive to gradients and lighting so making it work
was difficult. 
"""

import cv2
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

cv2.imwrite("images/part1_output.png", output)
print("Saved result to images/part1_output.png")