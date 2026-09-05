import cv2
import numpy as np

# Load the image
image_path = "images/PennAir 2024 App Static.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

#blur grass texture
blurred = cv2.GaussianBlur(image, (15, 15), 0)

#convert to lab
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
h, w = image.shape[:2]
patch = 40

#compute avg background color
corners = [
    lab[0:patch, 0:patch],
    lab[0:patch, w - patch:w],
    lab[h - patch:h, 0:patch],
    lab[h - patch:h, w - patch:w],
]
bg_samples = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
bg_mean = bg_samples.mean(axis=0)

#color distances
dist = np.sqrt(((lab - bg_mean) ** 2).sum(axis=2))
bg_dist = np.sqrt(((bg_samples - bg_mean) ** 2).sum(axis=1))
noise_ceiling = np.percentile(bg_dist, 99.9)  # background's worst-case noise
threshold = noise_ceiling * 2  # safety margin above that noise
mask = (dist > threshold).astype(np.uint8) * 255

#morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

#contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#leftover noise
MIN_CONTOUR_AREA = 500
real_shapes = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
print(f"Found {len(contours)} raw contours, {len(real_shapes)} real shapes after filtering.")

#draw outlines and centers
output = image.copy()
for contour in real_shapes:
    cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
    
    M = cv2.moments(contour)
    if M["m00"] == 0:
        continue  # skip degenerate contours
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)
    cv2.putText(
        output, f"({cx},{cy})", (cx + 12, cy),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )