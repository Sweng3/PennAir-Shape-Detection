import cv2
import numpy as np


def detect_shapes(frame, min_area=500, blur_ksize=(15, 15), percentile=85, safety_margin=1.3, detect_width=None):
    scale = 1.0
    original_frame = frame
    if detect_width is not None and frame.shape[1] > detect_width:
        scale = detect_width / frame.shape[1]
        frame = cv2.resize(frame, (detect_width, int(frame.shape[0] * scale)))

    blurred = cv2.GaussianBlur(frame, blur_ksize, 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

    #background color estimate
    bg_mean = np.median(lab.reshape(-1, 3), axis=0)

    dist = np.sqrt(((lab - bg_mean) ** 2).sum(axis=2))

    threshold = np.percentile(dist, percentile) * safety_margin
    mask = (dist > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scaled_min_area = min_area * (scale ** 2)

    #finding shapes
    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= scaled_min_area:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        #scaling it back up
        if scale != 1.0:
            contour = (contour / scale).astype(np.int32)
            cx, cy = int(cx / scale), int(cy / scale)
            area = area / (scale ** 2)

        shapes.append({"contour": contour, "center": (cx, cy), "area": area})

    return shapes

#draw outlines and centers
def draw_shapes(frame, shapes, outline_color=(0, 255, 0), center_color=(0, 0, 255)):
    output = frame.copy()
    for shape in shapes:
        cv2.drawContours(output, [shape["contour"]], -1, outline_color, 3)
        cx, cy = shape["center"]
        cv2.circle(output, (cx, cy), 6, center_color, -1)
        cv2.putText(
            output, f"({cx},{cy})", (cx + 12, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    return output