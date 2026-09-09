import cv2
import numpy as np

# detects shapes in a single image/frame using texture segmentation
def detect_shapes(frame, min_area=2000, var_ksize=5, var_thresh=20, min_solidity=0.85):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # calculate local variance
    mean = cv2.blur(gray, (var_ksize, var_ksize))
    sq_mean = cv2.blur(gray * gray, (var_ksize, var_ksize))
    variance = np.clip(sq_mean - mean * mean, 0, None) # high variance = textured, low variance = smooth

    mask = (variance < var_thresh).astype(np.uint8) * 255

    # clean up the binary mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2) # removes small isolated noise
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2) # fills small internal gaps

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= min_area:
            continue

        # helps makes countour lines smoother
        perimeter = cv2.arcLength(contour, True)
        smoothed = cv2.approxPolyDP(contour, 0.004 * perimeter, True)
        smoothed_area = cv2.contourArea(smoothed)
        if smoothed_area <= min_area:
            continue

        # checks for solid/compact shape
        hull_area = cv2.contourArea(cv2.convexHull(smoothed))
        solidity = smoothed_area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        M = cv2.moments(smoothed)
        if M["m00"] == 0:
            continue

        # finding centroid coordinates
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        shapes.append({"contour": smoothed, "center": (cx, cy), "area": smoothed_area})

    return shapes

# draw outlines and center dots for detected shapes
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