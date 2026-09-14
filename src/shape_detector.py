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

        smoothed_perimeter = cv2.arcLength(smoothed, True)
        circularity = ( # more circle -> higher value
            4 * np.pi * smoothed_area / (smoothed_perimeter ** 2)
            if smoothed_perimeter > 0 else 0
        )
 
        shapes.append({
            "contour": smoothed,
            "center": (cx, cy),
            "area": smoothed_area,
            "circularity": circularity,
            "vertices": len(smoothed),
        })

    return shapes

def estimate_3d_positions(shapes, K, reference_radius, min_circularity=0.85):
    if not shapes:
        return shapes

    # finding the circle
    circle = max(shapes, key=lambda s: s.get("circularity", 0))
    if circle.get("circularity", 0) < min_circularity:
        return shapes
 
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
 
    apparent_radius_px = np.sqrt(circle["area"] / np.pi)
    focal_length_px = (fx + fy) / 2  # average the two axes' focal lengths
    depth_z = focal_length_px * reference_radius / apparent_radius_px
 
    for shape in shapes:
        u, v = shape["center"]
        x = (u - cx) * depth_z / fx
        y = (v - cy) * depth_z / fy
        shape["position_3d"] = (x, y, depth_z)
 
    return shapes

# draw outlines and center dots for detected shapes
def draw_shapes(frame, shapes, outline_color=(0, 255, 0), center_color=(0, 0, 255), show_3d=False):
    output = frame.copy()
    for shape in shapes:
        cv2.drawContours(output, [shape["contour"]], -1, outline_color, 3)
        cx, cy = shape["center"]
        cv2.circle(output, (cx, cy), 6, center_color, -1)
        
        position_3d = shape.get("position_3d")
        if show_3d and position_3d is not None:
            x, y, z = position_3d
            label = f"({x:.1f}, {y:.1f}, {z:.1f})"
        else:
            label = f"({cx},{cy})"

        cv2.putText(
            output, label, (cx + 12, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    return output