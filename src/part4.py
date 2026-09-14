import cv2
import time
from shape_detector import detect_shapes, draw_shapes, estimate_3d_positions
from video_writer import FFmpegVideoWriter

K = [[2564.3186869, 0, 0],
     [0, 2569.70273111, 0],
     [0, 0, 1]]
CIRCLE_RADIUS_INCHES = 10.0

INPUT_VIDEO = "videos/PennAir 2024 App Dynamic Hard.mp4"
OUTPUT_VIDEO = "videos/part4_output.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video at {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = FFmpegVideoWriter(OUTPUT_VIDEO, fps, width, height)

frame_count = 0
frames_with_depth = 0
total_process_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    
    shapes = detect_shapes(frame, min_area=2000, min_solidity=0.6)
    shapes = estimate_3d_positions(shapes, K, CIRCLE_RADIUS_INCHES)
    annotated = draw_shapes(frame, shapes, show_3d=True)
    total_process_time += time.time() - start

    have_depth = any("position_3d" in s for s in shapes)
    if have_depth:
        frames_with_depth += 1

    cv2.putText(
        annotated, f"Shapes detected: {len(shapes)}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )

    # visually obvious when depth could not be computed
    if not have_depth:
        cv2.putText(
            annotated, "No depth reference (circle not visible)", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
        )

    writer.write(annotated)
    frame_count += 1

    if frame_count % 200 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
writer.release()

avg_ms = (total_process_time / frame_count) * 1000
effective_fps = frame_count / total_process_time
print(f"\nDone. Processed {frame_count} frames.")
print(f"\nFrames with depth : {frames_with_depth} frames.")
print(f"Average detection time per frame: {avg_ms:.2f} ms "
      f"({effective_fps:.1f} FPS -- source video is {fps:.1f} FPS)")
print(f"Saved annotated video to {OUTPUT_VIDEO}")