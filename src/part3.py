import cv2
import time
from shape_detector import detect_shapes, draw_shapes
from video_writer import FFmpegVideoWriter

INPUT_VIDEO = "videos/PennAir 2024 App Dynamic Hard.mp4"
OUTPUT_VIDEO = "videos/part3_output.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video at {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = FFmpegVideoWriter(OUTPUT_VIDEO, fps, width, height)

frame_count = 0
total_process_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    
    shapes = detect_shapes(frame, min_area=2000, min_solidity=0.6)
    annotated = draw_shapes(frame, shapes)
    total_process_time += time.time() - start

    cv2.putText(
        annotated, f"Shapes detected: {len(shapes)}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
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
print(f"Average detection time per frame: {avg_ms:.2f} ms "
      f"({effective_fps:.1f} FPS -- source video is {fps:.1f} FPS)")
print(f"Saved annotated video to {OUTPUT_VIDEO}")