import cv2
from shape_detector import detect_shapes, draw_shapes
from video_writer import FFmpegVideoWriter

INPUT_VIDEO = "videos/PennAir 2024 App Dynamic.mp4"
OUTPUT_VIDEO = "videos/part2_output.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video at {INPUT_VIDEO}")

#video properties so output video matches
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = FFmpegVideoWriter(OUTPUT_VIDEO, fps, width, height)

frame_count = 0

while True:
    #check if video runs out of frames
    ret, frame = cap.read()
    if not ret:
        break

    shapes = detect_shapes(frame, min_area=2000, min_solidity=0.6)
    annotated = draw_shapes(frame, shapes)

    #live shape count on frame
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

print(f"\nDone. Processed {frame_count} frames.")
print(f"Saved annotated video to {OUTPUT_VIDEO}")