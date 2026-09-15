Setup:
Requires Python 3 and ffmpeg installed

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Running each part:
Part 1: Reads "images/PennAir 2024 App Static.png", prints each detected shape's center and area, and saves it to "images/part1_output.png".

```bash
python3 src/part1-static.py
```

Part 2: Reads "videos/PennAir 2024 App Dynamic.mp4", processes it frame-by-frame, and writes the annotated result to "videos/part2_output.mp4". 

```bash
python3 src/part2-video.py
```

Part 3: Reads "videos/PennAir 2024 App Dynamic Hard.mp4" and outputs
it to "videos/part3_output.mp4".

```bash
python3 src/part3.py
```

Part 4: same as part 3 with depth, outputs "videos/part4_output.mp4"

```bash
python3 src/part4.py
```

Reports:
STATIC IMAGE RESULTS REPORT - in "part1-static.py"
VIDEO RESULTS REPORT - in "part2-video.py"
BACKGROUND AGNOSTIC RESULTS REPORT - in "part4.py"