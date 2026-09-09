import subprocess

class FFmpegVideoWriter:
    def __init__(self, path, fps, width, height):
        self.width = width
        self.height = height
        command = [
            "ffmpeg",
            "-y",  # overwrite the output file without prompting
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",  # matches OpenCV's in-memory frame format
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",  # read raw frames from standard input
            "-an",  # no audio
            "-c:v", "libx264",
            "-preset", "ultrafast", # fast encoding
            "-pix_fmt", "yuv420p", 
            "-movflags", "+faststart", 
            path,
        ]

        # separate running process
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def release(self):
        self.process.stdin.close()
        stderr_output = self.process.stderr.read()
        return_code = self.process.wait()
        
        # when an error occurs
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {return_code}:\n{stderr_output.decode(errors='replace')}"
            )