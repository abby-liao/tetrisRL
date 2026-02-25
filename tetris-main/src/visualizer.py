import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from subprocess import Popen, PIPE
import os
import imageio_ffmpeg

tf1.disable_eager_execution()

class TetrisVideoRecorder:
    def __init__(self, tb_log_dir, video_trigger, fps=15):
        self.video_trigger = video_trigger
        self.fps = fps
        self.file_writer = tf1.summary.FileWriter(tb_log_dir)
        self._recorded_frames = []

    def _encode_gif(self, frames):
        h, w, c = frames[0].shape
        pxfmt = {1: 'gray', 3: 'rgb24'}[c]
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        cmd = [
            ffmpeg_exe, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-r', str(self.fps), '-s', f'{w}x{h}', '-pix_fmt', pxfmt, '-i', '-',
            '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
            '-r', str(self.fps), '-f', 'gif', '-'
        ]
        
        proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        for image in frames:
            proc.stdin.write(image.tobytes())
        out, err = proc.communicate()
        if proc.returncode:
            raise IOError(f"FFmpeg error: {err.decode('utf8')}")
        return out

    def record_frame(self, frame):
        self._recorded_frames.append(frame)

    def finalize_video(self, tag, step):
        if not self._recorded_frames:
            return
        
        gif_data = self._encode_gif(self._recorded_frames)
        
        summary = tf1.Summary()
        h, w, c = self._recorded_frames[0].shape
        image = tf1.Summary.Image(height=h, width=w, colorspace=c)
        image.encoded_image_string = gif_data
        
        summary.value.add(tag=tag, image=image)
        self.file_writer.add_summary(summary, step)
        self.file_writer.flush()
        
        self._recorded_frames = []
        print(f"Successfully logged GIF for episode {step} to IMAGES tab.")
