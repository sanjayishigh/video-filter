import cv2
import time

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # CRITICAL FIX: 640x480 is 3x faster than 720p
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.prev_time = 0
        self.curr_time = 0
        self.fps = 0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        
        # FPS Calc
        self.curr_time = time.time()
        if self.curr_time - self.prev_time > 0:
            self.fps = 1 / (self.curr_time - self.prev_time)
        self.prev_time = self.curr_time
        
        return success, image

    def get_fps(self):
        return self.fps

cam_instance = VideoCamera()