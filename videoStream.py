import cv2
from dotenv import load_dotenv
import os

load_dotenv()



class VideoStream:
    def __init__(self):
        self.camIn = os.getenv('CAM_IN')
        self.cap = cv2.VideoCapture(self.camIn)

    def readFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            print('No frame')
            return None
        return frame
    
    def release(self):
        self.cap.release()
        
    def is_open(self):
        return self.cap.isOpened()
    
        
        
        
