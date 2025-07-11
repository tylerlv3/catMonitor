import numpy as np
import mss
import cv2


class ScreenStream:
    def __init__(self):
        self.mss = mss.mss()
        self.monitor = self.mss.monitors[1]

    def readFrame(self):
        try:
            img = self.mss.grab(self.monitor)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
    
    def release(self):
        self.mss.close()
            
            
        
        