import cv2

def get_roi(frame, box):
    x, y, w, h = box
    return frame[y:y+h, x:x+w]

def draw_info(frame, box, label):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)    
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame