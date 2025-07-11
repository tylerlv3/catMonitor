import cv2


camIn = 'rtsp://192.168.1.100:8554/test'

cap = cv2.VideoCapture(camIn)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frame')
        break
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()