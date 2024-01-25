import cv2
from cv2 import VideoWriter


webcam = cv2.VideoCapture(4)  # 0 for hd cam 2 for ir cam 4 for logi cam

while True:
    stream_ok, frame = webcam.read()

    if stream_ok:
        cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
webcam.release()