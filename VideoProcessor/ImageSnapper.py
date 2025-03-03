import cv2

cap = cv2.VideoCapture('../Data/parking_loop.mp4')
ret, frame = cap.read()

if ret:
    cv2.imwrite('../Data/parking_loop_frame.png', frame)

cap.release()