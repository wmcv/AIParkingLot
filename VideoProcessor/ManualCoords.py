import cv2
import numpy as np

image = cv2.imread('../Data/parking_loop_frame.png')

parking_slots = []
points = []

def click_and_get_coordinates(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked at: ({x}, {y})")

        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Parking Lot", image)

        if len(points) == 4:
            parking_slots.append(points)
            points = []
            print(f"Parking slot added: {parking_slots[-1]}")

cv2.imshow("Parking Lot", image)
cv2.setMouseCallback("Parking Lot", click_and_get_coordinates)

cv2.waitKey(0)
cv2.destroyAllWindows()

for slot in parking_slots:
    cv2.polylines(image, [np.array(slot)], isClosed=True, color=(0, 255, 0), thickness=2)

cv2.imshow("Parking Lot with Slots", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Parking slot coordinates:", parking_slots)
