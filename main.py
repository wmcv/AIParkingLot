import cv2
from ultralytics import YOLO
import numpy as np

# preprocess and load data
model = YOLO('AIModel/best.pt')
cap = cv2.VideoCapture('Data/car_lot_video15.mp4')
frame_count = 0
alpha = 0.15
scale_x = 1920 / 640
scale_y = 1080 / 384
parking_spaces = [[(76, 353), (135, 452), (245, 454), (163, 351)], [(163, 351), (248, 458), (335, 451), (256, 349)], [(256, 349), (339, 456), (434, 450), (354, 349)], [(354, 349), (435, 453), (530, 450), (452, 346)], [(452, 346), (532, 454), (633, 454), (549, 346)], [(549, 346), (637, 456), (737, 454), (646, 344)], [(646, 344), (738, 459), (832, 451), (749, 346)], [(749, 346), (835, 453), (934, 455), (850, 345)], [(850, 345), (938, 452), (1038, 450), (954, 345)], [(954, 345), (1039, 452), (1143, 454), (1055, 342)], [(1055, 342), (1146, 451), (1246, 448), (1161, 346)], [(1161, 346), (1243, 452), (1346, 452), (1264, 347)], [(1264, 347), (1345, 451), (1450, 456), (1363, 346)], [(1363, 346), (1452, 455), (1553, 452), (1466, 347)], [(1466, 347), (1550, 453), (1647, 453), (1570, 352)], [(1570, 352), (1651, 450), (1750, 452), (1668, 353)], [(1668, 353), (1750, 457), (1840, 452), (1767, 349)], [(1767, 350), (1840, 450), (1913, 453), (1863, 352)], [(14, 673), (129, 800), (225, 804), (108, 672)], [(108, 672), (225, 806), (313, 803), (196, 677)], [(196, 677), (314, 806), (397, 800), (285, 675)], [(285, 675), (400, 808), (498, 808), (381, 683)], [(381, 683), (500, 812), (591, 813), (473, 680)], [(473, 680), (593, 810), (691, 811), (564, 678)], [(564, 678), (693, 813), (785, 813), (664, 685)], [(664, 685), (788, 817), (883, 814), (758, 682)], [(758, 682), (885, 817), (978, 813), (856, 687)], [(856, 687), (979, 810), (1084, 819), (953, 685)], [(953, 685), (1084, 818), (1177, 817), (1054, 688)], [(1054, 688), (1178, 817), (1274, 814), (1155, 689)], [(1155, 689), (1276, 817), (1372, 817), (1249, 686)], [(1249, 686), (1377, 816), (1468, 815), (1349, 689)], [(1349, 689), (1472, 810), (1574, 818), (1453, 688)], [(1453, 688), (1579, 821), (1665, 816), (1544, 688)], [(1544, 688), (1666, 814), (1715, 769), (1654, 688)]]
parking_occupancy = [False] * len(parking_spaces)
parking_avaliable = len(parking_spaces)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    overlay = frame.copy()

    # frame skipping
    frame_count += 1
    if frame_count % 3 != 0:
        continue



    frame_resized = cv2.resize(frame, (640, 384))

    results = model(frame_resized, conf=0.5, iou=0.5)  # Increased confidence threshold to 0.5

    # drawing parking lots
    for idx, space in enumerate(parking_spaces):
        pts = [(x, y) for x, y in space]
        color = (0, 255, 0) if not parking_occupancy[idx] else (0, 0, 255)  # Green for empty, Red for occupied
        cv2.polylines(frame, [np.array(pts, np.int32)], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(overlay, [np.array(pts, np.int32)], color=color)

    # drawing car bboxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        # rescale coords to original pos
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        if conf > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # car bbox midpoint
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2

            # car in lot detector
            for idx, space in enumerate(parking_spaces):
                polygon = np.array(space, np.int32)
                polygon = polygon.reshape((-1, 1, 2))

                if cv2.pointPolygonTest(polygon, (midpoint_x, midpoint_y), False) >= 0:
                    parking_occupancy[idx] = True
                    break

    parking_used = 0
    for lot in parking_occupancy:
        if lot:
            parking_used+=1

    avaliable_parking_label = f'Avaliable Spots: {parking_avaliable-parking_used}'
    used_parking_label = f'Occupied Spots: {parking_used}'
    cv2.putText(frame, avaliable_parking_label, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    cv2.putText(frame, used_parking_label, (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

    # add semi-transparent mask for car lots
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.imshow('AI Car Parking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
