import cv2
import cvzone
import numpy as np
import pickle

vid_path = '/Users/parthagarwal/Desktop/pythonProject/data/easy1.mp4'

cap = cv2.VideoCapture(vid_path)
area_names = []
try:
    with open('data_points', 'rb') as f:
        data = pickle.load(f)
        polyLines, area_names = data['mask'], data['area_names']
except:
    polyLines = []
drawing = False
points = []


def draw(event, x, y, flags, param):
    global drawing, points
    drawing = True
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(points) == 4:
            name = input('enter name: ')
            area_names.append(name)
            polyLines.append(np.array(points, np.int32))
            points.clear()


while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame = cv2.resize(frame, (1020, 500))
    for i, j in enumerate(polyLines):
        cv2.polylines(frame, [j], True, (0, 0, 255), 2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(j[0]), 1, 1)
    for i in points:
        cv2.circle(frame, i, 3, (0, 255, 0), -1)
    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME', draw)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        with open('data_points', 'wb') as f:
            data = {'mask': polyLines, 'area_names': area_names}
            pickle.dump(data, f)
        break
cap.release()
cv2.destroyAllWindows()

print(polyLines)
