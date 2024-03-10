import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone

with open('data_points', 'rb') as f:
    data = pickle.load(f)
    polyLines, area_names = data['mask'], data['area_names']

my_file = open("data/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

model = YOLO('yolov8s.pt')
vid_path = '/Users/parthagarwal/Desktop/pythonProject/data/easy1.mp4'

cap = cv2.VideoCapture(vid_path)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    frame_copy = frame.copy()
    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    loc=[]
    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        if 'car' in c:
            loc.append([cx,cy])
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
    counter=[]
    for i, j in enumerate(polyLines):
        cv2.polylines(frame, [j], True, (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(j[0]), 1, 1)
        res=-1
        for n in loc:
            x = n[0]
            y = n[1]
            for x1 in range(x - 2, x + 3):
                for y1 in range(y - 2, y + 3):
                    temp_res = cv2.pointPolygonTest(j, (x1, y1), False)
                    if temp_res>=0:
                        res=temp_res
                        #cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        if res >= 0:
            cv2.polylines(frame,[j],True,(0,0,255),2)
            counter.append((x,y))

    cvzone.putTextRect(frame, f'{len(counter)}', (30, 30), 2, 2)
    cv2.imshow('FRAME', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

