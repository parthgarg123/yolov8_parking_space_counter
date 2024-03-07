import cv2
import numpy as np

vid_path='/Users/parthagarwal/Desktop/pythonProject/data/easy1.mp4'
cap = cv2.VideoCapture(vid_path)

drawing = False
points =[]
polyLines=[]
def draw(event,x,y,flags,param):
    global drawing,points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x,y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        polyLines.append(np.array(points,np.int32))

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1020,500))
    for i in polyLines:
        cv2.polylines(frame,[i],True,(0,0,255),2)
    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME',draw)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print(polyLines)