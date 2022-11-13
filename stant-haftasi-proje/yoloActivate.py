import torch
import cv2
import numpy as np
import time

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)
model.conf = 0.70
# From camera
vid = cv2.VideoCapture(0)

color = (255,0,0)
red = (0, 0, 255)

while True:
    ret, frame = vid.read()
    
    # Inference

    # Results
    


    results = model(frame)

    npResults = results.pandas().xyxy[0].to_numpy()
    #print("Girdi")

    if(len(npResults) != 0):
        for i in range(len(npResults)):
            #We define the class number into the classNumber variable and class name to className
            classNumber = npResults[i][5]
            className = npResults[i][6]
            top_left = (round(npResults[i][0]), round(npResults[i][1]))
            bottom_right = (round(npResults[i][2]), round(npResults[i][3]))
            if(classNumber == 0):
                frame = cv2.rectangle(frame,top_left, bottom_right, color, 2)
                frame = cv2.putText(frame, 'Future SKYLAB Member', (round(npResults[i][0]), round(npResults[i][1]-10)), 0, 0.5, color, 2)
            else:
                frame = cv2.rectangle(frame,top_left, bottom_right, color, 2)
                frame = cv2.putText(frame, className, (round(npResults[i][0]), round(npResults[i][1]-10)), 0, 0.5, red, 2)

    
    #results.show()

    #crops = results.crop(save=False)

    resized = cv2.resize(frame, (1024,768), interpolation = cv2.INTER_AREA)
    cv2.imshow("res1",resized)
    
    #del npResults, resized, results, frame, className, classNumber, top_left, bottom_right

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
#vid2.release()

# Destroy all the windows
cv2.destroyAllWindows()