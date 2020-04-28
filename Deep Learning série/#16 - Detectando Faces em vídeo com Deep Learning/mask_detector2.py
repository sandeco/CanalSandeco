from tensorflow.keras.models import load_model
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# Disable scientific notation for clarity
#np.set_printoptions(suppress=True)

path = "D://jupyter//10 - Face Recognition Systems//04 - MaskDetector//"

detector = MTCNN()
model = load_model(path + "detector.h5")
cap = cv2.VideoCapture(0)
size = (160, 160)

while True:

    ret, frame = cap.read()
    labels = []
    faces = detector.detect_faces(frame)

    people = 0

    for face in faces:

        x1, y1, w, h = face['box']
        # bug fix
        # #x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + w, y1 + h

        roi = frame[y1:y2, x1:x2]

        #RESIZE
        roi = cv2.resize(roi,size)

        if np.sum([roi])!=0:
            roi = (roi.astype('float')/255.0)

            # PREDIÇÃO
            pred = model.predict([[roi]])

            pred = pred[0] ## pegando o vetor interno da classificação

            if pred[0] >= pred[1]:
                label = 'NO MASK'
                color = (0,0,255)
                people = people + 1
            else:
                label = 'MASK'
                color = (0,255,0)

            #label_position = (x1-100, y1+250)
            label_position = (x1, y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,label, label_position, cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)

        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

    cv2.putText(frame, "NO MASKS : " + str(people), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('SANDECO MASK DETECTOR', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()