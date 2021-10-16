import numpy as np
import cv2
import os
import imutils

#SAVE_DIR Location
path = input("Name of the directory which you want to create : ")

if not os.path.exists(path):
    os.mkdir(path)

#Path to prototxtfile
prototxt_file = "face_detector//deploy.prototxt"
#Path to caffe_model
caffeModel = "face_detector//res10_300x300_ssd_iter_140000.caffemodel"

face_detector = cv2.dnn.readNet(prototxt_file, caffeModel)

#How many images you want
num_imgs = int(input("How many Images : "))

# Detect faces
def detect_faces(frame, face_detector):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400), (104.0, 177.0, 123.0))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    locs = []; confidences = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startx, starty, endx, endy) = box.astype("int")

            (startx, starty) = (max(0, startx), max(0, starty))
            (endx, endy) = (min(w-1, endx), min(h-1, endy))

            locs.append((startx, starty, endx, endy))
            confidences.append(confidence)

    locs = np.array(locs)
    confidences = np.array(confidences, dtype="int")
    # print(f"data type of : {type(confidences[0])}")
    
    return locs


#Load the Camera to save faces
vcap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = vcap.read()
    if ret:    
        frame = imutils.resize(frame, width=400)
        boxes = detect_faces(frame, face_detector)
        # print(boxes)
        largest_box = 0   
        for box in boxes:
            if box is None:
                continue
            (startX, startY, endX, endY) = box
            # cv2.rectangle(frame, (startX, startY), (endX, endY), (2, 220, 10), 2)
            cv2.putText(frame, f"Captured : {count}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2, 220, 10))            
            face = frame[startY:endY, startX:endX]
            cv2.imwrite(f"{path}//img_{count}.jpg", face)    
            count += 1
                  
        
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        if (count > num_imgs):
            print(f"collected {count} images")
            break

cv2.destroyAllWindows()
vcap.release()
