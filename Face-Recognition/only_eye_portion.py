import numpy as np
import cv2
import os

def eye_collector(path):
    count = 0
    for img in os.listdir(path):
        count += 1
        img = cv2.imread(os.path.join(path, img))
        h, w, c = img.shape
        new_img = img[0:h//2, :w]
        cv2.imwrite(f"eyes//Rudra_training//rpd_eye_{count}.jpg", new_img)
    print(f"collected {count} imgs")

eye_collector("Only_faces\Rudra_train")