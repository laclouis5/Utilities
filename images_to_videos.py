import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

img = []
path = "to_label_carotte"

list = os.listdir(path)
list = [os.path.join(path, item) for item in list]
list.sort()

for image in list:
    img.append(cv.imread(image))

h, w, _ = img[1].shape
fourcc = cv.VideoWriter_fourcc(*'M', 'J', 'P', 'G')
video = cv.VideoWriter('video.avi', fourcc, 1, (w, h))

for frame in img:
    video.write(frame)

video.release()
cv.destroyAllWindows()
