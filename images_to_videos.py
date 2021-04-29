import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

img = []
path = "homography"

list = os.listdir(path)
list = [os.path.join(path, item) for item in list if os.path.splitext(item)[1] == ".jpg"]
list.sort()
# list.sort(key=os.path.getctime)

i = 0
for image in list:
    print("image {}".format(i))
    i += 1
    image = cv.imread(image)
    image = cv.resize(image, None, fx = 4/3, fy = 1, interpolation = cv.INTER_CUBIC)
    img.append(image)

h, w, _ = img[0].shape
fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
video = cv.VideoWriter('video.avi', fourcc, 15, (w, h))

j = 0
for frame in img:
    print("writting frame {}".format(j))
    j += 1
    video.write(frame)

video.release()
cv.destroyAllWindows()
