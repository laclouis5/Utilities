import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

def video_to_images(video, save_dir, nb_images=None, step=None):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    i = 0
    video = cv.VideoCapture(video)

    lines = []

    while True:
        ret, frame = video.read()

        if ret == True and (nb_images == None or i < nb_images):
            file_name = os.path.join(save_dir, "im_{}.jpg".format(i))
            print("Writting image {}".format(i))
            cv.imwrite(file_name, frame)
            lines.append(file_name + "\n")

            if step is not None:
                [video.read() for _ in range(step - 1)]

            i += 1
        else:
            break

    with open("new_train.txt", "w") as f:
        f.writelines(lines)

def main():
    video_to_images("demo_mais.mov", "demo_mais", 100, 2)

if __name__ == "__main__":
    main()
