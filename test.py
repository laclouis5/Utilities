from skimage import io, data, filters, feature, color, exposure, morphology
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from PIL import Image, ImageDraw
import os
from joblib import Parallel, delayed
from BoundingBoxes import BoundingBoxes
from BoundingBox import BoundingBox
from utils import *

def egi_mask(image, thresh=1.15):
    image_np = np.array(image).astype(float)

    image_np = 2*image_np[:, :, 1] / (image_np[:, :, 0] + image_np[:, :, 2] + 0.001)
    image_gf = filters.gaussian(image_np, sigma=1, mode='reflect')

    image_bin = image_gf > 1.15

    image_morph = morphology.binary_erosion(image_bin, morphology.disk(3))
    image_morph = morphology.binary_dilation(image_morph, morphology.disk(3))

    image_out = morphology.remove_small_objects(image_morph, 400)
    image_out = morphology.remove_small_holes(image_out, 800)

    return image_out


def scatter3d(image, egi_mask):
    x_rand = np.random.randint(0, 2448, 4000)
    y_rand = np.random.randint(0, 2048, 4000)

    list   = []
    colors = []
    for x, y in zip(x_rand, y_rand):
        list.append(image[y, x, :])
        if egi_mask[y, x]:
            colors.append('g')
        else:
            colors.append('k')

    r, g, b = zip(*list)

    # HSV
    image_2 = color.rgb2hsv(image)

    list_2   = []
    for x, y in zip(x_rand, y_rand):
        list_2.append(image_2[y, x, :])

    h, s, v = zip(*list_2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = fig.gca(projection='3d')

    ax.scatter(r, g, b, c=colors)
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    plt.show()

    # ax.scatter(h, s, v, c=colors)
    # ax.set_xlabel("H")
    # ax.set_ylabel("S")
    # ax.set_zlabel("V")
    # plt.show()

def compute_struct_tensor(image_path, w, sigma=1.5):
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)

        border_type = cv.BORDER_REFLECT

        # Gradients
        Gx  = cv.Sobel(img, cv.CV_32F, 1, 0, 3, borderType=border_type)
        Gy  = cv.Sobel(img, cv.CV_32F, 0, 1, 3, borderType=border_type)

        # Filtered Structure Tensor Components
        Axx = cv.GaussianBlur(Gx * Gx, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=border_type)
        Ayy = cv.GaussianBlur(Gy * Gy, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=border_type)
        Axy = cv.GaussianBlur(Gx * Gy, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=border_type)

        # Eigenvalues
        tmp1 = Axx + Ayy
        tmp2 = (Axx - Ayy) * (Axx - Ayy)
        tmp3 = Axy * Axy
        tmp4 = cv.sqrt(tmp2 + 4.0 * tmp3)

        lambda1 = tmp1 + tmp4
        lambda2 = tmp1 - tmp4

        # Coherency and Orientation
        img_coherency = (lambda1 - lambda2) / (lambda1 + lambda2)
        img_orientation = 0.5 * cv.phase(Axx - Ayy, 2.0 * Axy, angleInDegrees=True)

        return img_coherency, img_orientation


def plot_bbox_distribution(boundingBoxes):
    classes = boundingBoxes.getClasses()
    cmap = matplotlib.cm.get_cmap(name='gist_rainbow')
    cmap = cmap(np.linspace(0., 1., num=len(classes)))
    fig, ax = plt.subplots()

    for i, classID in enumerate(classes):
        bounding_boxes = boundingBoxes.getBoundingBoxByClass(classID)
        x, y, area = [], [], []
        mean = 0.0

        for bounding_box in bounding_boxes:
            xc, yc, wc, hc = bounding_box.getRelativeBoundingBox()
            x.append(round(xc*bounding_box.getImageSize()[0]))
            y.append(round(yc*bounding_box.getImageSize()[1]))
            ar = round((wc*bounding_box.getImageSize()[0])*(hc*bounding_box.getImageSize()[1]))
            area.append(ar)
            mean += ar

        print("Mean area for {}: {:,}".format(classID, mean/len(area)))
        print("Min area for {}: {:,}".format(classID, min(area)))
        print("Max area for {}: {:,}".format(classID, max(area)))

        area = [item/max(area)*100 for item in area]
        ax.scatter(x, y, area, label=classID)

    ax.legend()
    ax.grid(True)
    plt.xlim([0, 2448])
    plt.ylim([0, 2048])
    # plt.xlim([0, 2048])
    # plt.ylim([0, 1536])
    plt.title("Box center distribution")
    plt.show()


def tile_database(boundingBoxes):
    # parameters
    (im_w, im_h) = 2448, 2048
    main_tile_size = 1664
    tile_size = 416

    X1 = [(im_w - main_tile_size)/2 + n*tile_size for n in range(4)]
    Y1 = [(im_h - main_tile_size)/2 + n*tile_size for n in range(4)]

    X2 = [(im_w - main_tile_size)/2  + tile_size/2 + n*tile_size for n in range(3)]
    Y2 = [(im_h - main_tile_size)/2  + tile_size/2 + n*tile_size for n in range(3)]

    X1, Y1 = np.meshgrid(X1, Y1)
    X2, Y2 = np.meshgrid(X2, Y2)

    # out_image = Image.new('RGBA', size=(im_w, im_h))
    # drawing = ImageDraw.Draw(out_image)
    # for (i, j) in zip(X1, Y1):
    #     for (k, l) in zip(i, j):
    #         drawing.rectangle(xy=[k, l, k +tile_size, l + tile_size], fill=(0, 0, 255, 64))
    # for (i, j) in zip(X2, Y2):
    #     for (k, l) in zip(i, j):
    #         drawing.rectangle([k, l, k + tile_size, l + tile_size], fill=(255, 0, 0, 64))

    image_in = Image.open('haricot.jpg')
    index = 0
    for (i, j) in zip(X1, Y1):
        for (k, l) in zip(i, j):
            imagette = image_in.crop([k, l, k + tile_size, l + tile_size])
            imagette.save('haricot_{}.jpg'.format(index))
            index += 1
    for (i, j) in zip(X2, Y2):
        for (k, l) in zip(i, j):
            imagette = image_in.crop([k, l, k + tile_size, l + tile_size])
            imagette.save('haricot_{}.jpg'.format(index))
            index += 1


def get_square_database(yolo_dir, save_dir=''):
    train_dir = os.path.join(yolo_dir, 'train/')
    val_dir   = os.path.join(yolo_dir, 'val/')

    train_folder = 'train/'
    val_folder   = 'val/'

    for d, directory in zip([train_folder, val_folder], [train_dir, val_dir]):
        images = [os.path.join(directory, item) for item in os.listdir(directory)]
        images = [item for item in images if os.path.splitext(item)[1] == '.jpg']

        for image in images:
            img = Image.open(image)
            (img_w, img_h) = img.size

            if img_h < img_w:
                bbox = [round(float(img_w)/2 - float(img_h)/2), 0, round(float(img_w)/2 + float(img_h)/2), img_h]
                img_out = img.crop(bbox)
            elif img_w < img_h:
                bbox = [0, round(float(img_h)/2 - float(img_w)/2), img_w, round(float(img_h)/2 + float(img_h)/2)]
                img_out = img.crop(bbox)
            else:
                img_out = img

            assert img_out.size[0] == img_out.size[1], "Can't crop to a square shape."

            # print(os.path.join(save_dir, d, os.path.basename(image)))
            img_out.save(os.path.join(save_dir, d, os.path.basename(image)), quality=100)

        annotations = [os.path.join(directory, item) for item in os.listdir(directory)]
        annotations = [item for item in annotations if os.path.splitext(item)[1] == '.txt']

        for annotation in annotations:
            content_out = []
            corresp_img = os.path.splitext(annotation)[0] + '.jpg'
            (img_w, img_h) = Image.open(corresp_img).size

            # If image is in landscape
            if img_h < img_w:
                print("In landscape mode: {} by {}".format(img_w, img_h))
                # Here are abs coords of square bounds (left and right)
                (w_lim_1, w_lim_2) = round(float(img_w)/2 - float(img_h)/2), round(float(img_w)/2 + float(img_h)/2)

                with open(annotation, 'r') as f:
                    print("Reading annotation...")
                    content = f.readlines()
                    content = [line.strip() for line in content]

                    for line in content:
                        print("Reading a line...")
                        line = line.split()
                        # Get relative coords (in old coords system)
                        (label, x, y, w, h) = line[0], float(line[1]), float(line[2]), float(line[3]), float(line[4])
                        print("Line is: {} {} {} {} {}".format(label, x, y, w, h))

                        # If bbox is not out of the new square frame
                        if not (x*img_w < w_lim_1 or x*img_w > w_lim_2):
                            print("In square bounds")
                            # But if bbox spans out of one bound (l or r)
                            if (x - w/2.0) < (float(w_lim_1)/img_w):
                                print("Spans out of left bound")
                                # Then adjust bbox to fit in the square
                                w = w - (float(w_lim_1)/img_w - (x - w/2.0))
                                x = float(w_lim_1+1)/img_w + w/2.0
                            if (x + w/2.0) > (float(w_lim_2)/img_w):
                                print("Span out of right bound")
                                w = w - (x + w/2.0 - float(w_lim_2)/img_w)
                                x = float(w_lim_2)/img_w - w/2.0
                            else:
                                print("Does not spans outside")

                        # If out of bounds...
                        else:
                            print("Out of square bounds")
                            # ...do not process the line
                            continue

                        # Do not forget to convert from old coord sys to new one
                        x = (x*img_w - float(w_lim_1))/float(w_lim_2 - w_lim_1)
                        w = w*img_w/float(w_lim_2 - w_lim_1)

                        assert x >= 0, "Value was {}".format(x)
                        assert x <= 1, "Value was {}".format(x)
                        assert (x - w/2) >= 0, "Value was {}".format(x - w/2)
                        assert (x + w/2) <= 1, "Value was {}".format(x + w/2)

                        new_line = "{} {} {} {} {}\n".format(label, x, y, w, h)
                        content_out.append(new_line)


            # If image is in portrait
            elif img_w < img_h:
                (h_lim_1, h_lim_2) = round(float(img_h)/2 - float(img_w)/2), round(float(img_h)/2 + float(img_h)/2)

                with open(annotation, 'r') as f:
                    content = f.readlines()
                    content = [line.strip() for line in content]

                    for line in content:
                        line = line.split()
                        (label, x, y, w, h) = line[0], float(line[1]), float(line[2]), float(line[3]), float(line[4])

                        if not (y*img_h < h_lim_1 or y*img_h > h_lim_2):
                            print('out')

            else:
                annotation_out = annotation

            # Write updated content to TXt file
            with open(os.path.join(save_dir, d, os.path.basename(annotation)), 'w') as f:
                f.writelines(content_out)


def draw_bbox_images(folder, save_dir):
    # Attention fonction à la zeub juste pour tester
    images = [os.path.join(folder, item) for item in os.listdir(folder)]
    images = [item for item in images if os.path.splitext(item)[1] == '.jpg']
    colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]

    for image in images:
        img = Image.open(image)
        (img_w, img_h) = img.size

        annotation = os.path.splitext(image)[0] + '.txt'
        with open(annotation, 'r') as f:
            content = f.readlines()

        content = [line.strip().split() for line in content]
        img_draw = ImageDraw.Draw(img)

        for line in content:
            (label, x, y, w, h) = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
            xmin = (x - w/2)*img_w
            xmax = (x + w/2)*img_w
            ymin = (y - h/2)*img_h
            ymax = (y + h/2)*img_h
            img_draw.rectangle([xmin, ymin, xmax, ymax], outline=colors[label])

        print(os.path.join(save_dir, os.path.basename(image)))
        img.save(os.path.join(save_dir, os.path.basename(image)))


def read_txt_annotation_file(file_path, img_size):
    bounding_boxes = BoundingBoxes(bounding_boxes=[])
    image_name = os.path.basename(os.path.splitext(file_path)[0] + '.jpg')

    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, x, y, w, h) = int(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
        bounding_boxes.addBoundingBox(BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, imgSize=img_size))

    return bounding_boxes


def parse_yolo_folder(data_dir):
    annotations = os.listdir(data_dir)
    annotations = [os.path.join(data_dir, item) for item in annotations if os.path.splitext(item)[1] == '.txt']
    images = [os.path.splitext(item)[0] + '.jpg' for item in annotations]
    bounding_boxes = BoundingBoxes(bounding_boxes=[])

    for (img, annot) in zip(images, annotations):
        img_size = Image.open(img).size
        image_boxes = read_txt_annotation_file(annot, img_size)
        [bounding_boxes.addBoundingBox(bb) for bb in image_boxes.getBoundingBoxes()]

    return bounding_boxes


def parse_yolo_dir(directory, disp_stats=False):
    train_dir = os.path.join(directory, "train/")
    val_dir = os.path.join(directory, "val/")

    train_boxes = parse_yolo_folder(train_dir)
    val_boxes = parse_yolo_folder(val_dir)

    if disp_stats:
        train_boxes.stats()
        val_boxes.stats()

    return train_boxes, val_boxes


# image = io.imread("data/carotte.jpg")
# mask = egi_mask(image)
# image_green = image.copy()
# image_green[mask==0] = 0
# # plt.subplot(221)
# # plt.imshow(image)
# # plt.subplot(222)
# # plt.imshow(mask)
# # plt.subplot(223)
# # plt.imshow(image_green)
# # plt.show()
#
# # scatter3d(image, mask)
# #structure_tensor(image)
#
# coherence, orientation = compute_struct_tensor("data/im_33.jpg", 32, 10)
#
# plt.subplot(121)
# plt.title("Coherence")
# plt.imshow(coherence)
# plt.subplot(122)
# plt.title("Orientation")
# plt.imshow(orientation)
# plt.show()
