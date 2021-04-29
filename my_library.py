from skimage import io, data, filters, feature, color, exposure, morphology
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from PIL import Image, ImageDraw
import os
from joblib import Parallel, delayed

from BoxLibrary import *
from numba import jit
import random


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

    list = []
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

    list_2 = [image_2[y, x, :] for x, y in zip(x_rand, y_rand)]
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

def pix2pix_square_stem_db(boxes, size=1024, save_dir="pix_to_pix/", label_size=2/100):
    images_dir = os.path.join(save_dir, "images/")
    segmentation_masks = os.path.join(save_dir, "masks/")

    create_dir(save_dir)
    create_dir(images_dir)
    create_dir(segmentation_masks)

    # labels = boxes.getClasses()
    image_names = boxes.getNames()
    random.shuffle(image_names)

    # Do things withs labels to assign colors
    hard_coded_colors = {
        "maize_stem": (0, 0, 255), # Red
        "bean_stem": (0, 255, 0), # Green
        "leek_stem": (255, 0, 0) # Blue
    }

    for im_id, name in enumerate(image_names):
        image_boxes = boxes.getBoundingBoxesByImageName(name)
        image = cv.imread(name)
        (img_height, img_width) = image.shape[:2]
        keypoint_radius = int(label_size * img_height / 2)
        segmentation_image = np.zeros((img_height, img_width, 3), np.uint8)
        out_basename = "{}".format(im_id).zfill(6)
        segmented_name = os.path.join(segmentation_masks, "im_{}.png".format(out_basename))
        img_out_name = os.path.join(images_dir, "im_{}.png".format(out_basename))

        for box in image_boxes:
            (x, y, _, _) = box.getAbsoluteBoundingBox(format=BBFormat.XYC)
            label = box.getClassId()
            color = hard_coded_colors[label]
            # print(name, label, color, x, y)
            cv.circle(segmentation_image, center=(int(x), int(y)), radius=keypoint_radius, color=color, thickness=cv.FILLED)

        padding = (img_width - img_height) / 2
        xmin = int(padding)
        xmax = int(img_width - padding)

        segmentation_image = segmentation_image[:, xmin:xmax]
        assert segmentation_image.shape[0] == segmentation_image.shape[1]
        segmentation_image = cv.resize(segmentation_image, dsize=(size, size), interpolation=cv.INTER_LINEAR)

        image = image[:, xmin:xmax]
        image = cv.resize(image, dsize=(size, size), interpolation=cv.INTER_LINEAR)

        cv.imwrite(segmented_name, segmentation_image)
        cv.imwrite(img_out_name, image)


def get_square_database(yolo_dir, save_dir=''):
    '''
    Takes as input the path to a yolo database. Crops this database to a
    square one and saves it in save_dir.
    '''
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
            img_out.save(os.path.join(save_dir, d, os.path.basename(image)))

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
    '''
    Takes as input a folder with images and yolo-style annotation (TXT file).
    Saves images with bounding boxes drawn in the save_dir folder.
    '''
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


def draw_boxes_bboxes(image, bounding_boxes, save_dir, color=[255, 64, 0]):
    '''
    Takes as input one image (numpy array) and a BoundingBoxes object
    representing the bounding boxes, draws them into and saves the image in
    save_dir.
    '''
    image = image.copy()
    for box in bounding_boxes:
        add_bb_into_image(image, box, color=color, label=str(box.getClassId()))
        img_name = os.path.basename(box.getImageName())
        image_path = os.path.join(save_dir, img_name)

    cv.imwrite(image_path, image)


def add_bboxes_image(image, bboxes, color=[255, 64, 0]):
    for box in bboxes:
        add_bb_into_image(image, box, color=color, label=str(box.getClassId()))


def create_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def read_txt_annotation_file(file_path, img_size):
    '''
    Input are TXT file path and corresponding image size. Output are
    bounding boxes as a BoundingBox object.
    '''
    bounding_boxes = BoundingBoxes()
    image_name = os.path.splitext(file_path)[0] + '.jpg'

    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4]), bounding_boxes.append(BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, imgSize=img_size))

    return bounding_boxes


def parse_yolo_folder(data_dir):
    '''
    Input is either train dir or val dir of yolo folder. This function reads
    TXT annotation files and returns a BoundingBoxes object.
    '''
    annotations = os.listdir(data_dir)
    annotations = [os.path.join(data_dir, item) for item in annotations if os.path.splitext(item)[1] == '.txt']
    images = [os.path.splitext(item)[0] + '.jpg' for item in annotations]
    bounding_boxes = BoundingBoxes()

    for (img, annot) in zip(images, annotations):
        img_size = Image.open(img).size
        image_boxes = read_txt_annotation_file(annot, img_size)
        [bounding_boxes.append(bb) for bb in image_boxes]

    return bounding_boxes


def parse_yolo_dir(directory, disp_stats=False):
    '''
    Input is the yolo folder containing train and val subfolders. Returns a
    BoundingBoxes object.
    '''
    train_dir = os.path.join(directory, "train/")
    val_dir = os.path.join(directory, "val/")

    train_boxes = parse_yolo_folder(train_dir)
    val_boxes = parse_yolo_folder(val_dir)

    if disp_stats:
        train_boxes.stats()
        val_boxes.stats()

    return train_boxes, val_boxes


def xywh_to_xyx2y2(x, y, w, h):
    '''
    Takes as input absolute coords and returns integers.
    '''
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def get_stem_database(yolo_folder, save_dir="plant_stem_db/"):
    # Global var
    resolution = 832
    create_dir(save_dir)
    yolo_boxes = parse_yolo_folder(yolo_folder)
    image_list = []

    for image_name in yolo_boxes.getNames():
        boxes = BoundingBoxes()

        for box in yolo_boxes.getBoundingBoxesByImageName(image_name):
            # If stem box
            # if box.getClassId() in [3, 4, 5]:
            if "stem" in box.getClassId():
                im_w, im_h = box.getImageSize()
                (x, y, w, h) = box.getAbsoluteBoundingBox(format=BBFormat.XYC)
                # Normalize size and convert to square
                size = 0.075 * min(im_w, im_h)
                x = round(x - size / 2.0)
                y = round(y - size / 2.0)
                box = BoundingBox(imageName=box.getImageName(), classId=box.getClassId(), x=x, y=y, w=size, h=size, imgSize=box.getImageSize())
                boxes.append(box)
                continue

            # Else
            im_w, im_h = box.getImageSize()
            (x, y, w, h) = box.getAbsoluteBoundingBox(format=BBFormat.XYC)

            # Convert to square and expand a little
            l = max(w, h)
            l = min(l, min(im_w, im_h))

            new_x = x
            new_y = y
            new_w = round(l + 0.075 * min(im_w, im_h))
            new_h = round(l + 0.075 * min(im_w, im_h))

            # Then clip shape to stay in original image
            xmin, ymin, xmax, ymax = xywh_to_xyx2y2(new_x, new_y, new_w, new_h)
            if xmin < 0:
                new_x = x - xmin
            if xmax >= im_w:
                new_x = x - (xmax - im_w)
            if ymin < 0:
                new_y = y - ymin
            if ymax >= im_h:
                new_y = y - (ymax - im_h)

            new_x = round(new_x - new_w / 2.0)
            new_y = round(new_y - new_h / 2.0)

            new_box = BoundingBox(imageName=image_name, classId=box.getClassId(), x=new_x, y=new_y, w=new_w, h=new_h, imgSize=box.getImageSize())
            boxes.append(new_box)

        # Retreive stems for each plant sub-image
        plants = boxes.getBoundingBoxByClass("maize")
        plants += boxes.getBoundingBoxByClass("bean")
        plants += boxes.getBoundingBoxByClass("leek")

        stems = boxes.getBoundingBoxByClass("stem_maize")
        stems += boxes.getBoundingBoxByClass("stem_bean")
        stems += boxes.getBoundingBoxByClass("stem_leek")

        for (i, box_plant) in enumerate(plants):
            stem_boxes = BoundingBoxes(bounding_boxes=[])
            (xmin, ymin, xmax, ymax) = box_plant.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
            (x_s, y_s, w_s, h_s) = box_plant.getAbsoluteBoundingBox(format=BBFormat.XYC)

            for box_stem in stems:
                (x, y, w, h) = box_stem.getAbsoluteBoundingBox(format=BBFormat.XYC)
                (x_min, y_min, x_max, y_max) = box_stem.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

                # If stem box is not in plant box pass
                if x + 0.7 * w / 2.0 >= xmax: continue
                if x - 0.7 * w / 2.0 < xmin: continue
                if y + 0.7 * h / 2.0 >= ymax: continue
                if y - 0.7 * h / 2.0 < ymin: continue

                # Change referencial
                x_min = round((x_min - xmin) / w_s * resolution)
                x_max = round((x_max - xmin) / w_s * resolution)
                y_min = round((y_min - ymin) / h_s * resolution)
                y_max = round((y_max - ymin) / h_s * resolution)

                x_min = max(x_min, 0)
                if x_max >= resolution : x_max = resolution - 1
                y_min = max(y_min, 0)
                if y_max >= resolution : y_max = resolution - 1

                new_box = BoundingBox(imageName=box_stem.getImageName(), classId=box_stem.getClassId()-3, x=x_min, y=y_min, w=x_max, h=y_max, format=BBFormat.XYX2Y2, imgSize=(resolution, resolution))
                stem_boxes.append(new_box)

            # Create new reshaped image
            image = cv.imread(box_plant.getImageName())
            patch = cv.getRectSubPix(image, (w_s, h_s), (x_s, y_s))
            patch = cv.resize(patch, (resolution, resolution))
            file_name = os.path.splitext(os.path.basename(box_plant.getImageName()))[0]
            file_name = "{}_{}.jpg".format(file_name, i)
            file_name = os.path.join(save_dir, file_name)
            # add_bboxes_image(patch, stem_boxes)
            cv.imwrite(file_name, patch)

            # New annotations
            annot_name = os.path.splitext(file_name)[0] + ".txt"
            annot = []
            for stem in stem_boxes.getBoundingBoxes():
                (x, y, w, h) = stem.getRelativeBoundingBox()
                label = str(stem.getClassId())
                line = "{} {} {} {} {}\n".format(label, x, y, w, h)
                annot.append(line)

            with open(annot_name, "w") as f:
                f.writelines(annot)

            # New train/val file
            file_name = os.path.join("data/val/", os.path.basename(file_name))
            image_list.append(file_name + '\n')


        # Save train/val file
        with open(os.path.join(save_dir, "val.txt"), "w") as f:
            f.writelines(image_list)

def normalized_stem_boxes(boxes, ratio=7.5/100, labels={"haricot_tige", "mais_tige", "poireau_tige"}):
	return BoundingBoxes([box.normalized(ratio) if box.getClassId() in labels else box for box in boxes])


def main(args=None):
    yolo_path = '/home/deepwater/github/darknet/data/database_6.1/val'
    normalize_stem_boxes(yolo_path, yolo_path)

if __name__ == "__main__":
    main()
