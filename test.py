from skimage import io, data, filters, feature, color, exposure, morphology
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from PIL import Image, ImageDraw

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
