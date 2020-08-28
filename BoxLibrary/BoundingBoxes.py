from .BoundingBox import BoundingBox
from .utils import *

from random import shuffle
import matplotlib.pyplot as plt
from collections.abc import MutableSequence

import cv2 as cv
from joblib import Parallel, delayed

class BoundingBoxes(MutableSequence):
    def __init__(self, bounding_boxes=None):
        if bounding_boxes is not None:
            self._boundingBoxes = bounding_boxes
        else:
            self._boundingBoxes = []

    def __len__(self):
        return len(self._boundingBoxes)

    def __getitem__(self, index):
        return self._boundingBoxes[index]

    def __setitem__(self, index, item):
        self._boundingBoxes[index] = item

    def __delitem__(self, index):
        del self._boundingBoxes[index]

    def __add__(self, otherBoxes):
        return BoundingBoxes(self._boundingBoxes + otherBoxes._boundingBoxes)

    def insert(self, index, box):
        self._boundingBoxes.insert(index, box)

    def getClasses(self):
        """
        Returns a list of BoundingBoxes class Ids. The list is sorted and all entries are unique.

        Returns:
            [str]: The sorted list of clas ids.
        """
        return sorted(set([box.getClassId() for box in self]))

    def getNames(self):
        """
        Returns a list of BoundingBoxes image names. The list is sorted and all entries are unique.

        Returns:
            [str]: The sorted list of image names.
        """
        return sorted(set([box.getImageName() for box in self]))

    def getBoundingBoxesByType(self, bbType):
        return BoundingBoxes([d for d in self if d.getBBType() == bbType])

    def getBoundingBoxesByImageName(self, imageName):
        return BoundingBoxes([d for d in self if d.getImageName() == imageName])

    def getBoundingBoxByClass(self, classId):
        if isinstance(classId, list):
            return BoundingBoxes([bb for bb in self if bb.getClassId() in classId])
        else:
            return BoundingBoxes([bb for bb in self if bb.getClassId() == classId])

    def getDetectionBoxesAsNPArray(self):
        import numpy as np
        detection_boxes = self.getBoundingBoxesByType(BBType.Detected)
        array = [[*box.getAbsoluteBoundingBox(BBFormat.XYX2Y2), box.getConfidence()] for box in detection_boxes]
        return np.array(array)

    def imageSize(self, imageName):
        """
        Returns the size of the specified image.

        Parameters:
            imageName (str): The name of the image.

        Raises:
            IOError: Raised if image cannot be found.

        Returns:
            (float, float): The size of the image: (width, height).
        """
        box = next((box for box in self if box.getImageName() == imageName), None)
        if box is None:
            raise IOError("Image with name '{}' not found, can't return its size.".format(imageName))
        return box.getImageSize()

    def mapLabels(self, mapping):
        """
        Map the BoundingBox classIds attribute to another value specified in a dictionnary. this method mutates the instances.

        Parameters:
            mapping (dict): A dictionnary that links the current classId to the new value. for instance if the possible values for classId are [0, 1, 2] and new values are ["car", "pedestrian", "bike"] the mapping dictionnary is: {0: "car", 1: "pedestrian", 2: "bike"}.
        """
        mapping = {str(key): value for (key, value) in mapping.items()}
        for box in self:
            box.setClassId(mapping[box.getClassId()])

    def clip(self, size=None):
        """
        Crops boxes to fit into a given rectangle. It correponds to the partition that is common with the given rectangle.

        Parameters:
            size (tuple(float, float)): The (width, height) in absolute coordinates of the bounding frame. If size is None and an image size is specified in the BoundingBox object, the last in used.
        """
        assert len(self.getNames()) == 1, "Boxes should belong to only one image when cliping."

        for box in self:
            box.clip(size)

    def cliped(self, size=None):
        """
        Returns the croped boxes that fit into a given rectangle. It correponds to the partition that is common with the given rectangle.

        Parameters:
            size (tuple(float, float)): The (width, height) in absolute coordinates of the bounding frame. If size is None and an image size is specified in the BoundingBox object, the last in used.

        Returns:
            BoundingBoxes: The croped boxes.
        """
        boxes = self.copy()
        boxes.clip(size)
        return boxes

    def movedBy(self, dx, dy, typeCoordinates=CoordinatesType.Absolute, imgSize=None):
        """
        Returns the boxes moved by the vector (dx, dy). By default coordinates are in absolute values. If using relative coordinates you should specify an image size if not already stored in the BoundingBox object.

        Parameters:
            dx (float):
                The horizontal offset.
            dy (float):
                The vertical offset.
            typeCoordinates (optional CoordinatesType):
                The type of coordinates used. If 'Relative', imgSize should be informed either as a property of BoundingBox or as parameter to this method.
            imgSize (tuple(float, float)):
                The image size: (widht, height).

        Raises:
            IOError: imgSize should be informed when using CoordinatesType.Relative
            AssertionError: Only use this method for a BoundingBoxes object storing boxes for one image.

        Returns:
            BoundingBoxes: The moved boxes.
        """
        assert len(self.getNames()) <= 1, "'movedBy()' is only available for BoundingBoxes representing one image. Image names: {}".format(self.getNames())

        return BoundingBoxes([box.movedBy(dx, dy, typeCoordinates, imgSize) for box in self])

    def moveBy(self, dx, dy, typeCoordinates=CoordinatesType.Absolute, imgSize=None):
        """
        Moves the boxes by the vector (dx, dy). By default coordinates are in absolute values. If using relative coordinates you should specify an image size if not already stored in the BoundingBox object.

        Parameters:
            dx (float):
                The horizontal offset.
            dy (float):
                The vertical offset.
            typeCoordinates (optional CoordinatesType):
                The type of coordinates used. If 'Relative', imgSize should be informed either as a property of BoundingBox or as parameter to this method.
            imgSize (tuple(float, float)):
                The image size: (widht, height).

        Raises:
            IOError: imgSize should be informed when using CoordinatesType.Relative.
            AssertionError: Only use this method for a BoundingBoxes object storing boxes for one image.
        """
        assert len(self.getNames()) <= 1, "'moveBy()' is only available for BoundingBoxes representing one image. Image names: {}".format(self.getNames())

        for box in self:
            box.moveBy(dx, dy, typeCoordinates, imgSize)

    def boxes_in(self, rect):
        boxes = []

        for box in self:
            if box.centerIsIn(rect):
                boxes.append(box)

        return BoundingBoxes(boxes)

    def shuffleBoundingBoxes(self):
        shuffle(self._boundingBoxes)

    def count(self, bbType=None):
        if bbType is None:  # Return all bounding boxes
            return len(self)
        count = 0
        for d in self:
            if d.getBBType() == bbType:  # get only specified bb type
                count += 1
        return count

    def stats(self):
        print("{:<20} {:<15} {}".format(
            "Label:", "Nb Images:", "Nb Annotations:"))
        for label in self.getClasses():
            boxes = self.getBoundingBoxByClass(label)
            nb_images = len(set([item.getImageName() for item in boxes]))
            nb_annot = len(boxes)

            print("{:<20} {:<15} {}".format(label, nb_images, nb_annot))

    def save(self, type_coordinates=CoordinatesType.Relative, format=BBFormat.XYC, save_dir=None):
        """
        Save all bounding boxes as Yolo annotation files in the specified directory.

        The files are TXT files named with the image name provided in each BoundingBox. Each line of the file is in the format "classId <optional confidence> coord1 coord2 coord3 coord4".

        Parameters:
            type_coordinates (CoordinatesType):
                The type of the coordinnates, either absolute or relative.
            format (BBFormat):
                The bounding box format, can be [xMin, yMin, xMax, yMax], [xCenter, yCenter, width, height] or other. See BBFormat for more information.
            save_dir (str):
                The directory where to save the files as TXT files.
        """
        for imageName in self.getNames():
            description = ""
            for box in self.getBoundingBoxesByImageName(imageName):
                description += box.description(type_coordinates, format)

            fileName = os.path.splitext(imageName)[0] + ".txt"
            if save_dir is not None:
                create_dir(save_dir)
                fileName = os.path.join(save_dir, os.path.basename(fileName))

            with open(fileName, "w") as f:
                f.write(description)

    def squareStemBoxes(self, ratio=0.075):
        boxes = BoundingBoxes()
        for box in self:
            if "stem" not in box.getClassId():
                boxes.append(box)
            else:
                (x, y, _, _) = box.getRelativeBoundingBox()
                (img_w, img_h) = box.getImageSize()
                new_size = ratio * min(img_w, img_h)
                w = new_size / img_w
                h = new_size / img_h
                boxes.append(BoundingBox(imageName=box.getImageName(), classId=box.getClassId(), x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, imgSize=box.getImageSize(), bbType=box.getBBType(), classConfidence=box.getConfidence, format=BBFormat.XYWH))
        return boxes

    def plotHistogram(self):
        areas = []
        for label in self.getClasses():
            boxes = self.getBoundingBoxByClass(label)
            areas.append([bbox.getArea() for bbox in boxes])
        # plt.hist(areas, bins=200, stacked=True, density=True, label=labels)
        # plt.hist([bbox.getArea() for bbox in boxes], bins=100)
        plt.title('Stacked histogram of bounding box area')
        # plt.title('Histogram for class "{}"'.format(c))
        plt.legend()
        plt.xlabel('Area in pixels')
        plt.ylabel('Normalized histogram')
        # plt.ylabel('Number of bbox')
        plt.show()

    def drawCVImage(self, image, imageName):
        """
        Draws boxes on a CV image.

        Parameters:
            image (CVImage):
                Reference to the image to draw to.
            imageName (str):
                The image name to query boxes to be drawn.

        Returns:
            CVImage: The reference to the resulting image.
        """
        boxes = self.getBoundingBoxesByImageName(imageName)
        for box in boxes:
            box.addIntoImage(image)
        return image

    def drawImage(self, name, save_dir="annotated_images/"):
        """
        Draws boxes on its corresponding image and save it to disk.

        The name of the image should be the absolute path to the actual image. You should store the real image path in BoundingBox._imageName if you want to use this method.

        Parameters:
            name (str):
                The image name to query boxes to be drawn.
            save_dir (str):
                The path where to store the image.
        """

        create_dir(save_dir)

        save_name = os.path.join(save_dir, os.path.basename(name))

        image = self.drawCVImage(cv.imread(name), name)
        cv.imwrite(save_name, image)

    def drawImageCenter(self, name, save_dir="annotated_images/"):
        create_dir(save_dir)
        save_name = os.path.join(save_dir, os.path.basename(name))

        image = cv.imread(name)
        for box in self.getBoundingBoxesByImageName(name):
            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)
            label = box.getClassId()
            color = (0, 255, 0) if box.getBBType() == BBType.GroundTruth else (0, 0, 255)

            cv.circle(image, (int(x), int(y)), 5, color, thickness=cv.FILLED)

        cv.imwrite(save_name, image)

    def drawAll(self, save_dir="annotated_images/"):
        """
        Draws all boxes on their corresponding images and save them to disk. You should store the real image paths in BoundingBox._imageName if you want to use this method.

        Warning:
            This method uses parallel computing with joblib package to accelerate the process. joblib should be installed.

        Parameters:
            save_dir (str):
                The path where to store the image.
        """

        names = self.getNames()
        save_dir = [save_dir for _ in range(len(names))]

        Parallel(n_jobs=-1, backend="multiprocessing")(delayed(self.drawImage)(name, sd) for (name, sd) in zip(names, save_dir))

    def drawAllCenters(self, save_dir="annotated_images/"):
        names = self.getNames()
        save_dir = [save_dir for _ in range(len(names))]

        Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(self.drawImageCenter)(name, sd) for (name, sd) in zip(names, save_dir)
        )

    def erase_image_names(self):
        for box in self:
            box._imageName = "no_name"

    def copy(self):
        return BoundingBoxes([box.copy() for box in self._boundingBoxes])

    def __str__(self):
        description = ""
        for box in self:
            description += str(box) + "\n"
        return description
