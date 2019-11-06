from BoundingBox import *
from utils import *
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

class BoundingBoxes:
    def __init__(self, bounding_boxes=[]):
        self._boundingBoxes = bounding_boxes

    def addBoundingBox(self, bb):
        if isinstance(bb, list):
            self._boundingBoxes.extend(bb)
        else:
            self._boundingBoxes.append(bb)

    def removeBoundingBox(self, _boundingBox):
        for d in self._boundingBoxes:
            if d == _boundingBox:
                del self._boundingBoxes[d]
                return

    def removeAllBoundingBoxes(self):
        self._boundingBoxes = []

    def getBoundingBoxes(self):
        return self._boundingBoxes

    def getBoundingBoxByClass(self, classId):
        boundingBoxes = []
        for d in self._boundingBoxes:
            if d.getClassId() == classId:  # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def mapLabels(self, dic):
        for bbox in self._boundingBoxes:
            bbox._classId = dic[bbox.getClassId()]

    def shuffleBoundingBoxes(self):
        shuffle(self._boundingBoxes)

    def getClasses(self):
        classes = []
        for d in self._boundingBoxes:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return sorted(classes)

    def getNames(self):
        names = []
        for bbox in self._boundingBoxes:
            name = bbox.getImageName()
            if name not in names:
                names.append(name)
        return sorted(names)

    def getBoundingBoxesByType(self, bbType):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getBBType() == bbType]

    def getBoundingBoxesByImageName(self, imageName):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getImageName() == imageName]

    def plotHistogram(self):
        areas   = []
        classes = sorted(self.getClasses())
        labels  = ['{}'.format(c) for c in classes]
        for c in classes:
            boxes = self.getBoundingBoxByClass(c)
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

    def __len__(self):
        return len(self._boundingBoxes)

    def count(self, bbType=None):
        if bbType is None:  # Return all bounding boxes
            return len(self._boundingBoxes)
        count = 0
        for d in self._boundingBoxes:
            if d.getBBType() == bbType:  # get only specified bb type
                count += 1
        return count

    def stats(self):
        print("{:<20} {:<15} {}".format("Label:", "Nb Images:", "Nb Annotations:"))
        for label in self.getClasses():
            boxes = self.getBoundingBoxByClass(label)
            nb_images = len(set([item.getImageName() for item in boxes]))
            nb_annot  = len(boxes)

            print("{:<20} {:<15} {}".format(label, nb_images, nb_annot))

    def normalizedSquareBoxes(self):
        boxes = []
        for box in self.getBoundingBoxes():
            if "tige" in box.getClassId():
                (x, y, _, _) = box.getRelativeBoundingBox()
                w = 0.05
                h = 0.05
                box = BoundingBox(imageName=box.getImageName(), classId=box.getClassId(), x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, imgSize=box.getImageSize(), bbType=box.getBBType(), classConfidence=box.getConfidence(), format=BBFormat.XYWH)
                boxes.append(box)
            else:
                boxes.append(box)
        return BoundingBoxes(bounding_boxes=boxes)

    def squareStemBoxes(self, ratio=0.075):
        boxes = []
        print("In squareStem")
        for box in self.getBoundingBoxes():
            if "stem" not in box.getClassId():
                boxes.append(box)
            else:
                (x, y, _, _) = box.getRelativeBoundingBox()
                (img_w, img_h) = box.getImageSize()
                new_size = ratio * min(img_w, img_h)
                w = new_size / img_w
                h = new_size / img_h
                boxes.append(BoundingBox(imageName=box.getImageName(), classId=box.getClassId(), x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, imgSize=box.getImageSize(), bbType=box.getBBType(), classConfidence=box.getConfidence, format=BBFormat.XYWH))
        return BoundingBoxes(bounding_boxes=boxes)


    def save(self, type_coordinates=CoordinatesType.Relative, format=BBFormat.XYWH, save_dir=None):
        for imageName in self.getNames():
            description = ""
            for box in self.getBoundingBoxesByImageName(imageName):
                bbox = (0, 0, 0, 0)
                if type_coordinates == CoordinatesType.Relative:
                    bbox = box.getRelativeBoundingBox()
                else:
                    bbox = box.getAbsoluteBoundingBox(format)
                    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

                if box.getBBType() == BBType.Detected:
                    description += "{} {} {} {} {} {}\n".format(box.getClassId(), box.getConfidence(), bbox[0], bbox[1], bbox[2], bbox[3])
                else:
                    description += "{} {} {} {} {}\n".format(box.getClassId(), bbox[0], bbox[1], bbox[2], bbox[3])

            fileName = os.path.splitext(imageName)[0] + ".txt"
            if save_dir is not None:
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                fileName = os.path.join(save_dir, os.path.basename(fileName))

            with open(fileName, "w") as f:
                f.write(description)


    def meanSize(self):
        import matplotlib.pyplot as plt
        boundingBoxes = self.normalizedSquareBoxes()
        for label in boundingBoxes.getClasses():
            boxes = boundingBoxes.getBoundingBoxByClass(label)
            widths = []
            heights = []
            for box in boxes:
                bbox = box.getAbsoluteBoundingBox()
                widths.append(bbox[2])
                heights.append(bbox[3])
            plt.scatter(widths, heights)
            plt.title(label)
            plt.xlim((0, 1000))
            plt.ylim((0, 1000))
            plt.show()

    def clone(self):
        newBoundingBoxes = BoundingBoxes()
        for d in self._boundingBoxes:
            det = BoundingBox.clone(d)
            newBoundingBoxes.addBoundingBox(det)
        return newBoundingBoxes

    def drawAllBoundingBoxes(self, image, imageName):
        bbxes = self.getBoundingBoxesByImageName(imageName)
        for bb in bbxes:
            if bb.getBBType() == BBType.GroundTruth:  # if ground truth
                image = add_bb_into_image(image, bb, color=(0, 255, 0))  # green
            else:  # if detection
                image = add_bb_into_image(image, bb, color=(255, 0, 0))  # red
        return image

    # def drawAllBoundingBoxes(self, image):
    #     for gt in self.getBoundingBoxesByType(BBType.GroundTruth):
    #         image = add_bb_into_image(image, gt ,color=(0,255,0))
    #     for det in self.getBoundingBoxesByType(BBType.Detected):
    #         image = add_bb_into_image(image, det ,color=(255,0,0))
    #     return image
