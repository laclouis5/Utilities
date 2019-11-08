import os
import lxml.etree as ET
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from utils import *
import PIL

# XML Parsers
def parse_xml_directories(directories, classes=None):
    boxes = BoundingBoxes()

    for directory in directories:
        boxes += parse_xml_folder(directory, classes=classes)

    return boxes


def parse_xml_folder(folder, classes=None):
    boxes = BoundingBoxes()

    for file in files_with_extension(folder, ".xml"):
        boxes += parse_xml_file(file, classes=classes)

    return boxes


# Yolo parsers
def parse_xml_file(file, classes=None):
    boxes = BoundingBoxes()
    tree = ET.parse(file).getroot()

    name   = os.path.join(os.path.dirname(tree.find('path').text), tree.find('filename').text)
    width  = tree.find('size').find('width').text
    height = tree.find('size').find('height').text

    for object in tree.findall('object'):
        class_id = object.find('name').text
        if classes and (class_id not in classes):
            continue

        xmin = float(object.find('bndbox').find('xmin').text)
        ymin = float(object.find('bndbox').find('ymin').text)
        xmax = float(object.find('bndbox').find('xmax').text)
        ymax = float(object.find('bndbox').find('ymax').text)

        box = BoundingBox(name, class_id, xmin, ymin, xmax, ymax, format=BBFormat.XYX2Y2, imgSize=(int(width), int(height)))
        boxes.append(box)

    return boxes


def parse_yolo_directories(directories, classes=None):
    boxes = BoundingBoxes()

    for directory in directories:
        boxes += parse_yolo_folder(directory, classes=classes)

    return boxes


def parse_yolo_folder(folder, classes=None):
    boxes = BoundingBoxes()

    for file in files_with_extension(folder, ".txt"):
        boxes += parse_yolo_file(file, classes=classes)

    return boxes


def parse_yolo_file(file, classes=None):
    '''
    Designed to read Yolo annotation files that are in the same folders
    as their corresponding image.
    '''
    boxes = BoundingBoxes()
    image_name = os.path.splitext(file)[0] + '.jpg'
    img_size = PIL.Image.open(image_name).size

    with open(file, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4])

        if classes is not None and label not in classes: continue

        box = BoundingBox(imageName=image_name, classId=label,x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, format=BBFormat.XYWH, imgSize=img_size, bbType=BBType.GroundTruth)
        boxes.append(box)

    return boxes


def parse_yolo_det_file(file, img_size=None, classes=None, bbFormat=BBFormat.XYWH, typeCoordinates=CoordinatesType.Relative):
    """
    If coordinates are relative you should provide img_size.
    """
    boxes = BoundingBoxes()

    with open(file, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, confidence, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4]), float(det[5])

        if classes is not None and label not in classes: continue

        box = BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, confidence=confidence, typeCoordinates=typeCoordinates, format=bbFormat, imgSize=img_size, bbType=BBType.Detected)
        boxes.append(box)

    return boxes


def parse_yolo_detector_detections(detections, image_name, img_size=None, classes=None):
    """
    Parses a detection returned by yolo detector wrapper.
    """
    boxes = BoundingBoxes()

    for detection in detections:
        (label, confidence, box) = detection
        (x_topLeft, y_topLeft, x_bottomRight, y_bottomRight) = box

        if classes is not None and label not in classes: continue

        box = BoundingBox(imageName=image_name, classId=label, x=x_topLeft, y=y_topLeft, w=x_bottomRight, h=y_bottomRight, confidence=confidence, typeCoordinates=CoordinatesType.Absolute, format=BBFormat.XYX2Y2, imgSize=img_size, bbType=BBType.Detected)

        boxes.append(box)

    return boxes
