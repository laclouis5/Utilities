import os
import cv2
import shutil
import glob

import lxml.etree as ET
from random import shuffle

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *

def parse_xml(directories, classes=None):
    """
    Parses a list of directories with XML GT files labeled with labelImg
    and returns a boundingBoxes object holding boxes. Classes can be filtered.
    """
    xml_files = []
    for directory in directories:
        items = os.listdir(directory)
        [xml_files.append(os.path.join(directory, item)) for item in items if os.path.splitext(item)[1] == '.xml']

    allBoundingBoxes = BoundingBoxes()

    # Can be parallelized
    for file in xml_files:
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

            bbox = BoundingBox(name, class_id, xmin, ymin, xmax, ymax, format=BBFormat.XYX2Y2, imgSize=(int(width), int(height)))
            allBoundingBoxes.append(bbox)

    return allBoundingBoxes


def parse_yolo_folder(directories, classes=None):
    # Put here the code to parse Yolo subfolders and create a Parser Class
    # and create a Python Package
    return

def parse_yolo_dir(directories, classes=None):
    # Put here the code to parse Yolo subfolders and create a Parser Class
    # and create a Python Package
    return

def parse_yolo_file(directories, classes=None):
    # Put here the code to parse Yolo subfolders and create a Parser Class
    # and create a Python Package
    return
