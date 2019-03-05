import os
import cv2
import shutil
import glob

import xml.etree.ElementTree as ET
from random import shuffle

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *

base_path = '/media/deepwater/Data2/Louis/RetinaNet/datasets/'

folders = [
	'training_set/mais_haricot_feverole_pois/50/1',
	'training_set/mais_haricot_feverole_pois/50/2',
	'training_set/mais_haricot_feverole_pois/60/1',
	'training_set/mais_haricot_feverole_pois/60/2',
	'training_set/mais_haricot_feverole_pois/100/1',
	'training_set/mais_haricot_feverole_pois/100/2',
]

directories = [os.path.join(base_path, folder) for folder in folders]

def parse_xml(directories, classes=None):
    xml_files = []
    for directory in directories:
        items = os.listdir(directory)
        items = [os.path.join(directory, item) for item in items if os.path.splitext(item)[1] == '.xml']
        xml_files += items

    allBoundingBoxes = BoundingBoxes()

    for file in xml_files:
        tree = ET.parse(file).getroot()

        name   = tree.find('path').text
        width  = tree.find('size').find('width').text
        height = tree.find('size').find('height').text

        for object in tree.findall('object'):
            class_id = object.find('name').text
            if classes and class_id not in classes:
                continue

            xmin = float(object.find('bndbox').find('xmin').text)
            ymin = float(object.find('bndbox').find('ymin').text)
            xmax = float(object.find('bndbox').find('xmax').text)
            ymax = float(object.find('bndbox').find('ymax').text)

            bbox = BoundingBox(name, class_id, xmin, ymin, xmax, ymax, format=BBFormat.XYX2Y2, imgSize=(int(width), int(height)))
            allBoundingBoxes.addBoundingBox(bbox)

    return allBoundingBoxes

if __name__ == '__main__':
    classes = ['mais', 'haricot', 'carotte']

    allBoundingBoxes = parse_xml(directories, classes)
    print('count: {}'.format(allBoundingBoxes.count()))
    print(allBoundingBoxes.stats())
