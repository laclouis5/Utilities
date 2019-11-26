# ToolBox for Dealing with Bounding Box Objects and Evaluating mAP for Detection Task
Based on this [this repo](https://github.com/rafaelpadilla/Object-Detection-Metrics#how-to-use-this-project).

# Usage
The framework is built around:

- A `BoundingBox` class that represents a rectangular box along with its label and confidence, if it's a detection box.
- A `BoundingBoxes` class that is an iterable object heriting from `Sequence`. You can use usual methods such has:
    - `boxes.append(box)` and `boxes.extend(otherBoxes)`
    - `boxes.remove(box)` and `boxes.clear()`
    - `box = boxes[index]`, `boxes[index] = box` and `boxes.insert(index, box)`
    - `boxes = gt_boxes + det_boxes` (join two `BoundingBoxes` objects)
    - etc...
- A `Parser` package with functions to parse Yolo, Pascal VOC and Coco annotations to `BoundingBoxes`.
- An `Evaluator` class to compute mAP@0.5 and Coco AP.
- Misc utilities to convert `BoundingBoxes` object to various database format including Yolo and Coco.

## Example
```Python
# Import things

coco_boxes = Parser.parse_coco("path_to_json_gt_file", "path_to_json_det_file")
xml_boxes = Parser.parse_xml_folder("folder_with_xml_files", ["dog", "cat"])
yolo_boxes = Parser.parse_yolo_gt_folder("path_to_yolo_gts")
yolo_boxes.mapLabels({1: "dog", 2: "cat"})

# When using darknet framework:
for image_name in images:
    detections = performDetect(image_name, ...)
    yolo_boxes += Parser.parse_yolo_darknet_detections(detections, image_name)
    
all_boxes = coco_boxes + xml_boxes + yolo_boxes
mAP = Evaluator.getAP(all_boxes, 0.5)
cocoAP = Evaluator.getCocoAP(all_boxes)

Evaluator.PlotPrecisionRecallCurve(all_boxes, ...)
```

## BoundingBox
Object that stores a rectangular object detection box. 3 formats are supported:

- `BBFormat.XYWH`: `(x_center, y_center, width, height)`
- `BBFormat.XYX2Y2`: `(x_topLeft, y_topLeft, x_bottomRight, y_bottomRight)`

The box is internaly stored in a third format `(x_topLeft, y_topLeft, width, height)` without rounding. This box can be in 2 types of coordinates:

- `CoordinatesTypes.Relative`: coordinates are  normalized with image size and are in `[0, 1[`.
- `CoordinatesTypes.Absolute`: real box coordinates in pixels stored as `float` to avoid loosing precision with rounding errors.

The bounding box has also a label (`str`) and an optional `confidence` (`float` in `[0, 1]`) if the type of the box is `BBType.Detected` (`.GroundTruth` otherwise).

`BoundingBox` comes with setters to retreive the absolute or relative bounding box in the specified format and coordinates type.

If openCV is defined, a method to add the boudning box in an image is provided.

## BoundingBoxes
This object is a sub-class of `MutableSequence` and can be used like a standard array of `BoundingBox` objects. Filtering methods are provided to retreive specific data. A save function allows you to build a Yolo database. Methods to draw annotations on images are also present (joblib may be required for parallel computations).

## Evaluator
Evaluate a `BoundingBox` object that contains ground truths and assotiated detections with mAP and Coco AP. This class can draw precision-recall curves.

## Parsers
```Python
# Parse Pascal VOC style annotations
boxes = Parser.parse_xml_file(path_to_xml_file)
boxes = Parser.parse_xml_folder(path_to_folder)
boxes = Parser.parse_xml_directories([path_1, path_2, ...])
...
```
