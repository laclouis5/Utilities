# ToolBox for Dealing with Bounding Box Objects and Evaluating mAP
Based on this [this repo](https://github.com/rafaelpadilla/Object-Detection-Metrics#how-to-use-this-project).

# Usage
The framework is built around:

- A `BoundingBox` class that represents a rectangular box, its label and confidence if it's a detection box.
- A `BoundingBoxes` class that is an iterable object and that herits from all `Sequence` usual methods such has `.append()`, `.remove()` and subscript getter `box = boxes[index]`.
- A `Parser` class with static methods to parse Yolo and Pascal VOC annotations and that return `BoundingBox`.
- An `Evaluator` class to compute mAP@0.5 and Coco AP.

## BoundingBox
Object that stores a rectangular object detection box. 3 formats are supported:

- `BBFormat.XYWH`: `(x_center, y_center, width, height)`
- `BBFormat.XYX2Y2`: `(x_topLeft, y_topLeft, x_bottomRight, y_bottomRight)`

The box is internaly stored in a third format `(x_topLeft, y_topLeft, width, height)` without rounding. This box can be in 2 types of coordinates:

- `CoordinatesTypes.Relative`: coordinates are  normalized with image size and are in `[0, 1[`.
- `CoordinatesTypes.Absolute`: real box coordinates in pixels stored as `float` to avoid loosing precision with rounding errors.

The bounding box has also a label (`string`) and an optional `confidence` (`float` in `[0, 1]`) if the type of the box is `BBType.Detected` (`.GroundTruth` otherwise).

`BoundingBox` comes with setters to retreive the absolute or relative bounding box in the specified format and coordinates type.

If openCV is defined a method to add the rectangular box in an image is provided.

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

# Parse Yolo style annotations
boxes = Parser.parse_txt_file(path_to_xml_file)
boxes = Parser.parse_txt_folder(path_to_folder)
boxes = Parser.parse_txt_directories([path_1, path_2, ...])
```
