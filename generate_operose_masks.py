import argparse
from collections import defaultdict
from joblib import Parallel, delayed
import lxml.etree as ET
import os
from PIL import Image, ImageDraw


class ROSEAnnotation:

    def __init__(self, name, points, kind=None):
        self.name = name
        self.points = points
        self.kind = kind


class ROSEImageAnnotation:

    def __init__(self, xml_file):
        root = ET.parse(xml_file).getroot()
        self.image_name = root.find("metadata").find("imageFiles").find("RGB").text
        self.crops = []

        items = root.find("data").find("clippings").findall("clipping")

        for item in items:
            crop_name = item.find("name").text
            kind = item.find("class").text

            x_s = item.find("points").findall("x")
            y_s = item.find("points").findall("y")

            points = [(float(x.text), float(y.text))
                for (x, y) in zip(x_s, y_s)]

            self.crops.append(ROSEAnnotation(crop_name, points, kind))


def create_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def files_with_ext(directory, extension):
    return [os.path.join(directory, file)
        for file in os.listdir(directory)
        if os.path.splitext(file)[1] == extension]


def generate_operose_masks(folder, save_dir=None, n_jobs=-1, verbose=True):
    """
    Draws masks of crops (haricot, mais or other like pois).
    """
    if save_dir is None: save_dir = "operose_masks"
    create_dir(save_dir)
    xml_files = files_with_ext(folder, ".xml")

    def inner(file):
        annotation = ROSEImageAnnotation(file)
        image = Image.open(os.path.join(folder, annotation.image_name))
        img_size = image.size
        id = 0

        for crop in annotation.crops:
            if crop.kind != "crop": continue

            img = Image.new(mode="1", size=img_size)
            draw = ImageDraw.Draw(img)
            draw.polygon(crop.points, fill=1)
            out_name = os.path.basename(annotation.image_name)
            out_name = os.path.splitext(out_name)[0] +  f"_{id}" + ".png"
            out_name = os.path.join(save_dir, out_name)
            img.save(out_name)
            id += 1

    verbose = 10 if verbose else 0
    Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(inner)(file) for file in xml_files)


def visualize_masks(folder, save_dir=None, n_jobs=-1, verbose=True):
    """
    Draw weeds in red, 'haricot' and 'mais' in green and other crops in blue.
    """
    if save_dir is None: save_dir = "operose_imgs"
    create_dir(save_dir)
    xml_files = files_with_ext(folder, ".xml")

    def inner(file):
        annotation = ROSEImageAnnotation(file)
        image = Image.open(os.path.join(folder, annotation.image_name))
        img_size = image.size

        for crop in annotation.crops:
            draw = ImageDraw.Draw(image)
            color = (0, 255, 0) if crop.kind == "crop" else (255, 0, 0)
            if crop.name not in ["haricot", "mais"] and crop.kind == "crop":
                color = (0, 0, 255)

            draw.polygon(crop.points, outline=color)

        out_name = os.path.join(save_dir, annotation.image_name)
        image.save(out_name)

    verbose = 10 if verbose else 0
    Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(inner)(file) for file in xml_files)


def get_stats(folder):
    xml_files = files_with_ext(folder, ".xml")
    annotations = [ROSEImageAnnotation(file) for file in xml_files]

    stats = defaultdict(lambda: defaultdict(int))

    for annotation in annotations:
        for crop in annotation.crops:
            stats[crop.kind][crop.name] += 1

    for (kind, names) in stats.items():
        print(kind)
        for (name, count) in names.items():
            print(f"  - {name}: {count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Converts annotations in the Operose format (xml + image) into binary masks.")
    parser.add_argument("directory", type=str,
        help="The directory where to parse Operose xml files.")
    parser.add_argument("--save_dir", "-s", type=str, default=None,
        help="The directory where to save png masks.")
    parser.add_argument("--n_jobs", "-j", default=-1, type=int,
        help="Number of parallel jobs. Default is maximum possible.")
    return parser.parse_args()


def main():
    args = parse_args()
    # generate_operose_masks(args.directory, args.save_dir, args.n_jobs)
    # visualize_masks(args.directory, args.save_dir, args.n_jobs)
    get_stats(args.directory)


if __name__  == "__main__":
    main()
