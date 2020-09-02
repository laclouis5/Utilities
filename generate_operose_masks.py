import argparse
from joblib import Parallel, delayed
import lxml.etree as ET
import os
from PIL import Image, ImageDraw


class ROSEAnnotation:

    def __init__(self, name, points):
        self.name = name
        self.points = points


class ROSEImageAnnotation:

    def __init__(self, xml_file):
        root = ET.parse(xml_file).getroot()
        self.image_name = root.find("metadata").find("imageFiles").find("RGB").text
        self.crops = []

        items = root.find("data").find("clippings").findall("clipping")

        for item in items:
            if item.find("class").text != "crop":
                continue

            crop_name = item.find("name").text

            x_s = item.find("points").findall("x")
            y_s = item.find("points").findall("y")

            points = [(float(x.text), float(y.text)) for (x, y) in zip(x_s, y_s)]

            self.crops.append(ROSEAnnotation(crop_name, points))


def create_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def files_with_ext(directory, extension):
    return [os.path.join(directory, file)
        for file in os.listdir(directory)
        if os.path.splitext(file)[1] == extension]


def generate_operose_masks(folder, save_dir="operose_masks/", n_jobs=-1, verbose=True):
    create_dir(save_dir)
    xml_files = files_with_ext(folder, ".xml")

    def inner(file):
        annotation = ROSEImageAnnotation(file)
        image = Image.open(os.path.join(folder, annotation.image_name))
        img_size = image.size
        id = 0

        for crop in annotation.crops:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Converts annotations in the Operose format (xml + image) into binary masks.")
    parser.add_argument("directory", type=str,
        help="The directory where to parse Operose xml files.")
    parser.add_argument("--save_dir", "-s", default="operose_masks", type=str,
        help="The directory where to save png masks.")
    parser.add_argument("--n_jobs", "-j", default=-1, type=int,
        help="Number of parallel jobs. Default is maximum possible.")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_operose_masks(args.directory, args.save_dir, args.n_jobs)


if __name__  == "__main__":
    main()
