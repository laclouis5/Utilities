'''
This library provides tools for converting VOC style XML to different formats (RetinaNet, YOLO).
'''
import os
import shutil
import lxml.etree as ET
from random import shuffle

from glob import glob, iglob
# from skimage import io
import PIL
import json
import my_library
import random
from pathlib import Path
from tqdm.contrib import tenumerate, tzip
import tqdm as tqdm

from BoxLibrary import *


def clean_xml_files(folders):
	for folder in folders:
		for xml_file in Path(folder).glob("*.xml"):
			tree = ET.parse(xml_file.open()).getroot()
			tree.find("path").text = str(xml_file)
			xml_file.write_text(ET.tostring(tree, encoding="unicode"))


def rename_all_files(folders):
	i = 0
	for folder in folders:
		for file in sorted(files_with_extension(folder, ".xml")):
			image = os.path.splitext(file)[0] + ".jpg"
			new_name = "im_{:09d}".format(i)
			new_image_name = new_name + ".jpg"
			new_xml_name = new_name + ".xml"
			new_xml_path = os.path.join(os.path.split(file)[0], new_xml_name)
			new_image_path = os.path.join(os.path.split(image)[0], new_image_name)
			print(new_name)
			print(new_image_name)
			print(new_xml_name)
			print(new_image_path)
			print(new_xml_path)

			tree = ET.parse(file).getroot()
			tree.find("filename").text = new_image_name
			tree.find("path").text = new_xml_path

			with open(file, "w") as f:
				content = ET.tostring(tree, encoding="unicode")
				f.write(content)

			os.rename(file, new_xml_path)
			os.rename(image, new_image_path)

			i += 1


def normalized_stem_boxes(boxes,
	ratio=7.5/100, labels=["haricot_tige", "mais_tige", "poireau_tige"]
):
	normalized_boxes = BoundingBoxes()

	for box in boxes:
		if box.getClassId() in labels:
			normalized_boxes.append(box.normalized(ratio))
		else:
			normalized_boxes.append(box)

	return normalized_boxes


def xml_to_csv_2(boundingboxes, save_dir="", ratio=0.8, no_obj_dir=None):
	names = boundingboxes.getNames()
	shuffle(names)

	nb_train = int(ratio * len(names))

	train = []
	val = []

	for (i, name) in enumerate(names):
		boxes = boundingboxes.getBoundingBoxesByImageName(name)

		for box in boxes:
			(xmin, ymin, xmax, ymax) = box.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
			label = box.getClassId()
			name = box.getImageName()

			line = "{},{},{},{},{},{}\n".format(
				name, int(xmin), int(ymin), int(xmax), int(ymax), label)

			if i < nb_train:
				train.append(line)
			else:
				val.append(line)

	if no_obj_dir != None:
		no_obj_images = [os.path.join(no_obj_dir, item)
			for item in os.listdir(no_obj_dir)
			if os.path.splitext(item)[1] == '.jpg']

		nb_train_bo_obj = int(ratio * len(no_obj_images))

		for j, no_obj_image in enumerate(no_obj_images):
			no_obj_line = "{},,,,,\n".format(no_obj_image)
			if j < nb_train_bo_obj:
				train.append(no_obj_line)
			else:
				val.append(no_obj_line)
	with open(os.path.join(save_dir, "train.csv"), "w") as f_train:
		f_train.writelines(train)
	with open(os.path.join(save_dir, "val.csv"), "w") as f_val:
		f_val.writelines(val)


# Outdated, need update
def create_VOC_database(database_path, folders, test_folders, names_to_labels, no_eval=False):
	'''
	Takes as input folders to images with corresponding VOC XML files and outputs
	a VOC database ready for LMDB.

	DATABASE_PATH - Directory where the database will be created and saved.

	FOLDERS - List of strings representing absolute paths to directories with training images and XML.

	TEST_FOLDER - List of abs path to test images and XML for validation.

	NAMES_TO_LABELS - Dictionary with object name as key and label number as value. This makes
	explicit the relation object-label contrary to a list.

	NO_EVAL - Set to true to train with eval files (when dealing with Operose data).
	'''
	print('Creating VOC database...')

	image_folder = os.path.join(database_path, "JPEGImages")
	annotations_folder = os.path.join(database_path, "Annotations")
	imageset_folder = os.path.join(database_path, "ImageSets/Main")

	trainval_file = open(os.path.join(imageset_folder, 'trainval.txt'), 'w')

	with open(os.path.join(imageset_folder, 'test.txt'), 'w') as test_file:
		# Stuff to count things
		image_counter = -1

		for folder in folders + test_folders:
			for file in os.listdir(folder):
				# Only process XML files
				if (os.path.splitext(file)[1] != '.xml'): continue

				image_counter += 1

				# Parse XML tree
				tree = ET.parse(os.path.join(folder, file)).getroot()

				# Ingnore Operose evaluation files
				if 'eval' in tree.find('filename').text and no_eval: continue

				# Retreive path
				image_filename = tree.find('filename').text
				basepath = os.path.split(tree.find('path').text)[0]
				xml_path = os.path.join(basepath, file)
				image_path = os.path.join(basepath, image_filename)

				# Copy XML files to the new directory
				shutil.copy(image_path, image_folder)
				shutil.copy(xml_path, annotations_folder)

				# Rename with a unique name
				new_image_name = '{}.jpg'.format(image_counter)
				new_xml_name = '{}.xml'.format(image_counter)

				os.rename(
					os.path.join(image_folder, image_filename),
					os.path.join(image_folder, new_image_name))
				os.rename(
					os.path.join(annotations_folder, file),
					os.path.join(annotations_folder, new_xml_name))

				# Change 'path' and 'filename' fields to reflect the name changes.
				new_xml_file_path = os.path.join(annotations_folder, new_xml_name)
				new_tree = ET.parse(new_xml_file_path).getroot()

				new_tree.find('filename').text = new_image_name
				new_tree.find('path'). text = new_xml_file_path

				# Remove labels not used for training
				for obj in new_tree.findall('object'):
					if obj.find('name').text not in names_to_labels.keys():
						new_tree.remove(obj)

				# Update the XML file
				with open(new_xml_file_path, 'w') as file:
					tree_str = ET.tostring(new_tree, encoding='unicode')
					file.write(tree_str)

				# Write TXT files for training and testing
				if folder in folders:
					trainval_file.write(os.path.splitext(new_image_name)[0] + '\n')
				elif folder in test_folders:
					test_file.write(os.path.splitext(new_image_name)[0] + '\n')

		trainval_file.close()

	print('Done!')

def xml_to_yolo_3(boundingBoxes, yolo_dir, names_to_labels, ratio=0.8, shuffled=True):
	train_dir = os.path.join(yolo_dir, 'train')
	val_dir = os.path.join(yolo_dir, 'val')
	train_file = os.path.join(yolo_dir, 'train.txt')
	val_file = os.path.join(yolo_dir, 'val.txt')

	if not os.path.isdir(yolo_dir):
		os.mkdir(yolo_dir)
	if not os.path.isdir(train_dir):
		os.mkdir(train_dir)
	if not os.path.isdir(val_dir):
		os.mkdir(val_dir)

	boxes_by_name = boundingBoxes.getBoxesBy(lambda box: box.getImageName())
	names = sorted(boxes_by_name.keys())
	new_names = []

	if shuffled == True:
		random_gen = random.Random(498_562_751)
		names = random_gen.sample(names, len(names))

	number_train = round(ratio*len(names))

	for (i, name) in tenumerate(names):
		yolo_rep = []
		img_path = os.path.splitext(name)[0] + '.jpg'
		idenfier = 'im_{}'.format(i)
		new_names.append(idenfier + ".jpg")

		save_dir = train_dir if i < number_train else val_dir
		for box in boxes_by_name[name]:
			label = names_to_labels[box.getClassId()]
			x, y, w, h = box.getRelativeBoundingBox()

			yolo_rep.append('{} {} {} {} {}\n'.format(label, x, y, w, h))

		with open(os.path.join(save_dir, idenfier + '.txt'), 'w') as f_write:
			f_write.writelines(yolo_rep)

		shutil.copy(img_path, os.path.join(save_dir, idenfier + '.jpg'))

	with open(train_file, "w") as f:
		for item in new_names[:number_train]:
			relative_path = os.path.split(item)[1]
			new_path = os.path.join("data/train/", relative_path)
			f.write(new_path + "\n")

	with open(val_file, "w") as f:
		for item in new_names[number_train:]:
			relative_path = os.path.split(item)[1]
			new_path = os.path.join("data/val/", relative_path)
			f.write(new_path + "\n")

def xml_to_yolo_4(boxes, label_map, save_dir="yolo/", ratio=0.8, shuffle=True, no_obj_dir=None):
	save_dir = Path(save_dir)

	train_dir = save_dir / "train"
	valid_dir = save_dir / "val"

	save_dir.mkdir()
	train_dir.mkdir()
	valid_dir.mkdir()

	boxes_by_name = boxes.getBoxesBy(lambda box: box.getImageName())
	image_names = sorted(boxes_by_name.keys())
	nb_train = round(ratio * len(boxes_by_name))
	new_names = []

	if shuffle:
		random_gen = random.Random(498_562_751)
		image_names = random_gen.sample(image_names, len(image_names))

	for i, image_name in tenumerate(image_names, unit="img"):
		image_boxes = boxes_by_name[image_name]
		folder = train_dir if i < nb_train else valid_dir

		new_image_name = folder / Path(image_name).with_stem(f"im_{i:06}").name
		new_names.append(new_image_name)
		description = "\n".join("{} {} {} {} {}".format(label_map[box.getClassId()], *box.getRelativeBoundingBox()) for box in image_boxes)

		new_image_name.with_suffix(".txt").write_text(description)
		shutil.copy(image_name, new_image_name)

	(save_dir / "train.txt").write_text("".join(f"{Path('data/train') / n.name}\n" for n in new_names[:nb_train]))
	(save_dir / "val.txt").write_text("".join(f"{Path('data/val') / n.name}\n" for n in new_names[nb_train:]))

	if no_obj_dir:
		no_obj_dir = Path(no_obj_dir)
		image_names = sorted(list(no_obj_dir.glob("*.jpg")))
		nb_train = round(ratio * len(image_names))
		new_names = []

		if shuffle:
			rand_gen = random.Random(478_737_303)
			image_names = rand_gen.sample(image_names, len(image_names))
		
		for i, image_name in tenumerate(image_names):
			folder = train_dir if i < nb_train else valid_dir
			new_image_name = folder / Path(image_name).with_stem(f"im_no_obj_{i:06}").name
			new_names.append(new_image_name)
			shutil.copy(image_name, new_image_name)
			new_image_name.with_suffix(".txt").touch()
		
		with (save_dir / "train.txt").open("a") as f:
			f.write("".join(f"{Path('data/train') / n.name}\n" for n in new_names[:nb_train]))

		with (save_dir / "val.txt").open("a") as f:
			f.write("".join(f"{Path('data/val') / n.name}\n" for n in new_names[nb_train:]))

def add_no_obj_images(yolo_dir, no_obj_dir, ratio, shuffle=True):
	'''
	Make sure that there is train.txt and val.txt in yolo folder passed
	as argument.
	'''
	train_dir = os.path.join(yolo_dir, 'train/')
	val_dir = os.path.join(yolo_dir, 'val/')
	train_file = os.path.join(yolo_dir, 'train.txt')
	val_file = os.path.join(yolo_dir, 'val.txt')

	assert os.path.isfile(train_file), 'train.txt does not exist in yolo directory'
	assert os.path.isfile(val_file), 'val.txt does not exist in yolo directory'

	images = sorted([item for item in os.listdir(no_obj_dir) if os.path.splitext(item)[1] == '.jpg'])

	if shuffle:
		rand_gen = random.Random(478_737_303)
		images = rand_gen.sample(images, len(images))

	annotations = [os.path.splitext(item)[0] + '.txt' for item in images]
	nb_train_samples = round(ratio*len(images))

	# Train folder
	for (image, annotation) in zip(images[:nb_train_samples], annotations[:nb_train_samples]):
		image_path = os.path.join(no_obj_dir, image)
		save_image_path = os.path.join(train_dir, image)
		save_annot_path = os.path.join(train_dir, annotation)
		if os.path.exists(save_image_path):
			print('Image {} already exists in yolo train folder'.format(image))
			continue
		# Else
		shutil.copy(image_path, save_image_path)
		with open(save_annot_path, 'w') as f:
			f.write('')
		with open(train_file, 'a') as f:
			f.write(os.path.join('data/train/', image) + '\n')

	# Val folder
	for (image, annotation) in zip(images[nb_train_samples:], annotations[nb_train_samples:]):
		image_path = os.path.join(no_obj_dir, image)
		save_image_path = os.path.join(val_dir, image)
		save_annot_path = os.path.join(val_dir, annotation)
		if os.path.exists(save_image_path):
			print('Image {} already exists in yolo val folder'.format(image))
			continue
		# Else
		shutil.copy(image_path, save_image_path)
		with open(save_annot_path, 'w') as f:
			f.write('')
		with open(val_file, 'a') as f:
			f.write(os.path.join('data/val/', image) + '\n')

	#Train
	with open(train_file, 'r') as f:
		content = f.readlines()
	random.shuffle(content)
	with open(train_file, 'w') as f:
		f.writelines(content)

	#Val
	with open(val_file, 'r') as f:
		content = f.readlines()
	random.shuffle(content)
	with open(val_file, 'w') as f:
		f.writelines(content)


def remove_to_close(folder, save_dir, class_id, margin=0.001):
	annotations = files_with_extension(folder, ".txt")

	for annotation in annotations:
		with open(annotation, 'r') as f:
			content = f.readlines()

		content = [c.strip() for c in content]

		with open(os.path.join(save_dir, os.path.split(annotation)[1]), 'w') as f:
			for line in content:
				label, x, y, w, h = line.split(' ')

				if int(label) == class_id:
					xmin = float(x) - float(w) / 2.0
					xmax = float(x) + float(w) / 2.0
					ymin = float(y) - float(h) / 2.0
					ymax = float(y) + float(h) / 2.0

					if (xmin > margin) and (ymin > margin) \
					and (xmax < 1 - margin) \
					and (ymax < 1 - margin):
						f.write('{} {} {} {} {}\n'.format(label, x, y, w, h))

				else:
					f.write('{} {} {} {} {}\n'.format(label, x, y, w, h))


def get_image_sizes(folders):
	sizes = []
	for folder in folders:
		for image_path in os.listdir(folder):
			if os.path.splitext(image_path)[1] == ".jpg" \
			and os.path.isfile(os.path.join(folder, os.path.splitext(image_path)[0] + '.xml')):
				full_path = os.path.join(folder, image_path)
				try:
					# Use PILLOW instead of Skimage
					image = io.imread(full_path)
					shape = image.shape
					sizes.append(shape)
				except:
					print('Error occured while reading: {}'.format(full_path))
	print(set(sizes))


def replace_label(old_label, new_label, folders):
	'''
	Be careful when using this function, it changes labels in xml_files.
	Consider backing up your annotations before.
	'''
	count = 0
	for folder in folders:
		xml_files = files_with_extension(folder, ".xml")

		for xml_file in xml_files:
			tree = ET.parse(xml_file).getroot()

			for obj in tree.findall('object'):
				label = obj.find('name').text
				if label == old_label:
					count += 1
					obj.find('name').text = new_label

			with open(xml_file, 'w') as f:
				tree_str = ET.tostring(tree, encoding='unicode')
				f.write(tree_str)


def draw_center_point(bounding_boxes, save_path="save/"):
	radius = 10
	color="red"

	if not os.path.isdir(save_path):
		os.mkdir(save_path)

	for name in bounding_boxes.getNames():
		image = os.path.splitext(name)[0] + ".jpg"
		img_save = os.path.join(save_path, os.path.basename(image))
		boxes = bounding_boxes.getBoundingBoxesByImageName(name)
		img_PIL = PIL.Image.open(image)
		img_draw = PIL.ImageDraw.Draw(img_PIL)

		for box in boxes:
			xmin, ymin, xmax, ymax = box.getAbsoluteBoundingBox(BBFormat.XYC)
			x1, y1, x2, y2 = x - radius, y - radius, x + radius, y + radius

			img_draw.ellipse(xy=(x1, y1, x2, y2), fill=color)
			img_draw.rectangle(xy=(xmin, ymin, xmax, ymax))

		img_PIL.save(img_save)


def voc_to_coco(bounding_boxes, save_path="", ratio=0.8, no_obj_path=None, copy_images=False):
	# Create dir if necessary
	if copy_images:
		train_dir = os.path.join(save_path, "images_train/")
		valid_dir = os.path.join(save_path, "images_val/")
		create_dir(train_dir)
		create_dir(valid_dir)

	labels = bounding_boxes.getClasses()
	image_paths = bounding_boxes.getNames()
	nb_train_samples = int(ratio * len(image_paths))

	# Categories
	categories = []
	category_to_id = {}

	for i, label in enumerate(labels):
		category_to_id[label] = i
		category = {
			"supercategory": "none",
			"id": i,
			"name": str(label)}
		categories.append(category)

	# Images and Annotations
	images_valid = []
	images_train = []
	annotations_valid = []
	annotations_train = []

	random_gen = random.Random(498_562_751) # Fix seed for reproductibility
	random_image_paths = random_gen.sample(image_paths, len(image_paths))

	unique_box_id = 0
	image_id = 0

	for image_path in random_image_paths:
		(width, height) = bounding_boxes.imageSize(image_path)
		image_name = os.path.basename(image_path)

		image = {
			"id": image_id,
			"file_name": image_name,
			"width": width,
			"height": height}

		if image_id < nb_train_samples:
			if copy_images:
				new_image_path = os.path.join(train_dir, image_name)
				shutil.copy(image_path, new_image_path)
			images_train.append(image)

		else:
			if copy_images:
				new_image_path = os.path.join(valid_dir, image_name)
				shutil.copy(image_path, new_image_path)
			images_valid.append(image)

		for box in bounding_boxes.getBoundingBoxesByImageName(image_path):
			(x, y, w, h) = box.getAbsoluteBoundingBox(format=BBFormat.XYWH)
			label = str(box.getClassId())
			category_id = category_to_id[label]

			annotation = {
				"id": unique_box_id,
				"image_id": image_id,
				"category_id": category_id,
				"iscrowd": 0,
				"ignore": 0,
				"area": w * h,
				"bbox": [int(x), int(y), int(w), int(h)],
				"segmentation": [[x, y, x, y + h, x + w, y + h, x + w, h]]}

			unique_box_id += 1

			if image_id < nb_train_samples:
				annotations_train.append(annotation)
			else:
				annotations_valid.append(annotation)

		image_id += 1

	# No-obj images
	if no_obj_path is not None:
		no_obj_images = files_with_extension(no_obj_path, ".jpg")
		no_obj_images = random_gen.sample(no_obj_images, len(no_obj_images))
		nb_no_obj_samples = int(ratio * len(no_obj_images))

		for i, no_obj_image in enumerate(no_obj_images):
			image_name = os.path.basename(no_obj_image)
			(width, height) = image_size(no_obj_image)

			image = {
				"id": image_id,
				"file_name": image_name,
				"width": width,
				"height": height}

			if i < nb_no_obj_samples:
				images_train.append(image)
				if copy_images:
					new_image_path = os.path.join(train_dir, image_name)
					shutil.copy(no_obj_image, new_image_path)

			else:
				images_valid.append(image)
				if copy_images:
					new_image_path = os.path.join(valid_dir, image_name)
					shutil.copy(no_obj_image, new_image_path)

			image_id += 1


	# Json File
	train_file = os.path.join(save_path, "train.json")
	valid_file = os.path.join(save_path, "val.json")

	data_train = {
		"images": images_train,
		"annotations": annotations_train,
		"categories": categories}
	data_valid = {
		"images": images_valid,
		"annotations": annotations_valid,
		"categories": categories}

	json.dump(data_train, open(train_file, "w"), indent=2)
	json.dump(data_valid, open(valid_file, "w"), indent=2)


def voc_to_coreML(bounding_boxes, save_dir="", ratio=0.8, no_obj_path=None):
	"""
	Converter from VOC annotation to CoreML annotation format for Apple
	platforms.
	"""
	# Directories
	train_dir = os.path.join(save_dir, "Training")
	valid_dir = os.path.join(save_dir, "Validation")

	create_dir(save_dir)
	create_dir(train_dir)
	create_dir(valid_dir)

	# JSON file names
	train_json_file = os.path.join(train_dir, "annotations.json")
	valid_json_file = os.path.join(valid_dir, "annotations.json")

	# JSON files
	train_data = []
	valid_data = []

	# Split and shuffle data
	image_names = bounding_boxes.getNames()
	nb_train = int(len(image_names) * ratio)

	random_generator = random.Random(498_562_751)
	random_image_names = random_generator.sample(image_names, len(image_names))

	# Loop through images
	for id, image_name in enumerate(random_image_names):
		# New unique name to avoid name conflits
		new_image_name = "image_{}.jpg".format(id)

		main_dir = train_dir if id < nb_train else valid_dir
		# Copy image in save folder
		shutil.copy(image_name, os.path.join(main_dir, new_image_name))

		image_boxes = []
		# Loop through image boxes
		for box in bounding_boxes.getBoundingBoxesByImageName(image_name):
			(x, y, w, h) = box.getAbsoluteBoundingBox(BBFormat.XYWH)
			label = str(box.getClassId())

			image_boxes.append({
				"label": label,
				"coordinates": {
					"x": x, "y": y, "width": w, "height": h
				}
			})

		# Fill-up JSON data
		image_annotation = {
			"imagefilename": new_image_name,
			"annotation": image_boxes
		}

		if id < nb_train:
			train_data.append(image_annotation)
		else:
			valid_data.append(image_annotation)

	# Same thing with images without objects
	if no_obj_path is not None:
		image_names = files_with_extension(no_obj_path, ".jpg")
		random_image_names = random_generator.sample(image_names, len(image_names))

		nb_train = int(ratio * len(random_image_names))

		for id, image_name in enumerate(random_image_names):
			new_image_name = "no-obj_image_{}.jpg".format(id)

			main_dir = train_dir if id < nb_train else valid_dir
			shutil.copy(image_name, os.path.join(main_dir, new_image_name))

			annotation = {
				"imagefilename": new_image_name,
				"annotation": []
			}

			if id < nb_train:
				train_data.append(annotation)
			else:
				valid_data.append(annotation)

	# Dump JSON files
	json.dump(train_data, open(train_json_file, "w"), indent=2)
	json.dump(valid_data, open(valid_json_file, "w"), indent=2)


def main(args=None):
	base_path = '/media/deepwater/DATA/Shared/Louis/datasets/'

	folders = [
		# Dataset 4.2
		'training_set/mais_haricot_feverole_pois/50/1',
		'training_set/mais_haricot_feverole_pois/50/2',
		'training_set/mais_haricot_feverole_pois/60/1',
		'training_set/mais_haricot_feverole_pois/60/2',
		'training_set/mais_haricot_feverole_pois/100/1',
		'training_set/mais_haricot_feverole_pois/100/2',
		'training_set/haricot_jeune',
		'training_set/carotte/2',
		'training_set/carotte/5',
		'training_set/mais/2',
		'training_set/mais/7',
		'training_set/mais/6',
		'validation_set',
		'training_set/2019-05-23_montoldre/mais/1',
		'training_set/2019-05-23_montoldre/mais/2',
		'training_set/2019-05-23_montoldre/mais/3',
		'training_set/2019-05-23_montoldre/mais/4',
		'training_set/2019-05-23_montoldre/haricot/1',
		'training_set/2019-05-23_montoldre/haricot/2',
		'training_set/2019-05-23_montoldre/haricot/3',
		'training_set/2019-05-23_montoldre/haricot/4',
		"training_set/2019-07-03_larrere/poireau/3",
		"training_set/2019-07-03_larrere/poireau/4",
		# Dataset 5.0
		"training_set/2019-09-25_montoldre/mais/1",
		"training_set/2019-09-25_montoldre/mais/2",
		"training_set/2019-09-25_montoldre/mais/3",
		"training_set/2019-09-25_montoldre/haricot",
		"training_set/2019-10-05_ctifl/mais_1",
		"training_set/2019-10-05_ctifl/mais_2",
		"training_set/2019-10-05_ctifl/haricot",
		# Dataset 6.0
		# "haricot_debug_montoldre_2",
		# "mais_debug_montoldre_2",
		# Database 6.1
		"training_set/2019-07-03_larrere/poireau/5",
		# Dataset 7.0
		"training_set/2020-10-01_ctifl/p0619_0928",
		"training_set/2020-10-01_ctifl/p0623_1241",
		"training_set/2020-10-01_ctifl/p0626_0816",
		"training_set/2020-10-01_ctifl/p0626_1420",
		"training_set/2020-10-01_ctifl/p0626_1423",
		"training_set/2020-10-01_ctifl/p0630_1420",  # Used
		"training_set/2020-10-01_ctifl/p0630_1427",
		"training_set/2020-10-01_ctifl/p0630_1428",
		"training_set/2020-10-01_ctifl/p0701_1308",  # ++
		"training_set/2020-10-01_ctifl/p0923_1627",  # ++  Used
		"training_set/2020-10-01_ctifl/p0928_1042",  #  TODO for StructureDet
		# Dataset 8.0
		"training_set/2020-10-12_montoldre/bean_1",
		"training_set/2020-10-12_montoldre/bean_2",
		"training_set/2020-10-12_montoldre/bean_3",
		"training_set/2020-10-12_montoldre/maize_1",
		"training_set/2020-10-12_montoldre/maize_2",
		"training_set/2020-10-12_montoldre/maize_3",
		# Dataset 8.1
		"training_set/2020-10-12_montoldre/bean_4",
		"training_set/2020-10-12_montoldre/maize_4",
		# Dataset 9.0
		"training_set/2021-03-29_larrere/row_1",
	]

	folders = [os.path.join(base_path, folder) for folder in folders]
	no_obj_dir = '/media/deepwater/DATA/Shared/Louis/datasets/training_set/no_obj/'

	classes = ["mais", "haricot", "poireau", 'mais_tige', 'haricot_tige', 'poireau_tige']

	names_to_labels = {
		'mais': 0, 'haricot': 1, 'poireau': 2, 'mais_tige': 3,
		'haricot_tige': 4, 'poireau_tige': 5}

	labels_to_names = {
		0: "maize", 1: "bean", 2: "leek", 3: "stem_maize",
		4: "stem_bean", 5: "stem_leek"}

	fr_to_en = {
		"mais": "maize",
		"haricot": "bean",
		"poireau": "leek",
		"mais_tige": "maize_stem",
		"haricot_tige": "bean_stem",
		"poireau_tige": "leek_stem"}

	yolo_path = "yolo/"

	clean_xml_files(folders)
	boxes = Parser.parse_xml_directories(folders, classes)
	boxes = my_library.normalized_stem_boxes(boxes, ratio=2.5/100)
	boxes.stats()

	# xml_to_yolo_3(boxes, yolo_path, names_to_labels,
	# 	shuffled=True, ratio=0.8)
	# add_no_obj_images(yolo_path, no_obj_dir,
	# 	ratio=0.8, shuffle=True)
	xml_to_yolo_4(boxes, names_to_labels, save_dir=yolo_path, no_obj_dir=no_obj_dir)

	# voc_to_coreML(boundingBoxes, save_dir="CoreML", no_obj_path=no_obj_dir)
	# my_library.pix2pix_square_stem_db(boundingBoxes, label_size=8/100)
	# voc_to_coco(boundingBoxes, no_obj_path=no_obj_dir)

	# boxes = Parser.parse_coco_gt("val.json")
	# boxes += Parser.parse_coco_gt("train.json")
	# boxes.stats()

	# boxes = Parser.parse_coco("groundTruth.json", "results.json")
	# boxes.stats()
	# Evaluator().printAPs(boxes)

	# xml_to_csv_2(boundingBoxes, no_obj_dir=no_obj_dir)

	# test.get_square_database(yolo_path, '/home/deepwater/yolo_tige_sqr/')
	# test.draw_bbox_images("/home/deepwater/yolo/val/", "/home/deepwater/yolo/result/")
	# test.crop_annotation_to_square()

	# draw_center_point(boundingBoxes)

if __name__ == '__main__':
	main()
