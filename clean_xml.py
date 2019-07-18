'''
This library provides tools for converting VOC style XML to different formats (RetinaNet, YOLO).
'''

import os
import shutil
import xml.etree.ElementTree as ET
from random import shuffle
from parse_xml import parse_xml
from utils import *
from BoundingBoxes import BoundingBoxes
from BoundingBox import BoundingBox
from glob import glob, iglob
from Evaluator import *
from skimage import io

import test

def clean_xml_files(folders):
	'''
	This function updates the path field of XML files located in 'folders'
	to reflect the current path of those files. All functions in this package use path written
	in the 'path' field of XML files so they need to be right.

	You may use this function if XML files are moved to a different location and before
	using another function of this library.

	FOLDERS - List of absolute paths to folders containing train or test images and XML files.

	# Admin priviledge may be mandatory.
	'''
	print('Changing path name...')

	for folder in folders:
		for file in os.listdir(folder):
			# Check if XML file
			if(os.path.splitext(file)[1] != '.xml'): continue

			# Retreive the XML tree
			tree = ET.parse(os.path.join(folder, file)).getroot()

			# Modification of path field
			path_field      = tree.find('path')
			path_field.text = os.path.join(folder, file)

			# Update the file
			with open(os.path.join(folder, file), 'w') as xml_file:
				tree_str = ET.tostring(tree, encoding='unicode')
				xml_file.write(tree_str)

	print('Done!')


def xml_to_csv(folders, csv_path, csv_valid, names_to_labels, ratio=0.8):
	'''
	This function transcript VOC XML files to a CSV file for training (RetinaNet for instance).
	!!! Use clean_xml_files func before !!!

	FOLDERS - List of absolute paths to folders containing train or test images and XML files.

	CSV_PATH - Absolute path including file name and extention to the CSV file to be generated.
	For instance '/home/upser_name/train.csv' will create file 'train.csv' in '/home/user_name/'.

	NAMES_TO_LABELS - Dictionary with object name as key and label number as value. This makes
	explicit the relation object-label contrary to a list.
	'''
	print('Transcripting XML to CSV...')

	csv_temp = []

	for folder in folders:
		for file in sorted(os.listdir(folder)):
			# Check if XML file
			if(os.path.splitext(file)[1] == '.xml'):
				csv_temp.append(os.path.join(folder, file))

	shuffle(csv_temp)
	nb_train_samples = int(len(csv_temp) * ratio)
	train_xml = csv_temp[:nb_train_samples]
	val_xml   = csv_temp[nb_train_samples:]

	train_data = []
	val_data   = []

	for file in train_xml+val_xml:
		# Retreive and process each 'object' in the etree
		root = ET.parse(os.path.join(folder, file)).getroot()

		for obj in root.findall('object'):
			name = obj.find('name').text

			# Save only train classes
			if name not in names_to_labels.keys(): continue
			coords = []

			# Retreive bounding box coordinates
			bounding_box = obj.find('bndbox')
			[coords.append(int(coord.text)) for coord in bounding_box.getchildren()]

			line = '{},{},{},{},{},{}\n'.format(
				root.find('path').text,
				coords[0],
				coords[1],
				coords[2],
				coords[3],
				name)

			if file in train_xml:
				train_data.append(line)
			elif file in val_xml:
				val_data.append(line)

	with open(os.path.join(csv_path), 'w') as f_train:
		f_train.writelines(train_data)

	with open(os.path.join(csv_valid), 'w') as f_valid:
		f_valid.writelines(val_data)

	print('Done!')


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

	image_folder       = os.path.join(database_path, "JPEGImages")
	annotations_folder = os.path.join(database_path, "Annotations")
	imageset_folder    = os.path.join(database_path, "ImageSets/Main")

	trainval_file = open(os.path.join(imageset_folder, 'trainval.txt'), 'w')
	test_file     = open(os.path.join(imageset_folder, 'test.txt'), 'w')

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
			basepath       = os.path.split(tree.find('path').text)[0]
			xml_path       = os.path.join(basepath, file)
			image_path     = os.path.join(basepath, image_filename)

			# Copy XML files to the new directory
			shutil.copy(image_path, image_folder)
			shutil.copy(xml_path, annotations_folder)

			# Rename with a unique name
			new_image_name = '{}.jpg'.format(image_counter)
			new_xml_name   = '{}.xml'.format(image_counter)

			os.rename(os.path.join(image_folder, image_filename), os.path.join(image_folder, new_image_name))
			os.rename(os.path.join(annotations_folder, file), os.path.join(annotations_folder, new_xml_name))

			# Change 'path' and 'filename' fields to reflect the name changes.
			new_xml_file_path = os.path.join(annotations_folder, new_xml_name)
			new_tree          = ET.parse(new_xml_file_path).getroot()

			new_tree.find('filename').text = new_image_name
			new_tree.find('path'). text    = new_xml_file_path

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
	test_file.close()

	print('Done!')


def xml_to_yolo_3(boundingBoxes, yolo_dir, names_to_labels, ratio=0.8):
	train_dir  = os.path.join(yolo_dir, 'train')
	val_dir    = os.path.join(yolo_dir, 'val')
	train_file = os.path.join(yolo_dir, 'train.txt')
	val_file   = os.path.join(yolo_dir, 'val.txt')

	if not os.path.isdir(yolo_dir):
		os.mkdir(yolo_dir)
	if not os.path.isdir(train_dir):
		os.mkdir(train_dir)
	if not os.path.isdir(val_dir):
		os.mkdir(val_dir)

	unique_ID = -1
	names     = boundingBoxes.getNames()
	shuffle(names)

	for i, name in enumerate(names):
		unique_ID += 1
		yolo_rep   = []
		img_path   = os.path.splitext(name)[0] + '.jpg'
		if i < len(names) * ratio:
			save_dir = train_dir
		else:
			save_dir = val_dir

		for box in boundingBoxes.getBoundingBoxesByImageName(name):
			idenfier   = 'im_{}'.format(unique_ID)
			label      = names_to_labels[box.getClassId()]
			x, y, w, h = box.getRelativeBoundingBox()

			yolo_rep.append('{} {} {} {} {}\n'.format(label, x, y, w, h))

		with open(os.path.join(save_dir, idenfier + '.txt'), 'w') as f_write:
			f_write.writelines(yolo_rep)

		with open(train_file, 'w') as f_write_1, open(val_file, 'w') as f_write_2:
			for item in iglob(os.path.join(train_dir, '*.jpg')):
				tail = os.path.split(item)[1]
				f_write_1.write('{}\n'.format(os.path.join('data/train/', tail)))

			for item in iglob(os.path.join(val_dir, '*.jpg')):
				tail = os.path.split(item)[1]
				f_write_2.write('{}\n'.format(os.path.join('data/val/', tail)))

		shutil.copy(img_path, os.path.join(save_dir, idenfier + '.jpg'))


def add_no_obj_images(yolo_dir, no_obj_dir, ratio=0.8):
	'''
	Make sure that there is train.txt and val.txt in yolo foler passed
	as argument.
	'''
	train_dir  = os.path.join(yolo_dir, 'train/')
	val_dir    = os.path.join(yolo_dir, 'val/')
	train_file = os.path.join(yolo_dir, 'train.txt')
	val_file   = os.path.join(yolo_dir, 'val.txt')

	assert os.path.isfile(train_file), 'train.txt does not exist in yolo directory'
	assert os.path.isfile(val_file), 'val.txt does not exist in yolo directory'

	images      = [item for item in os.listdir(no_obj_dir) if os.path.splitext(item)[1] == '.jpg']
	annotations = [os.path.splitext(item)[0] + '.txt' for item in images]
	nb_train_samples = int(0.8*float(len(images)))

	# Train folder
	for (image, annotation) in zip(images[:nb_train_samples], annotations[:nb_train_samples]):
		image_path      = os.path.join(no_obj_dir, image)
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
		image_path      = os.path.join(no_obj_dir, image)
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
	shuffle(content)
	with open(train_file, 'w') as f:
		f.writelines(content)

	#Val
	with open(val_file, 'r') as f:
		content = f.readlines()
	shuffle(content)
	with open(val_file, 'w') as f:
		f.writelines(content)


def remove_to_close(folder, save_dir, class_id, margin=0.001):
	annotations = [os.path.join(folder, item) for item in os.listdir(folder) if os.path.splitext(item)[1] == '.txt']

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

					if (xmin > margin) and (ymin > margin) and (xmax < 1 - margin) and (ymax < 1 - margin):
						f.write('{} {} {} {} {}\n'.format(label, x, y, w, h))

				else:
					f.write('{} {} {} {} {}\n'.format(label, x, y, w, h))


def change_path(folder, new_name, new_file):
	'''
	I dont remember why I wrote this function
	'''
	with open(folder, 'r') as f:
		content = f.readlines()

	content = [c.strip() for c in content]
	images = [os.path.join(new_name, os.path.split(item)[1]) + '\n' for item in content]

	with open(new_file, 'w') as f:
		f.writelines(images)


def get_image_sizes(folders):
	sizes = []
	for folder in folders:
		for image_path in os.listdir(folder):
			if os.path.splitext(image_path)[1] == '.jpg' and os.path.isfile(os.path.join(folder, os.path.splitext(image_path)[0] + '.xml')):
				full_path = os.path.join(folder, image_path)
				try:
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
		xml_files = [os.path.join(folder, item) for item in os.listdir(folder) if os.path.splitext(item)[1] == '.xml']

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

def main(args=None):
	base_path = '/media/deepwater/DATA/Shared/Louis/RetinaNet/datasets/'

	folders = [
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
		"training_set/2019-07-03_larrere/poireau/4"
		# 'training_set/mais/7-2',
		# 'training_set/database_operose',
		# 'validation_set_challenge'
		]

	folders    = [os.path.join(base_path, folder) for folder in folders]
	no_obj_dir = '/media/deepwater/DATA/Shared/Louis/RetinaNet/datasets/training_set/no_obj/'


	classes         = ['mais','haricot', 'corotte', 'poireau', 'mais_tige', 'mais_tige_old', 'haricot_tige', 'haricot_tige_old', 'poireau_tige']
	names_to_labels = {'poireau': 0, 'poireau_tige': 1}

	# classes         = ['poireau', 'poireau_tige']
	# names_to_labels = {'poireau': 0, 'poireau_tige': 1}

	yolo_path = '/home/deepwater/yolo/'

	clean_xml_files(folders)
	boundingBoxes = parse_xml(folders, classes)
	boundingBoxes.stats()

	# xml_to_yolo_3(boundingBoxes, yolo_path, names_to_labels)
	# add_no_obj_images(yolo_path, no_obj_dir)

	# test.get_square_database(yolo_path, '/home/deepwater/yolo_tige_sqr/')
	# test.draw_bbox_images("/home/deepwater/yolo/val/", "/home/deepwater/yolo/result/")
	# test.crop_annotation_to_square()

if __name__ == '__main__':
	main()
