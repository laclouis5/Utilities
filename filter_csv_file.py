import os

file_in  = '/Data/Louis/RetinaNet/datasets/training_set/train_data.csv'
file_out = '/Data/Louis/RetinaNet/datasets/training_set/train_data_bipbip.csv'

class_to_keep = 'haricot'

with open(file_in) as f:
	content = f.readlines()

content = [x.strip() for x in content]
content = [line for line in content if line.endswith(class_to_keep) and ('operose' not in line)]

with open(file_out, 'w') as f:
	for item in content:
		f.write(item + '\n')
