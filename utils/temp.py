import os

### REPLACE 

# Read in the file

def replace_text(file_path,old_text,new_text):
	with open(file_path, 'r') as file :
	  filedata = file.read()

	# Replace the target string
	filedata = filedata.replace(old_text,new_text)

	# Write the file out again
	with open(file_path, 'w') as file:
	  file.write(filedata)

def get_file_name(file):
	file_name = file.split('.')[0]
	return file_name


def unfold_mat():
	unfold = torch.nn.Unfold(kernel_size=(4, 4),stride=4)
	input = torch.randn(1, 3, 20, 20)
	output = unfold(input)
	# each patch contains 30 values (2x3=6 vectors, each of 5 channels)
	# 4 blocks (2x3 kernels) in total in the 3x4 input
	print(output.size())


if __name__ == '__main__':
	# my_folder = "../DATA/Gaofen/val/label_xml"

	# for file in os.listdir(my_folder):
	# 	file_path = f"{my_folder}/{file}"

	# 	file_name = get_file_name(file)

	# 	old_text = f"A:{chr(92)}Datasets{chr(92)}GAOFEN{chr(92)}val{chr(92)}label_xml{chr(92)}{str(int(file_name)+1000)}.xml"
	# 	new_text = f"{file_name}.xml"
	# 	replace_text(file_path,old_text,new_text)
	import torch
	unfold_mat()