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


if __name__ == '__main__':
	my_folder = "../DATA/Gaofen/val/label_xml"

	for file in os.listdir(my_folder):
		file_path = f"{my_folder}/{file}"

		file_name = get_file_name(file)

		old_text = f"A:{chr(92)}Datasets{chr(92)}GAOFEN{chr(92)}val{chr(92)}label_xml{chr(92)}{str(int(file_name)+1000)}.xml"
		new_text = f"{file_name}.xml"
		replace_text(file_path,old_text,new_text)