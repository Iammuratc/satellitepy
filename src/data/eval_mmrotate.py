import json
import os
import pickle as pkl
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from .cutout import geometry
# from mmdet.apis.inference import init_detector, inference_detector, show_result_pyplot
# import mmcv


class EvalMMRotate:
	"""docstring for MMRotate"""
	def __init__(self, settings):
		# super(MMRotate, self).__init__()
		self.settings = settings
		self.logger = self.settings['logger']
		self.df=self.get_test_df()

	def get_test_df(self,save=True):
		csv_path = self.settings['test_csv_path']
		if os.path.exists(csv_path):
			df = pd.read_csv(csv_path)
		else:
			test_pkl_path = self.settings['test_pkl_path']
			with open(test_pkl_path, "rb") as f:
			    pkl_object = pkl.load(f)
			df = pd.DataFrame(pkl_object)
			if save:
				df.to_csv(csv_path)
				self.logger.info(f"csv is saved at {csv_path}")
		return df

	def show_image(self,ind,show_bbox=True):
		## Read image path from settings (old) 
		# test_images_path = self.settings['test_images_path']
		# image_names = os.listdir(test_images_path)
		# image_names.sort()
		# img_path = os.path.join(self.settings['test_images_path'],image_names[ind])
		## Read image path from csv file
		img_path = self.df.iloc[ind,1]
		print(img_path)
		img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
		fig,ax=plt.subplots(1)
		ax.imshow(img)

		if show_bbox:
			bboxes = self.df.iloc[ind,2:]
			for bbox in bboxes:
				if bbox == []:
					continue
				geometry.Bbox.plot_bbox(bbox,ax,c='b')
		plt.show()


	# def inference_model(self,ind):
	# 	config_path = self.settings['config_path'] # 'configs/faster_rcnn_r50_fpn_1x.py'
	# 	checkpoint_path = self.settings['checkpoint_path'] # 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
	# 	### IMAGE
	# 	test_images_path = self.settings['test_images_path']
	# 	image_names = os.listdir(test_images_path)
	# 	# image_names.sort()

	# 	img_path = os.path.join(self.settings['test_images_path'],image_names[ind])
	# 	# img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
	# 	# img = cv2.imread(img_path)
	# 	img = mmcv.imread(img_path)
	# 	# build the model from a config file and a checkpoint file
	# 	model = init_detector(config_path, checkpoint_path, device='cuda:0')

	# 	# test a single image and show the results
	# 	# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
	# 	result = inference_detector(model, img)
	# 	print(result)
	# 	# visualize the results in a new window
	# 	show_result_pyplot(img, result, model.CLASSES)
	# 	# or save the visualization results to image files
	# 	# show_result(img, result, model.CLASSES, out_file='result.jpg')