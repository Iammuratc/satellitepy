import os
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
from models.models import * 

class Utilities:
    """docstring for Utilities"""
    def __init__(self, settings):
        # super(Utilities, self).__init__()
        self.settings = settings

            
    def get_file_paths(self,folder,sort=True):
        file_paths = [os.path.join(folder,file) for file in os.listdir(folder)]
        if sort:
            file_paths.sort()
        return file_paths 

    def get_model(self):
        model_name = self.settings['model']['name']
        if model_name == 'UNet':
            model = UNet(init_features=self.settings['model']['init_features'])
        elif model_name == 'Custom_0':
            model = Custom_0()
        else:
            print('Please define your model first.')
            return 0
        return model



class ImageViewer(object):
    def __init__(self, ax, instance_table,image_data):
        self.ax = ax
        self.image_data = image_data
        self.instance_table = list(instance_table.keys())


        self.slices = len(image_data)
        self.ind = 0

        self.im = ax.imshow(self.get_next_img())
        self.update()

    # def onscroll(self, event):
    def on_press(self, event):
        # print("%s %s" % (event.button, event.step))
        # if event.button == 'up':
        if event.key == 'd':
            self.ind = (self.ind + 1) % self.slices
        # else:
        if event.key == 'a':
            self.ind = (self.ind - 1) % self.slices
        print(self.ind)
        self.update()

    def get_next_img(self):
        img = cv2.cvtColor(cv2.imread(self.image_data[self.ind][0]),cv2.COLOR_BGR2RGB)
        return img

    def update(self):

        label = self.instance_table[self.image_data[self.ind][1]]
        predicted = self.instance_table[self.image_data[self.ind][2]]

        self.im.set_data(self.get_next_img())
        # self.ax.set_ylabel('slice %s' % self.ind)
        self.ax.set_title(f"Label: {label}, Predicted: {predicted}")
        self.im.axes.figure.canvas.draw()
