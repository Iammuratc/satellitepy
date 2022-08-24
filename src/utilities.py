# import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



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
