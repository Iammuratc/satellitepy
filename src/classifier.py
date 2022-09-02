from torchvision.transforms import Compose
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import os
# from tqdm import tqdm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2


from utilities import EarlyStopping


### TODO: Log files
class Classifier(object):
    """docstring for Classifier"""
    def __init__(self, settings):
        # super(Classifier, self).__init__()
        self.settings = settings
        
    def train(self,model,loss_func,optimizer,loaders,patience=10):#,load_last_state=False):
        ### DATA
        loader_train = loaders['train']
        loader_val = loaders['val']

        ### TRAINING HYPERPARAMETERS
        epochs = self.settings['training']['epochs']
        batch_size = self.settings['training']['batch_size']

        ### READ MODEL AND MOVE IT TO GPU
        model_path = self.settings['model']['path']
        # model = self.get_model()
        if os.path.exists(model_path):
            print(f'Model is read from the previous version at:\n{model_path}')
            if input('Do you confirm that? [y/n] ') != 'y':
                print('The existing model will be overwritten!\n')
                time.sleep(2)
                # return 0
            else:
                model.load_state_dict(torch.load(model_path))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        ### LOG
        # log_folder = self.settings['training']['log_folder']
        # writer = SummaryWriter(log_dir=log_folder)
        # stat_step = 20 # write log at every stat_step*batch_size image

        ### EARLY STOPPING
        early_stopping = EarlyStopping(patience=patience, verbose=True,path=model_path)

        for epoch in range(epochs):  # loop over the dataset multiple times

            ### LOSS
            train_losses = []
            val_losses = []

            ### TRAIN MODEL
            model.train()
            for i,data in enumerate(loader_train):
                # data is a dict with keys "image", "label"
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                
                outputs = model(data['image'].to(device))
                loss = loss_func(outputs, data['label'].to(device))

                # writer.add_scalar("Loss/train", loss, epoch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                
            ### VALIDATE MODEL
            model.eval()
            for data in loader_val:
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(data['image'].to(device))

                # calculate the loss
                loss = loss_func(outputs, data['label'].to(device))
                # writer.add_scalar("Loss/val", loss, epoch)
                # record validation loss
                val_losses.append(loss.item())

            train_loss = np.average(train_losses)
            val_loss = np.average(val_losses)

            # train_losses_avg.append(train_loss)
            # val_losses_avg.append(val_loss)

            msg = (f'[{epoch}/{epochs}] ' +
             f'train_loss: {train_loss:.5f} ' +
             f'valid_loss: {val_loss:.5f}')
            print(msg)


            early_stopping(val_loss,model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # torch.save(model.cpu().state_dict(), self.path)
        # writer.flush()
        # writer.close()
        print('Finished Training')

        # fig, ax = plt.subplots(1)
        # image_viewer = ImageViewer( ax=ax,
        #                             instance_table=self.settings['dataset']['instance_table'],
        #                             image_data=false_image_data)

        # # fig.canvas.mpl_connect('scroll_event', image_viewer.onscroll)
        # fig.canvas.mpl_connect('key_press_event', image_viewer.on_press)

        # false_image_paths = [img_data[0] for img_data in false_image_data]
        # # print('\n'.join(false_image_paths))
