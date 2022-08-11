from torchvision.transforms import Compose
import torch
import torch.optim as optim
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


from transforms import ToTensor, Normalize
from recognition import Recognition
from dataset import RecognitionDataset
from utilities import EarlyStopping, ImageViewer 
from models import Custom_0


### TODO: Log files

class Classifier:
    def __init__(self,settings):
        self.settings=settings
        self.patch_size=settings['patch']['size']
        self.project_folder = settings['project_folder']

    def train(self,patience,load_last_state=False):
        ### TRAINING HYPERPARAMETERS
        epochs = self.settings['training']['epochs']
        batch_size = self.settings['training']['batch_size']

        ### READ MODEL AND MOVE IT TO GPU
        model_path = self.settings['model']['path']
        model = self.get_model()
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
        

        ### COST AND OPTIMIZER FUNCS
        criterion = nn.CrossEntropyLoss(weight=self.get_class_weights().to(device))
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        instances = '    '.join(list(self.settings['dataset']['instance_table'].keys()))
        class_weight = '        '.join([str(weight) for weight in self.settings['training']['class_weight']])
        print('The following weights will be applied to the lost function:\n')
        print(instances)
        print(class_weight,'\n')
        ### DATA
        loader_train, _, loader_val = self.get_loaders(batch_size=self.settings['training']['batch_size'])

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
                loss = criterion(outputs, data['label'].to(device))

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
                loss = criterion(outputs, data['label'].to(device))
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


                # print statistics
                # running_loss += loss.item()
                # if i % stat_step == stat_step-1:    # print every 2000 mini-batches
                    # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / stat_step:.3f}')
                    # tepoch.set_postfix(loss=running_loss/stat_step)
                    # running_loss = 0.0
            early_stopping(val_loss,model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # torch.save(model.cpu().state_dict(), self.path)
        # writer.flush()
        # writer.close()
        print('Finished Training')


    def get_loaders(self,batch_size):
        if self.settings['training']['split_ratio']:
            dataset_train = self.get_dataset('train')
            dataset_test = self.get_dataset('test')
            dataset_val = self.get_dataset('val')
            ### MERGE DATASETS
            dataset_full = torch.utils.data.ConcatDataset([dataset_train, dataset_test,dataset_val])
            len_full_dataset = len(dataset_full)

            ### SPLIT RATIO
            ratio_train, ratio_test, ratio_val = self.settings['training']['split_ratio']
            train_size = int(ratio_train * len_full_dataset)
            test_size = int(ratio_test * len_full_dataset)
            val_size = len_full_dataset - train_size - test_size
            # dataset_train, dataset_test, dataset_val = torch.utils.data.random_split(dataset_full, [train_size, test_size,val_size])
            dataset_train = torch.utils.data.Subset(dataset_full, range(train_size))
            dataset_test = torch.utils.data.Subset(dataset_full, range(train_size, train_size + test_size))
            dataset_val = torch.utils.data.Subset(dataset_full, range(train_size + test_size,train_size + test_size + val_size))
            print(f'Full dataset (train+test+val) is split into:\n{len(dataset_train)},{len(dataset_test)},{len(dataset_val)}\n')
        else:
            dataset_train = self.get_dataset('train')
            dataset_test = self.get_dataset('test')
            dataset_val = self.get_dataset('val')

        loader_train = self.get_loader(dataset=dataset_train,batch_size=batch_size,shuffle=True)
        loader_test = self.get_loader(dataset=dataset_test,batch_size=batch_size,shuffle=False)
        loader_val = self.get_loader(dataset=dataset_val,batch_size=batch_size,shuffle=False)
        return loader_train, loader_test, loader_val


    def get_class_weights(self):
        class_weight = self.settings['training']['class_weight']
        return torch.FloatTensor(class_weight)        

    def get_dataset(self,dataset_part):

        # recognition_instance = Recognition(self.settings,dataset_part)
        dataset = RecognitionDataset(self.settings,
                                    dataset_part=dataset_part,
                                    transform=self.get_transform(dataset_part))
        return dataset

    def get_loader(self,dataset,shuffle,batch_size,num_workers=4):

        loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size,
                                            shuffle=shuffle, 
                                            num_workers=num_workers)      

        return loader

    def get_transform(self,dataset_part):
        # if dataset_part=='train':
        transform = Compose([ToTensor(),Normalize()])
        # else:
        #     transform = Compose([Normalize()])
        return transform

    def get_model(self):
        model_name = self.settings['model']['name']
        if model_name == 'resnet18':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) # resnet34  resnet50   resnet101   resnet152
        elif model_name == 'custom_0':
            model = Custom_0()
        else:
            print('Please define your model first.')
            return 0
        return model



    def get_predictions(self,dataset_part): #show_false_images=False,save=False,plot=True
        ### MODEL
        model = self.get_model()
        model.load_state_dict(torch.load(self.settings['model']['path'],map_location='cpu'))

        ### HOT ENCODING SETTINGS
        logSoftmax = nn.LogSoftmax(dim=1)    

        ### GET MY LOADER
        batch_size=1

        if self.settings['training']['split_ratio']:
            loader_train, loader_test, loader_val = self.get_loaders(batch_size)
            loaders = {'train':loader_train,
                        'test':loader_test,
                        'val':loader_val}
            my_loader = loaders[dataset_part]
        else:
            dataset = self.get_dataset(dataset_part)
            my_loader = self.get_loader(dataset=dataset,batch_size=batch_size)


        for i, data in enumerate(my_loader):
            y_pred_batch = model(data['image'])
            if self.settings['training']['hot_encoding']:
                # print(y_pred_batch)
                y_pred_batch = logSoftmax(y_pred_batch)
                # print(y_pred_batch)
                y_pred_batch = torch.argmax(y_pred_batch,dim=1).data.cpu()
                y_true_batch = torch.argmax(data['label'],dim=1).data.cpu()
            else:
                y_pred_batch = (torch.max(torch.exp(y_pred_batch), 1)[1]).data.cpu()#.numpy()

                y_true_batch = data['label']

            image_path = data['image_path'][0]
            # print('\n'.join(image_paths))
            # for t, p, img_path in zip(y_pred_batch, y_true_batch, image_paths):
                # confusion_matrix[t.long(), p.long()] += 1
                # if t!=p:
                    # false_image_data.append([img_path,t,p])
            yield y_pred_batch[0].long(), y_true_batch[0].long(), image_path
            # if i == 1:
            #     break

    def plot_conf_mat(self,dataset_part,plot,save):

        ### INSTANCE DICT
        instance_list = list(self.settings['dataset']['instance_table'].keys())

        ### PREDICTIONS
        prediction_generator = self.get_predictions(dataset_part)        

        ## CONFUSION MATRIX
        confusion_matrix = torch.zeros(len(instance_list), len(instance_list))
        while True:
            try:
                model_pred, model_true, image_path = next(prediction_generator)
                confusion_matrix[model_true, model_pred] += 1

            except StopIteration:
                break

        exp_no = self.settings['experiment']['no'] 

        ### PLOT
        df_conf_mat = pd.DataFrame(confusion_matrix,index=instance_list,columns=instance_list)
        fig, ax = plt.subplots(1)
        # plt.figure(figsize = (10,7))
        ax.set_title(f'Exp no: {exp_no} --- Dataset: {dataset_part}\nRows: Predicted --- Cols: Ground truth')
        # plt.ylabel('Predicted',fontsize=18,loc='top')
        # plt.xlabel('Ground truth',fontsize=18,loc='center')
        sn.heatmap(df_conf_mat, annot=True, fmt='g',ax=ax)
        if save:
            fig_path = os.path.join(self.settings['experiment']['folder'],f'conf_mat_{dataset_part}.png')
            fig.savefig(fig_path)
        if plot:
            plt.show()

    def plot_images(self,instances,dataset_part,save,plot=True):
        ### INSTANCES
        y_pred_name, y_true_name = instances
        y_pred = self.settings['dataset']['instance_table'][y_pred_name]
        y_true = self.settings['dataset']['instance_table'][y_true_name]

        ### PREDICTIONS
        prediction_generator = self.get_predictions(dataset_part)        

        ### IMAGE PATHS
        image_paths = []
        while True:
            try:
                model_pred, model_true, image_path = next(prediction_generator)
                if (model_true == y_true) and (model_pred == y_pred):
                    image_paths.append(image_path)
            except StopIteration:
                break
        ## IF SAVE PLOTS MAKE FOLDER
        if save:
            fig_folder = os.path.join(self.settings['experiment']['folder'],f'{y_true_name}_{y_pred_name}_{dataset_part}')
            os.makedirs(fig_folder,exist_ok=True)
            print(f'Figures will be stored at:\n{fig_folder}\n')
        # print(image_paths)
        ### FIGURE SETTINGS
        row, col = 6, 7
        fig_subplots = row*col
        quotient, remainder = divmod(len(image_paths), fig_subplots)
        fig_count = quotient + 1 if remainder !=0 else quotient

        for i_fig in range(fig_count):
            fig, ax = plt.subplots(row,col)
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            fig.suptitle(f'G.Truth: {y_true_name} Prediction: {y_pred_name}',fontsize=15)
            for ind,img_path in enumerate(image_paths[i_fig*fig_subplots:(i_fig+1)*fig_subplots]):
                subplot_ind = np.unravel_index(ind,(row,col))

                img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
                ax[subplot_ind[0],subplot_ind[1]].imshow(img)
                ax[subplot_ind[0],subplot_ind[1]].set_xticks([])
                ax[subplot_ind[0],subplot_ind[1]].set_yticks([])
            if save:
                fig.savefig(os.path.join(fig_folder,f'fig_{i_fig}.png'))
        if plot:
            plt.show()


        # fig, ax = plt.subplots(1)
        # image_viewer = ImageViewer( ax=ax,
        #                             instance_table=self.settings['dataset']['instance_table'],
        #                             image_data=false_image_data)

        # # fig.canvas.mpl_connect('scroll_event', image_viewer.onscroll)
        # fig.canvas.mpl_connect('key_press_event', image_viewer.on_press)

        # false_image_paths = [img_data[0] for img_data in false_image_data]
        # # print('\n'.join(false_image_paths))
