# from transforms import ToTensor, Normalize
# from dataset import RecognitionDataset
from .classifier import Classifier
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class ClassifierRecognition(Classifier):
    def __init__(self,utils,dataset):
        super(ClassifierRecognition, self).__init__(utils.settings)
        self.utils=utils
        self.dataset=dataset
        self.settings=utils.settings
        self.patch_size=self.settings['patch']['size']

    def train(self):
        ### COST AND OPTIMIZER FUNCS
        loss_func = nn.CrossEntropyLoss(weight=self.get_class_weights().to(device))
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        instances = '    '.join(list(self.settings['dataset']['instance_table'].keys()))
        class_weight = '        '.join([str(weight) for weight in self.settings['training']['class_weight']])
        print('The following weights will be applied to the lost function:\n')
        print(instances)
        print(class_weight,'\n')
        ### DATA
        loaders = self.get_loaders(batch_size=self.settings['training']['batch_size'])

        super().train(loss_func,optimizer,loaders)

    def get_loaders(self,batch_size):
        if self.settings['training']['split_ratio']:
            dataset_train = self.dataset['train']
            dataset_test = self.dataset['test']
            dataset_val = self.dataset['val']
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
            dataset_train = self.dataset['train']
            dataset_test = self.dataset['test']
            dataset_val = self.dataset['val']

        loader_train = self.get_loader(dataset=dataset_train,batch_size=batch_size,shuffle=True)
        loader_test = self.get_loader(dataset=dataset_test,batch_size=batch_size,shuffle=False)
        loader_val = self.get_loader(dataset=dataset_val,batch_size=batch_size,shuffle=False)
        return loader_train, loader_test, loader_val


    def get_class_weights(self):
        class_weight = self.settings['training']['class_weight']
        return torch.FloatTensor(class_weight)        

    def get_loader(self,dataset,shuffle,batch_size,num_workers=4):

        loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size,
                                            shuffle=shuffle, 
                                            num_workers=num_workers)      

        return loader

    def get_predictions(self,dataset_part):
        ### MODEL
        model = self.utils.get_model()
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
            dataset = self.dataset[dataset_part]
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
            fig.suptitle(f'Ground Truth: {y_true_name} Prediction: {y_pred_name}',fontsize=15)
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


