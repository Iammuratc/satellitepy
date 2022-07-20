from torchvision.transforms import Compose
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
# from tqdm import tqdm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from transforms import ToTensor, Normalize
from recognition import Recognition
from dataset import RecognitionDataset
from utilities import EarlyStopping
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
            model.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        

        ### COST AND OPTIMIZER FUNCS
        criterion, optimizer = self.get_funcs_for_train(model)

        ### DATA
        dataset_train = self.get_dataset('train')
        loader_train = self.get_loader(dataset=dataset_train,batch_size=batch_size)
        dataset_val = self.get_dataset('val')
        loader_val = self.get_loader(dataset=dataset_val,batch_size=batch_size)

        ### LOG
        # log_folder = self.settings['training']['log_folder']
        # writer = SummaryWriter(log_dir=log_folder)
        # stat_step = 20 # write log at every stat_step*batch_size image

        ### LOSS
        # train_losses_avg = []
        # val_losses_avg = []

        ### EARLY STOPPING
        early_stopping = EarlyStopping(patience=patience, verbose=True,path=model_path)

        for epoch in range(epochs):  # loop over the dataset multiple times

            ### LOSS
            train_losses = []
            val_losses = []
            # running_loss = 0.0
            # with tqdm(loader_train, unit="batch") as tepoch:

            ### TRAIN MODEL
            model.train()
            for i,data in enumerate(loader_train):
                # tepoch.set_description(f"Epoch {epoch}")
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

    def get_dataset(self,dataset_part):

        # recognition_instance = Recognition(self.settings,dataset_part)
        dataset = RecognitionDataset(settings,dataset_part=dataset_part,transform=self.get_transform(dataset_part))
        return dataset

    def get_loader(self,dataset,batch_size):

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=2)      

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


    def get_funcs_for_train(self,model):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        return criterion, optimizer

    def get_conf_mat(self,dataset_part,save=False,plot=True):
        exp_no = self.settings['exp_no']
        ### MODEL
        model = self.get_model()
        model.load_state_dict(torch.load(self.path,map_location='cpu'))

        ### DATA
        dataset = self.get_dataset(dataset_part)
        loader_test = self.get_loader(dataset=dataset,batch_size=20)

        ### INSTANCE DICT
        # instance_table_rev = dict((v, k) for k, v in dataset.instance_table.items())
        instance_list = list(dataset.instance_table.keys())
        ### APPEND BATCHES TO A LIST
        confusion_matrix = torch.zeros(len(instance_list), len(instance_list))
        for i, data in enumerate(loader_test):
            y_pred_batch = model(data['image'])
            y_pred_batch = (torch.max(torch.exp(y_pred_batch), 1)[1]).data.cpu()#.numpy()

            y_true_batch = data['label']
            for t, p in zip(y_pred_batch, y_true_batch):
                confusion_matrix[t.long(), p.long()] += 1

            # if i == 2:
            #     break

        # print(confusion_matrix)

        df_conf_mat = pd.DataFrame(confusion_matrix,index=instance_list,columns=instance_list)
        fig, ax = plt.subplots(1)
        # plt.figure(figsize = (10,7))
        ax.set_title(f'Dataset: {dataset_part} --- Rows: Predicted --- Cols: Ground truth')
        # plt.ylabel('Predicted',fontsize=18,loc='top')
        # plt.xlabel('Ground truth',fontsize=18,loc='center')
        sn.heatmap(df_conf_mat, annot=True, fmt='g',ax=ax)
        if save:
            fig_folder = f'{dataset.recognition.project_folder}/docs/experiments/exp_{exp_no}'
            os.makedirs(fig_folder,exist_ok=True)
            fig.savefig(f'{fig_folder}/conf_mat_{dataset_part}.png')
        if plot:
            plt.show()

if __name__ == "__main__":
    from settings import Settings


    ### MODEL DEFINITION
    exp_no = 0
    model_name = 'custom_0'

    # TRAINING HYPERPARAMETERS
    patch_size=128
    batch_size=20
    epochs=50

    settings = Settings(model_name=model_name,
                        exp_no=exp_no,
                        patch_size=patch_size,
                        batch_size=batch_size,
                        epochs=epochs,
                        hot_encoding=True,
                        update=True)()
    classifier = Classifier(settings)
    # print(classifier.get_model(name='resnet18').__dict__)

    ### TRAIN
    classifier.train(patience=10)
    ### TEST
    # classifier.get_conf_mat(dataset_part='val',save=True,plot=True)

    ### FORWARD PASS
    # dataiter = iter(classifier.get_loader('train'))
    # samples = dataiter.next()
    # print(samples['label'])

    # model =classifier.get_model()

    # print(model(samples))