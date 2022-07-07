from torchvision.transforms import Compose
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


from transforms import ToTensor, Normalize
from recognition import Recognition
from dataset import RecognitionDataset
from utilities import get_project_folder


class Classifier:
    def __init__(self,path,patch_size):
        self.path = path
        self.patch_size=patch_size
        self.project_folder = get_project_folder()

    def train(self,epochs,batch_size,exp_no,load_last_state=False):

        ### READ MODEL AND MOVE IT TO GPU
        model = self.get_model()
        if os.path.exists(self.path):
            model.load_state_dict(torch.load(self.path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        ### LOSS AND OPTIMIZER FUNCS
        criterion, optimizer = self.get_funcs_for_train(model)

        ### DATA
        dataset = self.get_dataset('train')
        loader_train = self.get_loader(dataset=dataset,batch_size=batch_size)

        ### LOG
        log_folder = f"{self.project_folder}/logs/exp_{exp_no}"
        writer = SummaryWriter(log_dir=log_folder)
        stat_step = 20 # write log at every stat_step*batch_size image

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            with tqdm(loader_train, unit="batch") as tepoch:
                for i,data in enumerate(loader_train):
                    tepoch.set_description(f"Epoch {epoch}")
                    # data is a dict with keys "image", "label"
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    
                    outputs = model(data['image'].to(device))
                    loss = criterion(outputs, data['label'].to(device))

                    writer.add_scalar("Loss/train", loss, epoch)
                    loss.backward()
                    optimizer.step()
                    

                    # print statistics
                    running_loss += loss.item()
                    if i % stat_step == stat_step-1:    # print every 2000 mini-batches
                        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / stat_step:.3f}')
                        tepoch.set_postfix(loss=running_loss/stat_step)
                        running_loss = 0.0

        torch.save(model.cpu().state_dict(), self.path)
        writer.flush()
        writer.close()
        print('Finished Training')

    def get_dataset(self,dataset_part):

        dataset_id = 'f73e8f1f-f23f-4dca-8090-a40c4e1c260e'
        dataset_name = 'Gaofen'

        recognition_instance = Recognition(dataset_id,dataset_part,dataset_name,self.patch_size)
        dataset = RecognitionDataset(recognition_instance,transform=self.get_transform(dataset_part))
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
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True) # resnet34  resnet50   resnet101   resnet152
        return model


    def get_funcs_for_train(self,model):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        return criterion, optimizer

    def get_conf_mat(self,dataset_part,exp_no,save=False,plot=True):
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
    from utilities import get_project_folder

    project_folder=get_project_folder()
    exp_no = 1
    patch_size=128    
    batch_size = 20
    epochs=50
    model_path = f'{project_folder}/binaries/resnet34_v{exp_no}.pth'

    classifier = Classifier(path=model_path,patch_size=patch_size)
    # print(classifier.get_model().eval())

    ### TRAIN
    # classifier.train(epochs=epochs,batch_size=batch_size,exp_no=exp_no)
    ### TEST
    classifier.get_conf_mat(dataset_part='val',exp_no=exp_no,save=True,plot=True)

    ### FORWARD PASS
    # dataiter = iter(classifier.get_loader('train'))
    # samples = dataiter.next()
    # print(samples['label'])

    # model =classifier.get_model()

    # print(model(samples))