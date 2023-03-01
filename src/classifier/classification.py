from src.classifier.classifier import Classifier
import logging
import torch
import torch.optim as optim
import numpy as np
from src.transforms import Normalize, ToTensor, AddAxis

class ClassifierClassification(Classifier):
    def __init__(self, data_settings, exp_settings):
        super(ClassifierClassification, self).__init__(data_settings,exp_settings)
        self.data_settings = data_settings
        self.exp_settings = exp_settings
        self.logger = logging.getLogger(__name__) 

        # self.loaders = self.get_loaders()

    def get_loaders(self):
        return super().get_loaders(task='classification')

    def train(self):
        model = super().get_model()
        # COST AND OPTIMIZER FUNCS
        my_class_weights = self.exp_settings['training']['class_weight']
        if my_class_weights != None:

            out_feauture_size = self.get_out_feauture_size(model)
            torch_class_weights = torch.zeros(out_feauture_size)
            for i, weight in enumerate(my_class_weights):
                torch_class_weights[i]=weight
        else:
            torch_class_weights=None
        loss_func = torch.nn.CrossEntropyLoss(weight=torch_class_weights)#DiceLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.exp_settings['training']['learning_rate'],
            momentum=0.9)

        # DATA
        loaders = self.get_loaders()

        super().train(model, loss_func, optimizer, loaders)

    def get_model(self):
        return super().get_model()

    def test(self,loader_val=None):
        from tqdm import tqdm
        import pandas as pd
        import seaborn as sn
        import matplotlib.pyplot as plt
        # import matplotlib.pyplot as plt
        # my_sample = next(iter(self.get_loaders()['val']))
        # batch_sample = my_sample['image']#.size()
        if loader_val is None:
            loaders = self.get_loaders()
            loader_val = loaders['val']

        conf_mat = np.zeros(shape=(len(self.data_settings['instance_names']),len(self.data_settings['instance_names']))) #
        # print(conf_mat.shape)
        model = self.get_model()
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model_path = self.exp_settings['model']['path']
        model = self.load_checkpoint(model,model_path,is_train=False)
        # print(model)
        acc_sums = 0
        # loss_func = torch.nn.CrossEntropyLoss()#DiceLoss()
        for data in tqdm(loader_val):
            # forward pass: compute predicted outputs by passing inputs to
            # the model
            outputs = model(data['image'].to(device))
            # READ MODEL AND MOVE IT TO GPU
            pred_int = torch.argmax(outputs.softmax(dim=1),dim=1)
            # print(my_sample['label'])
            gt_data = data['label'].to('cuda:0')
            acc_sum = torch.sum(pred_int == gt_data)#/len(pred)
            acc_sums += acc_sum

            for pred, gt in zip(pred_int,gt_data):
                if pred>conf_mat.shape[0]:
                    self.logger.warn('A value is clipped')
                    pred = int(torch.clip(pred,min=0,max=conf_mat.shape[0]))
                    # print(pred)
                else:
                    conf_mat[pred,gt]+=1
            # gt = data['label'].to(device)
            # loss = loss_func(outputs, gt)
            # print(loss)
        # self.logger.info(f'Accuracy sum: {acc_sums}')
        val_acc = acc_sums/len(loader_val.dataset)
        self.logger.info(f'Accuracy: {val_acc:.2f}')
        # print(conf_mat)
        # plt.figure(figsize = (10,7))
        # df_cm = pd.DataFrame(conf_mat, index = self.data_settings['instance_names'],
        #           columns = self.data_settings['instance_names'])
        # sn.heatmap(df_cm, annot=True)
        # # plt.hist(batch_sample.flatten())
        # plt.savefig('temp.png')
        return conf_mat

    def get_out_feature_size(self,model):
        # from torchsummary import summary
        # print(summary(model.cuda(), (3, 128, 128)))

        last_item_index = len(model.classifier) - 1
        fc = model.classifier.__getitem__(last_item_index)
        return fc.out_features

    # def inference_model(self,model,img_path):
    #     from src.data.dataset.classification import DatasetClassification 
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     transform=Compose([ToTensor(), Normalize(task='classification'), AddAxis()])
    #     dataset_classification = DatasetClassification(self.exp_settings,self.data_settings,dataset_part='val',transform=transform)
    #     img = dataset_classification.read_image(img_path)
    #     sample = {'image_path': img_path,
    #               'image': img}
    #     sample = dataset_classification.transform(sample)

    #     outputs = model(sample['image'].to(device))
    #     print(outputs)
        

