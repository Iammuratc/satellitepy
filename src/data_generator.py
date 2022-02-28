
from torch.utils.data import Dataset, DataLoader
import torch



class DataGenerator(Dataset):

	def __init__(self,)

    def __len__(self):
        return len(self.items) 

    def __getitem__(self, ind):
        if torch.is_tensor(ind):
            ind = ind.tolist()

        img = cv2.cvtColor(cv2.imread(self.items[ind][0],1),cv2.COLOR_BGR2RGB)
        label = self.items[ind][1]
        sample = {'image':img,'labels':label}
        return sample

    def get_items(self):
        items=[]

        img_paths = self.get_img_paths()
        labels = self.get_labels()

        for img_name, img_path in img_paths.items():
            label = labels[img_name]
            items.append([img_path,label])
        return items

    def img_show(self,ind,plot_bbox=True):
        sample = self.__getitem__(ind)
        img = sample['image']
        labels = sample['labels']
        print(labels)
        fig, ax = plt.subplots(1)
        ax.imshow(img,'gray')

        if plot_bbox==True:
            for coords in labels:
                for i, coord in enumerate(coords):
                    # PLOT BBOX
                    ax.plot([coords[i-1][0],coord[0]],[coords[i-1][1],coord[1]],c='r')
                    # PLOT CORNERS
                    # ax.scatter(coord[0],coord[1],c='r',s=5)
        plt.show()