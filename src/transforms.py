import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))

        # sample_processed = sample.copy()
        sample['image'] = torch.from_numpy(image)
        sample['label'] =  torch.from_numpy(label)
        return sample


class Normalize(object):
    """Normalize image pixels."""
    def __init__(self,task):
        self.task=task
    def __call__(self, sample):
        sample['image'] = sample['image']/255.0
        if self.task=='segmentation':
            sample['label'] = sample['label']/255.0
        return sample

class AddAxis(object):
    """docstring for AddAxis"""
    # def __init__(self, arg):
    #     super(AddAxis, self).__init__()
    #     self.arg = arg
    def __call__(self,sample):

        sample['label'] = torch.unsqueeze(sample['label'], 0)
        return sample