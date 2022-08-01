import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))

        sample_processed = sample.copy()
        sample_processed['image'] = torch.from_numpy(image)
        sample_processed['label'] =  torch.from_numpy(label)
        return sample_processed


class Normalize(object):
    """Normalize image pixels."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # image = image.transpose((2, 0, 1))
        sample_processed = sample.copy()
        sample_processed['image'] = sample['image']/255.0
        return sample_processed