import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
LABEL_NAMES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as Fs


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path,transform=transforms.ToTensor()):
        import csv
        from os import path
        self.data = []
        self.transform = transform
        with open(path.join('labels.csv'), newline='') as f:
            reader = csv.reader(f)
            for fname, label in reader:
                if label != 'Y' and int(fname) in LABEL_NAMES:
                    im_name = '%0*d' % (5, int(fname)+1) + ".jpg"
                    image = Image.open(path.join(dataset_path, im_name))
                    image.load()
                    label_id = LABEL_NAMES.index(int(label))
                    self.data.append((image, label_id))

    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """

        if(idx >= self.__len__()):
            raise IndexError()

        img, lbl = self.data[idx]
        return self.transform(img), lbl


def load_data(dataset_path, num_workers=4, batch_size=256):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, 
                        shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return (self.matrix / (self.matrix.sum(1, keepdim=True) + 1e-5)).cpu()
