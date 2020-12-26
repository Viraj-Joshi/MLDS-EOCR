import csv
import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as Fs


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path,transform=transforms.ToTensor()):
        from os import path
        self.data = []
        self.transform = transform
        LABEL_NAMES = [str(i) for i in range(0,43)]
        with open('labels.csv') as f:
            reader = csv.reader(f)
            for fname, label in reader:
                if label in LABEL_NAMES:
                    im_name = '%0*d' % (5, int(fname)+1) + ".jpg"
                    image = Image.open(path.join(dataset_path, im_name))
                    image.load()
                    label_id = LABEL_NAMES.index(label)
                    self.data.append((image, label_id))

    def __len__(self):
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

def generate_transition_matrix():
    LABEL_NAMES = [str(i) for i in range(0,43)]
    states = []
    with open('labels.csv') as f:
        reader = csv.reader(f)
        for fname, label in reader:
            if label in LABEL_NAMES:
                states.append(int(label))
    
    def transition_matrix(transitions):
        n = 1+ max(transitions) #number of states

        M = [[0]*n for _ in range(n)]

        for (i,j) in zip(transitions,transitions[1:]):
            M[i][j] += 1

        #now convert to probabilities:
        for row in M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
        return M
    
    M = transition_matrix(states)
    for row in M: print(' '.join('{0:.2f}'.format(x) for x in row))

    return M

def load_data(dataset_path, num_workers=0, batch_size=256, **kwargs):
    dataset = SuperTuxDataset(dataset_path,**kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, 
                        shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
  
def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()

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
