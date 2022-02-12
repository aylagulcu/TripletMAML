#!/usr/bin/env python3

import os
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

from learn2learn.data.utils import download_file_from_google_drive, download_file


def download_pkl(google_drive_id, data_root, mode):
    filename = 'birds-cache-' + mode
    file_path = os.path.join(data_root, filename)

    if not os.path.exists(file_path + '.pkl'):
        print('Downloading:', file_path + '.pkl')
        download_file_from_google_drive(google_drive_id, file_path + '.pkl')
    else:
        print("Data was already downloaded")


def index_classes(items):
    idx = {}
    for i in items:
        if (i not in idx):
            idx[i] = len(idx)
    return idx

class TripletCUB(Dataset):


    def __init__(self, root, transform=None, target_transform=None, download=True , mode = 'all'):
        
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        
        self.mode = None
        self.transform = transform
        self.target_transform = target_transform

        self.download_links = {"test": '1LqCCH5zNdVdD0R4IfPqGL-NkoznFEq-w',
                               "train": '1d2MmOsLgHKw6ofAt0w9BURSBOK4FjDmk',
                               "validation": '1TfXpSOnqONesc4Y7DgSkUvFIXKexSUv1'}

        self.data = dict()
        self.indexes = {"train": 0, "validation": 0, "test": 0}
        
        for mode in ['train', 'test', 'validation']:
            pickle_file = os.path.join(self.root, 'birds-cache-' + mode + '.pkl')
            
            if not self._check_exists(mode) and download:
                print('Downloading CUB --', mode)
                download_pkl(self.download_links[mode], self.root, mode)
            
            with open(pickle_file, 'rb') as f:
                self.data[mode] = pickle.load(f)
            
            self.data[mode]["image_data"] = torch.from_numpy(self.data[mode]["image_data"]).permute(0,1,2,3).float()
        

    def sample(self, type):
        # Mode change
        if type != self.mode:
            self.mode = type
            print("Mode Changed:", self.mode)
            self.labels = list(self.data[type]['class_dict'].keys())
            self.label_to_indices = self.data[type]['class_dict']
            self.sample_data = self.data[type]["image_data"]
        
        nbr_positive_sample = 4
        nbr_negative_sample = 8

        index = self.indexes[type] % len(self.labels)

        #1. select a anchor label
        anchor_label = self.labels[index]

        #2. we know that positive files should have same label as anchor
        positive_label = anchor_label # Not even needed to decleare

        #3. we need to select 1 anchor 1 positive for support - 2 positives for query and all of these should be different
        used_positive_indexes = set(np.random.choice(self.label_to_indices[anchor_label], nbr_positive_sample, replace=False))

        #4. Now that we have selected 4 positive examples we are gonna place them to the according elements
        used_positive_indexes = list(used_positive_indexes)

        # [x1a, x1p_1, x1p_2, x1p_3 ... x1p_n] - image tensors
        positive_x = [np.array(self.sample_data[used_positive_indexes[idx]]) for idx in range(nbr_positive_sample)]
        # [y1a, y1p_1, y1p_2, y1p_3 ... y1p_n] - image labels
        # positive_y = [self.all_labels[used_positive_indexes[idx]] for idx in range(nbr_positive_sample)]

        #5. Now we have all the positive examples ready , we can do same operation to get negative indexes.
        # To Pick Negative Examples we have different rules . we can pick all random classes , but if there happend to be same class we need to be sure we use differend indexs.
        remainingLabels = list(set(self.labels) - set([anchor_label]))
        negative_labels = list(np.random.choice(remainingLabels, nbr_negative_sample // 2, replace=False))
        used_negative_indexes = [idx for label in negative_labels for idx in np.random.choice(self.label_to_indices[label], 2, replace=False)]

        # [x1n_1, x1n_2, x1n_3, ... x1n_n] - image tensors
        negative_x = [np.array(self.sample_data[used_negative_indexes[idx]]) for idx in range(nbr_negative_sample)]
        # [y1n_1, y1n_2, y1n_3, ... y1n_n] - image labels
        # negative_y = [self.all_labels[used_negative_indexes[idx]] for idx in range(nbr_negative_sample)]

        X_support = []
        y_support = []

        # Transform
        if self.transform is not None:
            for idx, x in enumerate(positive_x):
                positive_x[idx] = self.transform(positive_x[idx])
            
            for idx, x in enumerate(negative_x):
                negative_x[idx] = self.transform(negative_x[idx])

        X_support.append(torch.stack([positive_x[0], positive_x[0], positive_x[0], positive_x[0]]).float())
        X_support.append(torch.stack([positive_x[1], positive_x[1], positive_x[1], positive_x[1]]).float())
        X_support.append(torch.stack([negative_x[0], negative_x[2], negative_x[4], negative_x[6]]).float())

        y_support.append([0,0,0,0])
        y_support.append([0,0,0,0])
        y_support.append([1,2,3,4])

        X_query = []
        y_query = []

        X_query.append(torch.stack([positive_x[2], positive_x[2], positive_x[2], positive_x[2]]).float())
        X_query.append(torch.stack([positive_x[3], positive_x[3], positive_x[3], positive_x[3]]).float())
        X_query.append(torch.stack([negative_x[1], negative_x[3], negative_x[5], negative_x[7]]).float())

        y_query.append([0,0,0,0])
        y_query.append([0,0,0,0])
        y_query.append([1,2,3,4])

        y_support = (torch.Tensor(y_support)).long()
        y_query = (torch.Tensor(y_query)).long()
        
        # Update index value
        self.indexes[type] += 1

        return X_support ,y_support, X_query, y_query


    def __len__(self):
        return len(self.data[self.mode]["image_data"])

    def _check_exists(self, mode):
        return os.path.exists(os.path.join(self.root, 'birds-cache-' + mode + '.pkl'))


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    cub = TripletCUB('./',transform=transform)
    print(len(cub.data['train']['image_data']))
    print(len(cub.data['test']['image_data']))
    print(len(cub.data['validation']['image_data']))

    """ print(cub.data['validation'])
    print(cub.data['validation']['image_data'])
    print(cub.data['validation']['class_dict']) """
    a = cub.sample('train')
    print(a)
    print(a[0][0].shape)



