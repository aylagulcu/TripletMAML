#!/usr/bin/env python3

import os
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch


from learn2learn.data.utils import download_file_from_google_drive, download_file


def download_pkl(google_drive_id, data_root, mode):
    filename = 'cifar100fs-cache-' + mode
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

class TripletFSCIFAR100(Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/cifarfs.py)
    **Description**
    The CIFAR Few-Shot dataset as originally introduced by Bertinetto et al., 2019.
    It consists of 60'000 colour images of sizes 32x32 pixels.
    The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples.
    The classes are sampled from the CIFAR-100 dataset, and we use the splits from Bertinetto et al., 2019.
    **References**
    1. Bertinetto et al. 2019. "Meta-learning with differentiable closed-form solvers". ICLR.
    **Arguments**
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    **Example**
    ~~~python
    train_dataset = l2l.vision.datasets.CIFARFS(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskGenerator(dataset=train_dataset, ways=ways)
    ~~~
    """


    def __init__(self, root, transform=None, target_transform=None, download=True):
        
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        
        self.mode = None
        self.transform = transform
        self.target_transform = target_transform

        self.download_links = {"test": '14zHv9jooH98Evk7ve5qoeH9BUvz_BchD',
                               "train": '1fOZsCSdfvo-2tIyD41LkT4rzHVuzpwVe',
                               "validation": '1QZymN9HQMXjXsmyv4XkfQqEHgFJPuwdF'}

        self.data = dict()
        self.indexes = {"train": 0, "validation": 0, "test": 0}
        
        for mode in ['train', 'test', 'validation']:
            pickle_file = os.path.join(self.root, 'cifar100fs-cache-' + mode + '.pkl')
            
            if not self._check_exists(mode) and download:
                print('Downloading CIFAR-FS --', mode)
                download_pkl(self.download_links[mode], self.root, mode)
            
            with open(pickle_file, 'rb') as f:
                self.data[mode] = pickle.load(f)
            
            self.data[mode]["image_data"] = torch.from_numpy(self.data[mode]["image_data"]).permute(0, 1, 2, 3).float()
            

    def sample(self, type,k_shots = 1):
        if k_shots == 1 :
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
        elif k_shots == 5:
            # Mode change
            if type != self.mode:
                self.mode = type
                print("Mode Changed:", self.mode)
                self.labels = list(self.data[type]['class_dict'].keys())
                self.label_to_indices = self.data[type]['class_dict']
                self.sample_data = self.data[type]["image_data"]

            nbr_positive_sample = 10
            nbr_negative_labels = 4
            nbr_negative_sample = 40

            index = self.indexes[type] % len(self.labels)

            #1. select a anchor label
            anchor_label = self.labels[index]

            #2. we know that positive files should have same label as anchor
            positive_label = anchor_label # Not even needed to decleare

            #3. we need to select 1 anchor 4 positive for support - 5 positives for query and all of these should be different  => 10 positive samples
            used_positive_indexes = set(np.random.choice(self.label_to_indices[anchor_label], nbr_positive_sample, replace=False))

            #4. Now that we have selected 10 positive examples we are gonna place them to the according elements
            used_positive_indexes = list(used_positive_indexes)

            # [x1a, x1p_1, x1p_2, x1p_3 ... x1p_n] - image tensors
            positive_x = [np.array(self.sample_data[used_positive_indexes[idx]]) for idx in range(nbr_positive_sample)]
            # [y1a, y1p_1, y1p_2, y1p_3 ... y1p_n] - image labels
            # positive_y = [self.all_labels[used_positive_indexes[idx]] for idx in range(nbr_positive_sample)]

            #5. Now we have all the positive examples ready , we can do same operation to get negative indexes.
            # To Pick Negative Examples we have different rules . we can pick all random classes , but if there happend to be same class we need to be sure we use differend indexs.
            remainingLabels = list(set(self.labels) - set([anchor_label]))
            negative_labels = list(np.random.choice(remainingLabels, nbr_negative_labels, replace=False))
            used_negative_indexes = [idx for label in negative_labels for idx in np.random.choice(self.label_to_indices[label], 10, replace=False)]

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

            X_support.append(torch.stack([positive_x[0] for i in range(20)]).float())
            X_support.append(torch.stack([positive_x[i] for i in range(1,5) for x in range(5) ] ).float())
            X_support.append(torch.stack([negative_x[i] for i in range(0,40,2)]).float())

            y_support.append([0 for i in range(20)])
            y_support.append([0 for i in range(20)])
            y_support.append([i for i in range(1,5) for x in range(5)])

            X_query = []
            y_query = []

            X_query.append(torch.stack([positive_x[5] for i in range(20)]).float())
            X_query.append(torch.stack([positive_x[i] for i in range(6,10) for x in range(5) ]).float())
            X_query.append(torch.stack([negative_x[i] for i in range(1,40,2)]).float())

            y_query.append([0 for i in range(20)])
            y_query.append([0 for i in range(20)])
            y_query.append([i for i in range(1,5) for x in range(5)])

            y_support = (torch.Tensor(y_support)).long()
            y_query = (torch.Tensor(y_query)).long()
            
            # Update index value
            self.indexes[type] += 1

            return X_support ,y_support, X_query, y_query


    def __len__(self):
        return len(self.data[self.mode]["image_data"])

    def _check_exists(self, mode):
        return os.path.exists(os.path.join(self.root, 'cifar100fs-cache-' + mode + '.pkl'))