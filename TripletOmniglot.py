#!/usr/bin/env python3

import os
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.omniglot import Omniglot
import numpy as np
import torch
class TripletOmniglot(Dataset):
    """

    [[Source]]()

    **Description**

    This class provides an interface to the Omniglot dataset.

    The Omniglot dataset was introduced by Lake et al., 2015.
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    While the original dataset is separated in background and evaluation sets,
    this class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017.
    The background and evaluation splits are available in the `torchvision` package.

    **References**

    1. Lake et al. 2015. “Human-Level Concept Learning through Probabilistic Program Induction.” Science.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**
    ~~~python
    omniglot = l2l.vision.datasets.FullOmniglot(root='./data',
                                                transform=transforms.Compose([
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)
    omniglot = l2l.data.MetaDataset(omniglot)
    ~~~

    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # Set up both the background and eval dataset
        omni_background = Omniglot(self.root, background=True, download=download)
        # Eval labels also start from 0.
        # It's important to add 964 to label values in eval so they don't overwrite background dataset.
        omni_evaluation = Omniglot(self.root,
                                   background=False,
                                   download=download,
                                   target_transform=lambda x: x + len(omni_background._characters))
        omniglot_backgroundlabels = set(range(len(omni_background._characters)))
        omniglot_evalutionlabels = set(range(964,964+len(omni_evaluation._characters)))
        self.labels_set = list(omniglot_backgroundlabels.union(omniglot_evalutionlabels))

        del omniglot_backgroundlabels
        del omniglot_evalutionlabels
        self.dataset = ConcatDataset((omni_background, omni_evaluation))
        self._bookkeeping_path = os.path.join(self.root, 'omniglot-bookkeeping.pkl')
        self.all_data = self.flatten(self.dataset.datasets)
        self.all_labels = np.array([i[1] for i in self.all_data])
        self.label_to_indices = {label: np.where(self.all_labels == int(label))[0] for label in self.labels_set}

        self.split_train_test_valid()

    def __len__(self):
        return len(self.dataset)
    
    def split_train_test_valid(self):
        np.random.shuffle(self.labels_set)
        self.labels = {"train": [],  "valid": [], "test": []}
        self.indexes = {"train": 0, "valid": 0, "test": 0}
        
        total_labels = len(self.labels_set)
        total_train_labels = int(total_labels * 0.8)
        total_valid_labels = int(total_train_labels * 0.1)

        self.labels["train"] = self.labels_set[: total_train_labels]
        self.labels["valid"] = self.labels["train"][-total_valid_labels: ]
        self.labels["train"] = self.labels["train"][: total_train_labels - total_valid_labels]   
        self.labels["test"] = self.labels_set[total_train_labels: ]

    def flatten(self,dataset):
        return [item for sublist in dataset for item in sublist]

    def sample(self, type):
        nbr_positive_sample = 4
        nbr_negative_sample = 8

        labels = self.labels[type]
        index = self.indexes[type] % len(labels)

        #1. select a anchor label
        anchor_label = labels[index]

        #2. we know that positive files should have same label as anchor
        positive_label = anchor_label # Not even needed to decleare

        #3. we need to select 1 anchor 1 positive for support - 2 positives for query and all of these should be different
        used_positive_indexes = set(np.random.choice(self.label_to_indices[anchor_label], nbr_positive_sample, replace=False))

        #4. Now that we have selected 4 positive examples we are gonna place them to the according elements
        used_positive_indexes = list(used_positive_indexes)

        # [x1a, x1p_1, x1p_2, x1p_3 ... x1p_n] - image tensors
        positive_x = [np.array(self.all_data[used_positive_indexes[idx]][0]) for idx in range(nbr_positive_sample)]
        # [y1a, y1p_1, y1p_2, y1p_3 ... y1p_n] - image labels
        positive_y = [self.all_labels[used_positive_indexes[idx]] for idx in range(nbr_positive_sample)]

        #5. Now we have all the positive examples ready , we can do same operation to get negative indexes.
        # To Pick Negative Examples we have different rules . we can pick all random classes , but if there happend to be same class we need to be sure we use differend indexs.
        remainingLabels = list(set(labels) - set([anchor_label]))
        negative_labels = list(np.random.choice(remainingLabels, nbr_negative_sample // 2, replace=False))
        used_negative_indexes = [idx for label in negative_labels for idx in np.random.choice(self.label_to_indices[label], 2, replace=False)]

        # [x1n_1, x1n_2, x1n_3, ... x1n_n] - image tensors
        negative_x = [np.array(self.all_data[used_negative_indexes[idx]][0]) for idx in range(nbr_negative_sample)]
        # [y1n_1, y1n_2, y1n_3, ... y1n_n] - image labels
        negative_y = [self.all_labels[used_negative_indexes[idx]] for idx in range(nbr_negative_sample)]

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
        y_support.append([1,2,3,4]) # [3,4] tensor veriyor
        
        X_query = []
        y_query = []

        X_query.append(torch.stack([positive_x[2], positive_x[2], positive_x[2], positive_x[2]]).float())
        X_query.append(torch.stack([positive_x[3], positive_x[3], positive_x[3], positive_x[3]]).float())
        X_query.append(torch.stack([negative_x[1], negative_x[3], negative_x[5], negative_x[7]]).float())

        y_query.append([0,0,0,0])
        y_query.append([0,0,0,0])
        y_query.append([1,2,3,4]) # [3,4] tensor veriyor

        y_support = (torch.Tensor(y_support)).long()
        y_query = (torch.Tensor(y_query)).long()
        
        # Update index value
        self.indexes[type] += 1

        return X_support ,y_support, X_query, y_query 