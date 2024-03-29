#!/usr/bin/env python3
import os
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import random

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
        

    def sample(self, type,k_shots = 1, mode = "maml",samples_per_class = 47):
        if mode == "maml":
            if k_shots == 1 :
                # Mode change
                if type != self.mode:
                    self.mode = type
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
        elif mode == "img_retrieval":
            self.mode = type
            self.labels = list(self.data[type]['class_dict'].keys())
            self.label_to_indices = self.data[type]['class_dict']
            self.sample_data = self.data[type]["image_data"]
            index = self.indexes[type] % len(self.labels)
            
            nbr_positive_sample = 2
            nbr_negative_sample = 4

            index = self.indexes[type] % len(self.labels)

            #1. select a anchor label
            anchor_label = self.labels[index]

            #2. we know that positive files should have same label as anchor
            positive_label = anchor_label # Not even needed to decleare

            #3. we need to select 1 anchor 1 positive for support  all of these should be different
            used_positive_indexes = set(np.random.choice(self.label_to_indices[anchor_label], nbr_positive_sample, replace=False))
            
            #4. Now that we have selected 2 positive examples we are gonna place them to the according elements
            used_positive_indexes = list(used_positive_indexes)

            # [x1a, x1p_1, x1p_2, x1p_3 ... x1p_n] - image tensors
            positive_x = [np.array(self.sample_data[used_positive_indexes[idx]]) for idx in range(nbr_positive_sample)]
            # [y1a, y1p_1, y1p_2, y1p_3 ... y1p_n] - image labels
            # positive_y = [self.all_labels[used_positive_indexes[idx]] for idx in range(nbr_positive_sample)]

            #5. Now we have all the positive examples ready , we can do same operation to get negative indexes.
            # To Pick Negative Examples we have different rules . we can pick all random classes , but if there happend to be same class we need to be sure we use differend indexs.
            remainingLabels = list(set(self.labels) - set([anchor_label]))
            negative_labels = list(np.random.choice(remainingLabels, nbr_negative_sample , replace=False))
            used_negative_indexes = [idx for label in negative_labels for idx in np.random.choice(self.label_to_indices[label], 1, replace=False)]

            # [x1n_1, x1n_2, x1n_3, ... x1n_n] - image tensors
            negative_x = [np.array(self.sample_data[used_negative_indexes[idx]]) for idx in range(nbr_negative_sample)]

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
            X_support.append(torch.stack([negative_x[0], negative_x[1], negative_x[2], negative_x[3]]).float())

            y_support.append([0,0,0,0])
            y_support.append([0,0,0,0])
            y_support.append([1,2,3,4])
            
            #we need to get remaining images(remaining indexes) in the selected label
            remaining_poz_indexes = list(set(self.label_to_indices[anchor_label]) - set(used_positive_indexes))
            random.shuffle(remaining_poz_indexes)

            remaining_poz_indexes= remaining_poz_indexes[:samples_per_class]
            
            remaining_negatives = [list(set(self.label_to_indices[label])-set(used_negative_indexes))for label in negative_labels ]
            
            for i in range(0,len(remaining_negatives)):
                random.shuffle(remaining_negatives[i])

            remaining_negatif_images = []
            for i in range(0,len(remaining_negatives)):
                for idx in remaining_negatives[i][:samples_per_class]:
                    remaining_negatif_images.append((idx,negative_labels[i]))

            #remaining_negatif_images = [(idx,label) for label in negative_labels for idx in set(self.label_to_indices[label])-set(used_negative_indexes)]
            
            X_query = []
            y_query = []
            
            
            for idx , x in enumerate(remaining_poz_indexes):
                img = np.array(self.sample_data[remaining_poz_indexes[idx]])
                if self.transform is not None:
                    img = self.transform(img)
                X_query.append(img)
                y_query.append(0)
            
            label_remapping = {}
            mapping_counter = 1
            for idx , x in enumerate(remaining_negatif_images):
                img = np.array(self.sample_data[remaining_negatif_images[idx][0]])
                if self.transform is not None:
                    img = self.transform(img)
                X_query.append(img)
                if label_remapping.get(x[1],None) is None :
                    label_remapping[x[1]] = mapping_counter
                    mapping_counter += 1
                y_query.append(label_remapping.get(x[1],None))
            
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
    import matplotlib.pyplot as plt
    def show_triplets(data,label):
        """Display image for testing"""
        fig = plt.figure(figsize=(45, 45))
        plt.title(label,pad =20)
        plt.axis('off')
        axislist = [fig.add_subplot(20,3,i) for i in range(1,61)]
        
        for i in axislist:
            i.set_axis_off()
        
        img_counter = 0
        label_counter = 60

        for number , ax in enumerate(axislist):
            if number % 3 == 0 :
                img_counter = number // 3 
                label_counter = 60 + number // 3
            ax.imshow(data[img_counter].permute(1, 2, 0).cpu().int())
            ax.title.set_text(data[label_counter].numpy())
            img_counter = img_counter + 20
            label_counter = label_counter +20

        plt.show()
        
    def show_helper(tripletx,triplety,title):
        #for i in range(len(tripletx)):
        #    show_triplets(*tripletx[i],*triplety[i])
        show_triplets([*tripletx[0],*tripletx[1],*tripletx[2],*triplety[0],*triplety[1],*triplety[2]],title)

    transform = transforms.Compose([transforms.ToTensor()])
    cub = TripletCUB('./',transform=transform)
    print(len(cub.data['train']['image_data']))
    print(len(cub.data['test']['image_data']))
    print(len(cub.data['validation']['image_data']))

    """ print(cub.data['validation'])
    print(cub.data['validation']['image_data'])
    print(cub.data['validation']['class_dict']) """
    a = cub.sample('train',5)
    show_helper(a[0],a[1],"Support Images")
    show_helper(a[2],a[3],"Query Images")





