import numpy as np
from PIL import Image
import pickle
import os


def write_to_pickle(mode, data):
    with open(os.path.join('../../', f'cifarfs100-cache-{mode}.pkl'), 'wb') as f:
        pickle.dump(data, f)


nbr_examples={}
for mode in os.listdir("./cifar100/processed"):
    totalmode = 0
    for lbl in os.listdir(os.path.join('./cifar100/processed',mode)):
        totalmode += len(os.listdir(os.path.join('./cifar100/processed',mode,lbl)))
    nbr_examples[mode] = totalmode

print(nbr_examples)

for mode in os.listdir('./cifar100/processed'):
    for lbl in os.listdir(os.path.join('./cifar100/processed',mode)):
        for img in os.listdir(os.path.join('./cifar100/processed',mode,lbl)) :
            res = Image.open(os.path.join('./cifar100/processed',mode,lbl,img)).convert('RGB')
            width, height = res.size
            print(width,height)
            break
        break
    break

for mode in os.listdir('./cifar100/processed'):
    result = {"image_data": np.zeros((nbr_examples[mode], 32, 32,3)),  'class_dict': {}}
    nbr_image = 0
    for lbl in os.listdir(os.path.join('./cifar100/processed',mode)):
        for img in os.listdir(os.path.join('./cifar100/processed',mode,lbl)) :
            image = Image.open(os.path.join('./cifar100/processed',mode,lbl,img)).convert('RGB')
            result["image_data"][nbr_image] = image
            result['class_dict'].setdefault(lbl, []).append(nbr_image)
            nbr_image+=1
    write_to_pickle(mode,result)
