#!/usr/bin/env python3

import os
import shutil


# download link for cifar100 file https://drive.google.com/u/1/uc?id=1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI&export=download download it , extract the file.

def process_zip():
    print('Creating CIFARFS splits')
    if not os.path.exists("./cifar100/processed"):
        os.mkdir("./cifar100/processed")
    split_path = os.path.join("./cifar100", 'splits', 'bertinetto')
    train_split_file = os.path.join(split_path, 'train.txt')
    valid_split_file = os.path.join(split_path, 'val.txt')
    test_split_file = os.path.join(split_path, 'test.txt')

    source_dir = os.path.join('./cifar100', 'data')
    print(source_dir)
    for fname, dest in [(train_split_file, 'train'),
                        (valid_split_file, 'val'),
                        (test_split_file, 'test')]:
        dest_target = os.path.join("./cifar100/processed", dest)
        if not os.path.exists(dest_target):
            os.mkdir(dest_target)
        with open(fname) as split:
            for label in split.readlines():
                source = os.path.join(source_dir, label.strip())
                target = os.path.join(dest_target, label.strip())
                shutil.copytree(source, target)


if __name__ == '__main__':
    process_zip()