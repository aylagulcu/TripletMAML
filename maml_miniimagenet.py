#!/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load mini-ImageNet, and
    * sample tasks and split them in adaptation and evaluation sets.

To contrast the use of the benchmark interface with directly instantiating mini-ImageNet datasets and tasks, compare with `protonet_miniimagenet.py`.
"""

import random
import numpy as np

import torch
from torch import nn, optim

import learn2learn as l2l

from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)




def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def main(
        ways=5,
        shots=1, 
        meta_lr=0.001,
        fast_lr=0.01,
        meta_batch_size=2, # for miniimage 1shot:4,
        adaptation_steps=5,
        test_adaptation_steps=10,
        num_iterations=60000, #
        num_test_episodes= 600,
        cuda=True,
        seed=42,
):

    run_details= "maml_mini-imagenet64_batchsize"+ str(meta_batch_size)+ "_shots"+ str(shots) + "_with_optimizer"
    PATH= "./"+ run_details+ ".pt"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet',  
                                                  train_samples=2*shots,
                                                  train_ways=ways,
                                                  test_samples=2*shots,
                                                  test_ways=ways,
                                                  root='~/data',
    )

    # Create model
    model = l2l.vision.models.CNN4(ways, hidden_size=64, layers=4, max_pool=True, embedding_size= 1600 ) #minimagenet embedding: 800
    # model = l2l.vision.models.CNN4(ways, hidden_size=32, layers=4, max_pool=True, embedding_size= 800 ) #minimagenet embedding: 800


    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    # META TRAIN #
    total_meta_train_error = []
    total_meta_train_accuracy = []
    total_meta_valid_error = []
    total_meta_valid_accuracy = []

    valid_loss_min = np.Inf # track change in validation loss   

    for iteration in range(num_iterations):
        if iteration % 50== 0:
            print('Iteration: ', iteration)

        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        # print('\n')
        # print('Iteration', iteration)
        # print('Meta Train Error', meta_train_error / meta_batch_size)
        # print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        # print('Meta Valid Error', meta_valid_error / meta_batch_size)
        # print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        total_meta_train_error.append(meta_train_error / meta_batch_size)
        total_meta_train_accuracy.append(meta_train_accuracy / meta_batch_size)
        total_meta_valid_error.append(meta_valid_error / meta_batch_size)
        total_meta_valid_accuracy.append(meta_valid_accuracy / meta_batch_size)

        # save model if validation loss has decreased
        if meta_valid_error <= valid_loss_min:
            #-- Save model parameters #
            torch.save({
                'epoch': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()
                }, PATH)
            valid_loss_min = meta_valid_error

    # write training results:

    f = open("./"+ run_details+"result_train.csv", "w")
    f.write('\t'.join(('tr_er', 'val_er', 'tr_acc', 'val_acc')))
    f.write('\n')
    for (tr_er, val_er, tr_acc, val_acc) in zip(total_meta_train_error, total_meta_valid_error, total_meta_train_accuracy, total_meta_valid_accuracy):
        items= (str(tr_er),',', str(val_er),',',  str(tr_acc),',',  str(val_acc))
        f.write('\t'.join(items))
        f.write('\n')
    f.close()

    total_meta_test_error = []
    total_meta_test_accuracy = []

    for i in range(num_test_episodes):
        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample() 
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               test_adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    
        total_meta_test_error.append(meta_test_error)
        total_meta_test_accuracy.append(meta_test_accuracy)
        print(i, 'Test accuracy: ', str(meta_test_accuracy))
    
    total_meta_test_accuracy = np.array(total_meta_test_accuracy)
    mean = np.mean(total_meta_test_accuracy)
    std = np.std(total_meta_test_accuracy, 0)
    ci95 = 1.96*std/np.sqrt(num_test_episodes)
    
    print('Average test accuracy: ', mean, '+/-', ci95)

    # write test results:
    f = open("./"+ run_details+"result_test.csv", "w")
    f.write('\t'.join(('test_er',',', 'test_acc')))
    f.write('\n')
    for (test_er, test_acc) in zip(total_meta_test_error, total_meta_test_accuracy):
        items= (str(test_er),',', str(test_acc))
        f.write('\t'.join(items))
        f.write('\n')
    f.write('\t'.join((str(mean),',', str(ci95))))
    f.close()



if __name__ == '__main__':
    main()
