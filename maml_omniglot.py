"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import random

import learn2learn as l2l
import numpy as np
import torch
from torch import nn, optim


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch #labels like: tensor([4, 4, 0, 0, 1, 1, 2, 2, 3, 3], device='cuda:0')
    data, labels = data.to(device), labels.to(device)


    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool) # support
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices) # query
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]


    # Adapt the model
    for step in range(adaptation_steps):
        out= learner(adaptation_data)
        train_error = loss(out, adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=2, # number of tasks, AG: 32 original value
        adaptation_steps=1,
        num_iterations=10, # 60000,
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Load train/validation/test tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot',
                                                  train_ways=ways,
                                                  train_samples=2*shots,
                                                  test_ways=ways,
                                                  test_samples=2*shots,
                                                  num_tasks=20000,learner
                                                  root='~/data',
    )

    # BenchmarkTasksets(train, validation, test); len(tasksets) is 3
    # len(tasksets[0]) = 20000; len(tasksets[1]) = 20000; len(tasksets[2]) = 20000; equal to num_tasks parameter value!
    # type(tasksets[0]) <class 'learn2learn.data.task_dataset.TaskDataset'>
    
    # Create model
    model = l2l.vision.models.OmniglotFC(28 ** 2, ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr)
    opt = optim.Adam(maml.parameters(), meta_lr) # meta-update
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        print('Iteration: ', iteration)

        opt.zero_grad() # for each batch, gradients should be cleaned.
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss

            print('Task no: ', task)

            learner = maml.clone()

            # learner and maml.module weights are the same.
            # print(id(maml.module.classifier.weight))
            # print(id(learner.classifier.weight))

            # print('***BEFORE FAST ADAPT***')
            # print('Weights:...')
            # print('maml module classifier weights \n',maml.module.classifier.weight)
            # print('learner classifier weights \n',learner.classifier.weight)

            # print('Now grads:...')
            # print('maml module classifier weights grads \n',maml.module.classifier.weight.grad)
            # print('learner classifier weights grads \n',learner.classifier.weight.grad)

            batch = tasksets.train.sample() # returns one task
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            # After fast adapt, maml.module weights do not change; but learner weights do change.
            # maml.module weights grads none; learner grads none


            print('***RIGHT AFTER FAST ADAPT***')
            print('Weights:...')
            print('maml module classifier weights \n',maml.module.classifier.weight)
            print('learner classifier weights \n',learner.classifier.weight)

            print('Now grads:...')
            print('maml module classifier weights grads \n',maml.module.classifier.weight.grad)
            print('learner classifier weights grads \n',learner.classifier.weight.grad)


            evaluation_error.backward()
            # Both maml.module weights and learner weights same as before
            # maml.module weights grads gets filled; but learner grads still none


            print('***RIGHT AFTER FAST BACKWARD()***')
            print('Weights:...')
            print('maml module classifier weights \n',maml.module.classifier.weight)
            print('learner classifier weights \n',learner.classifier.weight)

            print('Now grads:...')
            print('maml module classifier weights grads \n',maml.module.classifier.weight.grad)
            print('learner classifier weights grads \n',learner.classifier.weight.grad)


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
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        print('***RIGHT AFTER OPT STEP()***')
        print('Weights:...')
        print('maml module classifier weights \n',maml.module.classifier.weight)
        print('learner classifier weights \n',learner.classifier.weight)

        print('Now grads:...')
        print('maml module classifier weights grads \n',maml.module.classifier.weight.grad)
        print('learner classifier weights grads \n',learner.classifier.weight.grad)

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    main()
