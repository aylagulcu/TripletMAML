"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

from torchvision import transforms
import random
import numpy as np
import torch
from torch import nn, optim

import learn2learn as l2l
from  TripletOmniglot import *
from cnn4_triplet import *
from losses import *

print(torch.__version__)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch
    adaptation_data[0]= adaptation_data[0].to(device)
    adaptation_data[1]= adaptation_data[1].to(device)
    adaptation_data[2]= adaptation_data[2].to(device)

    adaptation_labels= adaptation_labels.to(device)

    evaluation_data[0]= evaluation_data[0].to(device)
    evaluation_data[1]= evaluation_data[1].to(device)
    evaluation_data[2]= evaluation_data[2].to(device)

    evaluation_labels= evaluation_labels.to(device)

    # Adapt the model
    for step in range(adaptation_steps):
        out= learner(adaptation_data[0], adaptation_data[1], adaptation_data[2] )
        train_error = loss(out[0],out[1],out[2], torch.vstack([out[3],out[4],out[5]]), torch.hstack([adaptation_labels[0],adaptation_labels[1],adaptation_labels[2]]))
        learner.adapt(train_error, allow_unused=True, allow_nograd=True)

    # Evaluate the adapted model
    predictions = learner(evaluation_data[0], evaluation_data[1], evaluation_data[2])
    valid_error= loss(predictions[0],predictions[1],predictions[2], torch.vstack([predictions[3],predictions[4],predictions[5]]), torch.hstack([evaluation_labels[0],evaluation_labels[1],evaluation_labels[2]]))
    valid_accuracy = accuracy(torch.vstack([predictions[3],predictions[4],predictions[5]]), torch.hstack([evaluation_labels[0],evaluation_labels[1],evaluation_labels[2]]))
    return valid_error, valid_accuracy


def main(
        ways=5, # in our triplet implementation, number of distinct classes is 5
        shots=1,
        meta_lr=0.001, # as in MAML
        fast_lr=0.4, # as in MAML
        meta_batch_size=32, # number of tasks, as in MAML
        adaptation_steps=1,
        test_adaptation_steps=3,
        num_iterations= 60000, # as in MAML
        cuda=True,
        seed=42,
        num_test_episodes= 600
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda')
    
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    

    triplet_omniglot_dataset = TripletOmniglot(root='~/data', download=True,
                                transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((28,28)),])
                                )

    # Create model
    model = TripletCNN4(output_size= ways, hidden_size=64, layers=4, channels=1, max_pool=False, embedding_size=256) # For Omniglot, if maxpool embedding size: 64; if nomaxpool(stride 2) then embedding size: 256

    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=True) 
    opt = optim.Adam(maml.parameters(), meta_lr) # meta-update
    
    #margin= 1.0
    lamda= 0.5
    combined_loss_fn= CombinedLoss2(lamda)

    # META TRAIN #
    total_meta_train_error = []
    total_meta_train_accuracy = []
    total_meta_valid_error = []
    total_meta_valid_accuracy = []

    for iteration in range(num_iterations):
        print('Iteration: ', iteration)

        opt.zero_grad() # for each batch, gradients should be cleaned.
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            # print('Task no: ', task)
            learner = maml.clone()
            batch = triplet_omniglot_dataset.sample("train") 
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               combined_loss_fn,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = triplet_omniglot_dataset.sample("valid")
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               combined_loss_fn,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()


        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        total_meta_train_error.append(meta_train_error / meta_batch_size)
        total_meta_train_accuracy.append(meta_train_accuracy / meta_batch_size)
        total_meta_valid_error.append(meta_valid_error / meta_batch_size)
        total_meta_valid_accuracy.append(meta_valid_accuracy / meta_batch_size)


    # write training results:
    f = open("result_train.csv", "w")
    f.write('\t'.join(('tr_er', 'val_er', 'tr_acc', 'val_acc')))
    f.write('\n')
    for (tr_er, val_er, tr_acc, val_acc) in zip(total_meta_train_error, total_meta_valid_error, total_meta_train_accuracy, total_meta_valid_accuracy):
        items= (str(tr_er),',', str(val_er),',',  str(tr_acc),',',  str(val_acc))
        f.write('\t'.join(items))
        f.write('\n')
    f.close()

    # META TEST #
    total_meta_test_error = []
    total_meta_test_accuracy = []

    for i in range(num_test_episodes):
        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        # Compute meta-testing loss
        learner = maml.clone()
        batch = triplet_omniglot_dataset.sample("test") 
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                        learner,
                                                        combined_loss_fn,
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
    f = open("result_test.csv", "w")
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