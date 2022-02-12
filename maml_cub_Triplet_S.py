
# %%
from torchvision import transforms
import random
import numpy as np
import torch
from torch import nn, optim

import learn2learn as l2l
from  TripletCUB import *
from cnn4_triplet import *
from losses import *

print(torch.__version__)
import matplotlib.pyplot as plt

# %%
# Batch_size, W, H, C
# Batch_size, C, W, H

def show_triplets(img1, img2, img3,img4,img5,img6,img7,img8,img9,img10,img11,img12,label1,label2,label3,label4,label5,label6,label7,label8,label9,label10,label11,label12,title):
    """Display image for testing"""
    fig = plt.figure(figsize=(15, 15))
    plt.title(title,pad =20)
    plt.axis('off')
    ax1 = fig.add_subplot(4,3,1)
    ax2 = fig.add_subplot(4,3,2)
    ax3 = fig.add_subplot(4,3,3)
    ax4 = fig.add_subplot(4,3,4)
    ax5 = fig.add_subplot(4,3,5)
    ax6 = fig.add_subplot(4,3,6)
    ax7 = fig.add_subplot(4,3,7)
    ax8 = fig.add_subplot(4,3,8)
    ax9 = fig.add_subplot(4,3,9)
    ax10 = fig.add_subplot(4,3,10)
    ax11 = fig.add_subplot(4,3,11)
    ax12 = fig.add_subplot(4,3,12)

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()
    ax5.set_axis_off()
    ax6.set_axis_off()
    ax7.set_axis_off()
    ax8.set_axis_off()
    ax9.set_axis_off()
    ax10.set_axis_off()
    ax11.set_axis_off()
    ax12.set_axis_off()

    ax1.imshow(img1.permute(1, 2, 0).cpu().int())
    ax2.imshow(img5.permute(1, 2, 0).cpu().int())
    ax3.imshow(img9.permute(1, 2, 0).cpu().int())
    ax4.imshow(img2.permute(1, 2, 0).cpu().int())
    ax5.imshow(img6.permute(1, 2, 0).cpu().int())
    ax6.imshow(img10.permute(1, 2, 0).cpu().int())
    ax7.imshow(img3.permute(1, 2, 0).cpu().int())
    ax8.imshow(img7.permute(1, 2, 0).cpu().int())
    ax9.imshow(img11.permute(1, 2, 0).cpu().int())
    ax10.imshow(img4.permute(1, 2, 0).cpu().int())
    ax11.imshow(img8.permute(1, 2, 0).cpu().int())
    ax12.imshow(img12.permute(1, 2, 0).cpu().int())

    ax1.title.set_text(label1.numpy())
    ax2.title.set_text(label5.numpy())
    ax3.title.set_text(label9.numpy())
    ax4.title.set_text(label2.numpy())
    ax5.title.set_text(label6.numpy())
    ax6.title.set_text(label10.numpy())
    ax7.title.set_text(label3.numpy())
    ax8.title.set_text(label7.numpy())
    ax9.title.set_text(label11.numpy())
    ax10.title.set_text(label4.numpy())
    ax11.title.set_text(label8.numpy())
    ax12.title.set_text(label12.numpy())

    plt.show()
    
def show_helper(tripletx,triplety,title):
    #for i in range(len(tripletx)):
    #    show_triplets(*tripletx[i],*triplety[i])

    show_triplets(*tripletx[0],*tripletx[1],*tripletx[2],*triplety[0],*triplety[1],*triplety[2],title)
        

# %%
def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    # return (predictions == targets).sum().float() / targets.size(0)

    mask = np.array([True, False, False, False, False, False, False, False, True, True, True, True])
    return (predictions[mask] == targets[mask]).sum().float() / targets[mask].size(0)

# %%
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
        learner.adapt(train_error, allow_unused=True)

    # Evaluate the adapted model
    predictions = learner(evaluation_data[0], evaluation_data[1], evaluation_data[2])
    valid_error= loss(predictions[0],predictions[1],predictions[2], torch.vstack([predictions[3],predictions[4],predictions[5]]), torch.hstack([evaluation_labels[0],evaluation_labels[1],evaluation_labels[2]]))
    valid_accuracy = accuracy(torch.vstack([predictions[3],predictions[4],predictions[5]]), torch.hstack([evaluation_labels[0],evaluation_labels[1],evaluation_labels[2]]))
    return valid_error, valid_accuracy

# %%
def main(
        ways=5, # in our triplet implementation, number of distinct classes is 5
        shots=1,
        meta_lr=0.001, # as in MAML
        fast_lr=0.01, # as in MAML
        meta_batch_size=4, # Maml Omniglot:32; miniImageNet: 4 
        adaptation_steps=5,
        test_adaptation_steps=3,
        num_iterations= 10, # as in MAML
        cuda=False,
        seed=42,
        num_test_episodes= 1
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')    

    triplet_cub_dataset = TripletCUB(root='./', download=True,
                                transform = transforms.Compose([transforms.ToTensor()])
                                )

    # Create model
    model = TripletCNN4(output_size= ways, hidden_size=64, layers=4, channels=3, max_pool=True, embedding_size=1600)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=True)
    opt = optim.Adam(maml.parameters(), meta_lr) # meta-update
     
    margin= 1.0
    combined_loss_fn= CombinedLoss(margin)

    # # META TRAIN #
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
            learner = maml.clone()
            batch = triplet_cub_dataset.sample("train") 
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
            batch = triplet_cub_dataset.sample("validation")
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
    # f = open("result_train.csv", "w")
    # f.write('\t'.join(('tr_er', 'val_er', 'tr_acc', 'val_acc')))
    # f.write('\n')
    # for (tr_er, val_er, tr_acc, val_acc) in zip(total_meta_train_error, total_meta_valid_error, total_meta_train_accuracy, total_meta_valid_accuracy):
    #     items= (str(tr_er),',', str(val_er),',',  str(tr_acc),',',  str(val_acc))
    #     f.write('\t'.join(items))
    #     f.write('\n')
    # f.close()

    #-- Save model parameters #
    #-- https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model,"./maml_model_cubsave.pt")

    # Create model using saved parameters:
    model = TripletCNN4(output_size= ways, hidden_size=64, layers=4, channels=3, max_pool=True, embedding_size=1600)
    model = torch.load("./maml_model_cubsave.pt")
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=True)

    # META TEST #
    total_meta_test_error = []
    total_meta_test_accuracy = []

    for i in range(num_test_episodes):
        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        # Compute meta-testing loss
        learner = maml.clone()
        batch = triplet_cub_dataset.sample("test") 
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           combined_loss_fn,
                                                           test_adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)

        # To Do:3
        # Print the images in the current batch in the form of triplets, their labels and also the accuracy for that batch:
        #RUN ONLY ON INTERACTIVE MODE
        show_helper(batch[0],batch[1],"Support Images")
        show_helper(batch[2],batch[3],"Query Images")


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

# %%
if __name__ == '__main__':
    main()
# %%
