"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""
import matplotlib.pyplot as plt

from torchvision import transforms
import random
import numpy as np
import torch
from torch import nn, optim

import learn2learn as l2l
from Triplets import *
from backbone import *
from losses import *

Triplet_Model_Parameter = {
    "CIFARFS": {"data": TripletFSCIFAR100, "root": "../data", "download": True, "transform": transforms.Compose([transforms.ToTensor()]), "hidden_size": 64, "layers": 4, "channels": 3, "max_pool": True, "embedding_size": 256, "margin": 1.0},
    "CUB": {"data": TripletCUB, "root": "../data", "download": True, "transform": transforms.Compose([transforms.ToTensor()]), "hidden_size": 64, "layers": 4, "channels": 3, "max_pool": True, "embedding_size": 1600, "margin": 1.0},
    "FLOWERS": {"data": TripletFlowers, "root": "../data", "download": True, "transform": transforms.Compose([transforms.ToTensor()]), "hidden_size": 64, "layers": 4, "channels": 3, "max_pool": True, "embedding_size": 1600, "margin": 1.0},
    "MINIIMAGENET": {"data": TripletMiniImageNet, "root": "../data", "download": True, "transform": transforms.Compose([transforms.ToTensor()]), "hidden_size": 32, "layers": 4, "channels": 3, "max_pool": True, "embedding_size": 800, "margin": 1.0},
    # Following is for retrieval experiments: miniimagenet hidden size=64; so the embedding_size= 1600
    "MINIIMAGENET_64": {"data": TripletMiniImageNet, "root": "../data", "download": True, "transform": transforms.Compose([transforms.ToTensor()]), "hidden_size": 64, "layers": 4, "channels": 3, "max_pool": True, "embedding_size": 1600, "margin": 1.0},
    "OMNIGLOT": {"data": TripletOmniglot, "root": "../data", "download": True, "transform": transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))]), "hidden_size": 64, "layers": 4, "channels": 1, "max_pool": False, "embedding_size": 256, "margin": 1.0}
}

print(torch.__version__)
print(torch.cuda.is_available())


def accuracy(predictions, targets, shots):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    # return (predictions == targets).sum().float() / targets.size(0)

    if shots == 1:
        mask = np.array([True, False, False, False, False,
                        False, False, False, True, True, True, True])
    else:  # shots==5
        mask = np.array(
            [True, False, False, False, False, False, False, False, False, False, False, False,	False, False, False, False, False, False, False, False,
             True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True, False, False, False, False,
             True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
    return (predictions[mask] == targets[mask]).sum().float() / targets[mask].size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch
    adaptation_data[0] = adaptation_data[0].to(device)
    adaptation_data[1] = adaptation_data[1].to(device)
    adaptation_data[2] = adaptation_data[2].to(device)

    adaptation_labels = adaptation_labels.to(device)

    evaluation_data[0] = evaluation_data[0].to(device)
    evaluation_data[1] = evaluation_data[1].to(device)
    evaluation_data[2] = evaluation_data[2].to(device)

    evaluation_labels = evaluation_labels.to(device)

    # Adapt the model
    for step in range(adaptation_steps):
        out = learner(adaptation_data[0],
                      adaptation_data[1], adaptation_data[2])
        train_error = loss(out[0], out[1], out[2], torch.vstack([out[3], out[4], out[5]]), torch.hstack(
            [adaptation_labels[0], adaptation_labels[1], adaptation_labels[2]]))
        learner.adapt(train_error, allow_unused=True)

    # Evaluate the adapted model
    predictions = learner(
        evaluation_data[0], evaluation_data[1], evaluation_data[2])
    valid_error = loss(predictions[0], predictions[1], predictions[2], torch.vstack(
        [predictions[3], predictions[4], predictions[5]]), torch.hstack([evaluation_labels[0], evaluation_labels[1], evaluation_labels[2]]))
    valid_accuracy = accuracy(torch.vstack([predictions[3], predictions[4], predictions[5]]), torch.hstack(
        [evaluation_labels[0], evaluation_labels[1], evaluation_labels[2]]), shots)
    return valid_error, valid_accuracy


def fast_adapt_image_retrieval(batch, learner, loss, adaptation_steps, shots, ways, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch
    adaptation_data[0] = adaptation_data[0].to(device)
    adaptation_data[1] = adaptation_data[1].to(device)
    adaptation_data[2] = adaptation_data[2].to(device)

    adaptation_labels = adaptation_labels.to(device)

    # Adapt the model
    for step in range(adaptation_steps):
        out = learner(adaptation_data[0],
                      adaptation_data[1], adaptation_data[2])
        train_error = loss(out[0], out[1], out[2], torch.vstack([out[3], out[4], out[5]]), torch.hstack(
            [adaptation_labels[0], adaptation_labels[1], adaptation_labels[2]]))
        learner.adapt(train_error, allow_unused=True)

    return train_error, train_error


def precision_at_k(y_true, y_pred, k=12):
    """ Computes Precision at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Precision at k
    """
    intersection = np.intersect1d(y_true, y_pred[:k])
    return len(intersection) / k


def rel_at_k(y_true, y_pred, k=12):
    """ Computes Relevance at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Relevance at k
    """
    if y_pred[k-1] in y_true:
        return 1
    else:
        return 0


def average_precision_at_k(y_true, y_pred, k=12):
    """ Computes Average Precision at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    rel_counter = 0
    for i in range(1, k+1):
        ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)
        rel_counter += rel_at_k(y_true, y_pred, i)
    # return ap / min(k, len(y_true))
    if rel_counter == 0:
        return 0
    return ap / rel_counter


def mean_average_precision(y_true, y_pred, k=12):
    """ Computes MAP at k

    Parameters
    __________
    y_true: np.array
            2D Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            2D Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           MAP at k
    """

    return np.mean([average_precision_at_k(gt, pred, k)
                    for gt, pred in zip(y_true, y_pred)])


def mAP_at_k(D, imgLab, gt, rank=1, posonly="False"):

    _, idx = D.topk(rank[-1],  dim=1 )
    preds = np.array([imgLab[i].numpy() for i in idx])

    if posonly == "True":
        return mean_average_precision(gt[:10],preds[:10], k= rank[-1]),idx
    elif posonly == "divide_by_class" : 
        return [mean_average_precision(gt[:10],preds[:10], k= rank[-1]),mean_average_precision(gt[10:20],preds[10:20], k= rank[-1]),mean_average_precision(gt[20:30],preds[20:30], k= rank[-1]),mean_average_precision(gt[30:40],preds[30:40], k= rank[-1]),mean_average_precision(gt[40:50],preds[40:50], k= rank[-1]),mean_average_precision(gt,preds, k= rank[-1])],idx
    else :
        return mean_average_precision(gt, preds, k= rank[-1]), idx

def createSimilarityMatrix(Fvec):
    nbr_images = Fvec.shape[0]

    # mm: matrix multiplication. (n×m) mm (m×p) results in  (n×p) tensor.
    M = Fvec.mm(torch.t(Fvec))
    # [8041, 128] mm [128, 8041] --> [8041, 8041] this is D matrix
    # There are 1's along the diagonal!

    M[torch.eye(nbr_images).bool()] = -1
    # torch.eye: Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
    return M




def draw_support(batch,save_as):
        img_batch_vis = np.array((batch[0][0].numpy(),batch[0][1].numpy(),batch[0][2].numpy())).transpose(1,0,2,3,4)

        lbl_batch_vis = np.stack((batch[1][0].numpy(),batch[1][1].numpy(),batch[1][2].numpy())).transpose((1,0))


        # Create a figure with 4 rows and 3 columns
        fig, axs = plt.subplots(4, 3)
        fig.subplots_adjust(hspace=1, wspace=0.5)
        fig.set_facecolor('white')
        fig.suptitle("Support", fontsize=16)
        # Plot each element in the array as an image in the figure
        for i in range(4):
                for j in range(3):
                        axs[i,j].set_title(lbl_batch_vis[i,j])
                        axs[i,j].set_xticks([])
                        axs[i,j].set_yticks([])
                        axs[i,j].imshow(torch.from_numpy(img_batch_vis[i,j]).permute(1, 2, 0).cpu().int())
        fig.savefig(save_as)

def draw_query(batch,save_as,sample_per_class):
        Query = np.array([i.int().numpy() for i in batch[2]])
        Query = Query.reshape(5,sample_per_class,3,84,84)
        Query.shape

        query_label = np.array(batch[3])
        query_label = query_label.reshape(5,sample_per_class)

        # Create a figure with 5 rows and 10 columns
        fig, axs = plt.subplots(5, sample_per_class)
        fig.subplots_adjust(hspace=1, wspace=0.5)
        fig.set_facecolor('white')
        fig.suptitle("Query Pool", fontsize=16)
        # Plot each element in the array as an image in the figure
        for i in range(5):
                for j in range(sample_per_class):
                        axs[i,j].set_title(query_label[i,j])
                        axs[i,j].set_xticks([])
                        axs[i,j].set_yticks([])
                        axs[i,j].imshow(torch.from_numpy(Query[i,j]).permute(1, 2, 0).cpu().int())
        fig.savefig(save_as)

def draw_preds(batch,pred_ids,save_as,k):
        Query = np.array([i.int().numpy() for i in batch[2]])

        query_label = np.array(batch[3])
        
        fig, axs = plt.subplots(Query.shape[0], k+1, figsize=(25, 50))
        fig.subplots_adjust(hspace=1, wspace=0.5)
        fig.set_facecolor('white')
        fig.suptitle("Predictions", fontsize=16)
        for i in range(Query.shape[0]):
                for j in range(k+1):
                        if j == 0 :
                                axs[i,j].set_title(query_label[i])
                                axs[i,j].set_xticks([])
                                axs[i,j].set_yticks([])
                                axs[i,j].imshow(torch.from_numpy(Query[i]).permute(1, 2, 0).cpu().int())
                        else : 
                                axs[i,j].set_title(query_label[pred_ids[i][j-1]])
                                axs[i,j].set_xticks([])
                                axs[i,j].set_yticks([])
                                axs[i,j].imshow(torch.from_numpy(Query[pred_ids[i][j-1]]).permute(1, 2, 0).cpu().int())

        fig.savefig(save_as)

def generate_ground_truth(labels , k ):
    # 
    empty_array =    np.empty((len(labels),k),dtype= np.int32)
    for idx, label in enumerate(labels) : 
        empty_array[idx] = np.repeat(label,k)
    
    return empty_array


def main(
        ways=5,  # in our triplet implementation, number of distinct classes is 5
        shots=1, #retrieval shot is 1
        meta_lr=0.001,  # as in MAML
        fast_lr=0.01,
        meta_batch_size=4,  # Maml miniImageNet: 2 (5-shot); 4 (1-shot)
        adaptation_steps= 5,  # Maml Omniglot:1; miniImageNet: 5
        test_adaptation_steps=10,  # Maml Omniglot:3 ; miniImageNet: 10
        num_iterations=60000,  # as in MAML
        cuda=True,
        seed=42,
        num_test_episodes=600,
        selected_model="MINIIMAGENET_64"
):

    PATH= "Tripletmaml_MINIIMAGENET_64_batchsize4_shots1_with_optimizer.pt"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')

    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    triplet_imagenet_dataset = Triplet_Model_Parameter[selected_model]["data"](
        root=Triplet_Model_Parameter[selected_model]["root"], download=Triplet_Model_Parameter[selected_model]["download"], transform=Triplet_Model_Parameter[selected_model]["transform"])

    # Create model using saved parameters:
    model = TripletCNN4(output_size= ways, hidden_size=Triplet_Model_Parameter[selected_model]["hidden_size"], layers=Triplet_Model_Parameter[selected_model]["layers"], channels=Triplet_Model_Parameter[selected_model]["channels"], max_pool=Triplet_Model_Parameter[selected_model]["max_pool"], embedding_size=Triplet_Model_Parameter[selected_model]["embedding_size"])
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=True)
    opt = optim.Adam(maml.parameters(), meta_lr) # meta-update
 
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    combined_loss_fn = CombinedLoss(shots, lamda=1)  # lamda: metric loss weight

    nbr_samples_per_class= 10
    nbr_iterations= 600
    for k in [9]:
        mean_map = []
        with open("Tripletmaml_MINIIMAGENET_64_batchsize4_shots1_with_optimizer"+"_map9.txt", "w") as file:
            for i in range(nbr_iterations):
                print("iteration" ,i)
                a = triplet_imagenet_dataset.sample(
                    "test", mode="img_retrieval", samples_per_class= nbr_samples_per_class)
                opt.zero_grad()  # for each batch, gradients should be cleaned.
                # Compute meta-training loss
                # print('Task no: ', task)
                learner = maml.clone()
                batch = a

                #visualization:
                #draw_support(batch,"./visualization/"+"64_imagenet_all_10iter_visuzalization_labelization_sup_MAP"+str(k)+"_"+str(i)+".png")
                #draw_query(batch,"./visualization/"+"64_imagenet_all_10iter_visuzalization_labelization_query_MAP"+str(k)+"_"+str(i)+".png",sample_per_class= nbr_samples_per_class)
            
                train_error, _ = fast_adapt_image_retrieval(batch,
                                                            learner,
                                                            combined_loss_fn,
                                                            adaptation_steps,
                                                            shots, # retrieval task always 1 shot
                                                            ways,
                                                            device)
                embeddings = []
                labels = torch.from_numpy(np.array(list(a[3])))
                for idx, image in enumerate(a[2]):
                    embedding, class_prob = learner.forward_once(torch.unsqueeze(image, 0).to(device))
                    # data: tensor([[0.0951, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]] ilk eleman sıfırdan farklı sadece gibi garip değil mi?
                    embedding = F.normalize(embedding, p=2, dim=1).cpu()
                    embeddings.append(np.array(embedding[0].tolist()))
                embeddings = torch.from_numpy(np.array(embeddings))

                SimMatrix = createSimilarityMatrix(embeddings)

                map,pred_ids = mAP_at_k(SimMatrix,labels,generate_ground_truth(labels,k),[k],posonly="False")
                #draw_preds(batch,pred_ids,"./visualization/"+"64_imagenet_all_10iter_visualization_labelization_preds_MAP"+str(k)+"_"+str(i)+".png",k)
                file.write("iteration - " + str(i) + " mAP@"+str(k)+" :" + str(map) + "\n" )
                mean_map.append(map)
            print(np.mean(mean_map))
            file.write("General Mean - " + str(i) + " mAP@" +
                       str(k)+"  :" + str(np.mean(mean_map)) + "\n")


if __name__ == '__main__':
    # data Parameters can be :
    # CIFARFS
    # CUB
    # FLOWERS
    # MINIIMAGENET
    # OMNIGLOT
    main(selected_model="CUB")
