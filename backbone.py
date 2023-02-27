#!/usr/bin/env python3

import torch
import learn2learn as l2l
from torch.nn.utils.weight_norm import WeightNorm

from scipy.stats import truncnorm

def truncated_normal_(tensor, mean=0.0, std=1.0):
    # PT doesn't have truncated normal.
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor


def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias.data, 0.0)
    return module


def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class LinearBlock(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearBlock, self).__init__()
        self.relu = torch.nn.ReLU()
        self.normalize = torch.nn.BatchNorm1d(
            output_size,
            affine=True,
            momentum=0.999,
            eps=1e-3,
            track_running_stats=False,
        )
        self.linear = torch.nn.Linear(input_size, output_size)
        fc_init_(self.linear)

    def forward(self, x):
        x = self.linear(x)
        x = self.normalize(x)
        x = self.relu(x)
        return x


class ConvBlock(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_pool=True,
                 max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=False,
            )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = torch.nn.BatchNorm2d(
            out_channels,
            affine=True,
            # eps=1e-3,
            # momentum=0.999,
            # track_running_stats=False,
        )
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU() 

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=True,
        )
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(torch.nn.Sequential):

    # NOTE:
    #     Omniglot: hidden=64, channels=1, no max_pool
    #     MiniImagenet: hidden=32, channels=3, max_pool

    def __init__(self,
                 hidden=64,
                 channels=1,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0):
        core = [ConvBlock(channels,
                          hidden,
                          (3, 3),
                          max_pool=max_pool,
                          max_pool_factor=max_pool_factor),
                ]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden,
                                  hidden,
                                  kernel_size=(3, 3),
                                  max_pool=max_pool,
                                  max_pool_factor=max_pool_factor))
        super(ConvBase, self).__init__(*core)


class OmniglotFC(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The fully-connected network used for Omniglot experiments, as described in Santoro et al, 2016.

    **References**

    1. Santoro et al. 2016. “Meta-Learning with Memory-Augmented Neural Networks.” ICML.

    **Arguments**

    * **input_size** (int) - The dimensionality of the input.
    * **output_size** (int) - The dimensionality of the output.
    * **sizes** (list, *optional*, default=None) - A list of hidden layer sizes.

    **Example**
    ~~~python
    net = OmniglotFC(input_size=28**2,
                     output_size=10,
                     sizes=[64, 64, 64])
    ~~~
    """

    def __init__(self, input_size, output_size, sizes=None):
        super(OmniglotFC, self).__init__()
        if sizes is None:
            sizes = [256, 128, 64, 64]
        layers = [LinearBlock(input_size, sizes[0]), ]
        for s_i, s_o in zip(sizes[:-1], sizes[1:]):
            layers.append(LinearBlock(s_i, s_o))
        layers = torch.nn.Sequential(*layers)
        self.features = torch.nn.Sequential(
            l2l.nn.Flatten(),
            layers,
        )
        self.classifier = fc_init_(torch.nn.Linear(sizes[-1], output_size))
        self.input_size = input_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class OmniglotCNN(torch.nn.Module):
    """

    [Source](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The convolutional network commonly used for Omniglot, as described by Finn et al, 2017.

    This network assumes inputs of shapes (1, 28, 28).

    **References**

    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.” ICML.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = OmniglotCNN(output_size=20, hidden_size=128, layers=3)
    ~~~

    """

    def __init__(self, output_size=5, hidden_size=64, layers=4):
        super(OmniglotCNN, self).__init__()
        self.hidden_size = hidden_size
        self.base = ConvBase(
             hidden=hidden_size,
             channels=1,
             max_pool=False,
             layers=layers,
        )
        self.features = torch.nn.Sequential(
            l2l.nn.Lambda(lambda x: x.view(-1, 1, 28, 28)),
            self.base,
            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
            l2l.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNN4Backbone(ConvBase):

    def __init__(
        self,
        hidden_size=64,
        layers=4,
        channels=3,
        max_pool=False,
        max_pool_factor=1.0,
    ):
        super(CNN4Backbone, self).__init__(
            hidden=hidden_size,
            layers=layers,
            channels=channels,
            max_pool=max_pool,
            max_pool_factor=max_pool_factor,
        )

    def forward(self, x):
        x = super(CNN4Backbone, self).forward(x) # 31 dec 2022: x.shape[4,64, 5,5]
        x = x.reshape(x.size(0), -1)
        return x


class CNN4(torch.nn.Module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.

    This network assumes inputs of shapes (3, 84, 84).

    Instantiate `CNN4Backbone` if you only need the feature extractor.

    **References**

    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=32) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = CNN4(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(
        self,
        output_size,
        hidden_size=64,
        layers=4,
        channels=3,
        max_pool=True,
        embedding_size=None,
    ):
        super(CNN4, self).__init__()
        if embedding_size is None:
            embedding_size = 25 * hidden_size
        self.features = CNN4Backbone(
            hidden_size=hidden_size,
            channels=channels,
            max_pool=max_pool,
            layers=layers,
            max_pool_factor=1 // layers,
        )
        self.classifier = torch.nn.Linear(
            embedding_size,
            output_size,
            bias=True,
        )
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# MiniImagenetCNN = CNN4



######### TRIPLET #########

class TripletCNN4(torch.nn.Module):

    def __init__(
        self,
        output_size,
        hidden_size=32,
        layers=4,
        channels=1,
        max_pool=False,
        embedding_size= 256 
    ):
        super(TripletCNN4, self).__init__()
        self.features = CNN4Backbone(
            hidden_size=hidden_size,
            channels=channels,
            max_pool=max_pool,
            layers=layers,
            max_pool_factor=1
        )
        self.classifier = torch.nn.Linear(
            embedding_size,
            output_size,
            bias=True,
        )
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward_once(self, x1):
        x1 = self.features(x1) # Omniglot input [1, 28, 28])
        x1_clsprob = self.classifier(x1) 
        
        return x1,x1_clsprob # embeddings and class probabilities

    def forward(self, x1,x2,x3):
        x1 = self.features(x1) # Omniglot input [1, 28, 28])
        x1_clsprob = self.classifier(x1) 
        x2 = self.features(x2)
        x2_clsprob = self.classifier(x2)
        x3 = self.features(x3)
        x3_clsprob = self.classifier(x3)
        return x1,x2,x3, x1_clsprob, x2_clsprob, x3_clsprob # embeddings and class probabilities

        



class TripletCNN4_BaselinePP(torch.nn.Module):

    def __init__(
        self,
        output_size,
        hidden_size=32,
        layers=4,
        channels=1,
        max_pool=False,
        embedding_size= 256 
    ):
        super(TripletCNN4_BaselinePP, self).__init__()
        self.features = CNN4Backbone(
            hidden_size=hidden_size,
            channels=channels,
            max_pool=max_pool,
            layers=layers,
            max_pool_factor=1
        )
        self.classifier = torch.nn.Linear(
            embedding_size,
            output_size,
            bias=True,
        )
        if output_size <=200:
            self.scale_factor = 2; 
            #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; 
            #in omniglot, a larger scale factor is required to handle >1000 output classes.
        maml_init_(self.classifier)
        self.hidden_size = hidden_size


    def forward(self, x1,x2,x3):
        self.classifier.weight.data = self.classifier.weight.data.div(torch.norm(self.classifier.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.classifier.weight.data) + 0.00001)

        x1 = self.features(x1) # Omniglot input [1, 28, 28])
        x1_normalized= x1.div( torch.norm(x1, p=2, dim =1).unsqueeze(1).expand_as(x1) + 0.00001) # now, the norm of each x is 1
        cos_dist = self.classifier(x1_normalized) 
        x1_clsprob = self.scale_factor* (cos_dist) 

        x2 = self.features(x2) # Omniglot input [1, 28, 28])
        x2_normalized= x2.div( torch.norm(x2, p=2, dim =1).unsqueeze(1).expand_as(x2) + 0.00001)
        cos_dist = self.classifier(x2_normalized) 
        x2_clsprob = self.scale_factor* (cos_dist) 

        x3 = self.features(x3) # Omniglot input [1, 28, 28])
        x3_normalized= x3.div( torch.norm(x3, p=2, dim =1).unsqueeze(1).expand_as(x3) + 0.00001)
        cos_dist = self.classifier(x3_normalized) 
        x3_clsprob = self.scale_factor* (cos_dist) 

        return x1,x2,x3, x1_clsprob, x2_clsprob, x3_clsprob # embeddings and class probabilities

