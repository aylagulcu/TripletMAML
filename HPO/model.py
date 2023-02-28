from cmath import nan
import copy
import torch
import torch.nn as nn
import numpy as np
from losses import *
from train import *

# TRIPLET-MAML için yazıldı

class Model(nn.Module):

    def __init__(self, chromosome = None, config = None):
        super(Model, self).__init__()

        self.solNo = None
        self.fitness = 0
        self.config = config
        self.chromosome = chromosome

    def evaluate(self, dataset):

        try:
            print(f"Model {self.solNo} Training...")
            self.fitness = main(self.solNo, selected_model=dataset, **self.config)

        except Exception as e:
            print(e)
            torch.cuda.empty_cache() # Clear Memory
            return -1

        
        print("Model Fitness:", self.fitness)
        torch.cuda.empty_cache() # Clear Memory

        return self.fitness