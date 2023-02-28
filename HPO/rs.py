import os
import copy
import torch
import random
import pickle
import numpy as np
from model import *
from torch.utils.data import DataLoader

class RS():
    
    def __init__(self,
                 seed = None,
                 dataset = None):



        # Global trackers
        self.history = []
        self.allModels = dict()
        self.best_arch = None
        self.seed = seed
        self.dataset = dataset

        # CONSTANTS
        self.MAX_SOL = 200
        self.DIMENSIONS = 7
        self.hyperparameters = {"number_of_filters": [32, 64, 128],
                                "meta_batch_size": [2, 3, 4, 8, 16],
                                "fast_lr": [0.01, 0.1, 0.4, 0.5],
                                "adaptation_steps": [1, 5, 10],
                                "test_adaptation_steps": [5, 10],
                                "meta_lr": [1e-4, 1e-3, 1e-2],
                                "optimizer": ["Adam", "SGD"]}
    
    def reset(self):
        self.best_arch = None
        self.allModels = dict()
        self.history = []
        self.best_arch = None
        self.init_rnd_nbr_generators()
    
    def init_rnd_nbr_generators(self):
        # Random Number Generators
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.init_pop_rnd = np.random.RandomState(self.seed)
    
    def writePickle(self, data, name):
        # Write History
        with open(f"results/model_{name}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)

    def get_param_value(self, value, step_size):
        ranges = np.arange(start=0, stop=1, step=1/step_size)
        return np.where((value < ranges) == False)[0][-1]

    def vector_to_config(self, vector):
        '''Converts numpy array to discrete values'''

        try:
            #config = np.zeros(self.DIMENSIONS, dtype='uint8')
            config = dict()

            idx = 0
            for key, params in self.hyperparameters.items():
                value = self.get_param_value(vector[idx], len(params))
                config[key] = params[value]
                idx += 1
        except:
            print("HATA...", vector)

        return config

    def f_objective(self, model):

        # Else  
        fitness = model.evaluate(self.dataset)
        if fitness != -1:
            self.totalTrainedModel += 1
        return fitness
    
    def readPickleFile(self, file):
        with open(f"results/model_{file}.pkl", "rb") as f:
            data = pickle.load(f)
        
        return data

    def checkSolution(self, model):
        for i in self.allModels.keys():
            model_2 = self.allModels[i]
            if model.config == model_2.config:
                return True, model_2
        
        return False, None 


    def random_search(self):

        i = 0
        self.reset()
        self.solNo = 0
        self.solutions = []
        self.totalTrainedModel = 0
        print(self.dataset)
        while i < self.MAX_SOL:
            chromosome = self.init_pop_rnd.uniform(low=0.0, high=1.0, size=self.DIMENSIONS)
            config = self.vector_to_config(chromosome)
            model = Model(chromosome, config)
            # Same Solution Check
            isSame, _ = self.checkSolution(model)
            if not isSame:
                model.solNo = self.solNo
                self.solNo += 1
                
                self.solutions.append(model)
                self.allModels[model.solNo] = copy.deepcopy(model)
                self.f_objective(model)
                self.writePickle(model, model.solNo)
                
                if i == 0:
                    self.best_arch = model
                if model.fitness >= self.best_arch.fitness:
                    self.best_arch = model

                i += 1

        bestSol = max(self.solutions, key=lambda x: x.fitness)
        print(f"Best Sol No: {bestSol.solNo}, Best Fitness: {bestSol.fitness}")


if __name__ == "__main__":
    device = torch.device('cuda')
	

    rs = RS(seed=42, dataset="MINIIMAGENET")
    rs.random_search()
