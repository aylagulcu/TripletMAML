import os
import copy
import torch
import random
import pickle
import numpy as np
from model import *
from torch.utils.data import DataLoader

class DE():
    
    def __init__(self, pop_size = None, 
                 mutation_factor = None, 
                 crossover_prob = None, 
                 boundary_fix_type = 'random', 
                 seed = None,
                 mutation_strategy = 'rand1',
                 crossover_strategy = 'bin',
                 dataset = None):

        # DE related variables
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.mutation_strategy = mutation_strategy
        self.crossover_strategy = crossover_strategy
        self.boundary_fix_type = boundary_fix_type

        # Global trackers
        self.population = []
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
        self.population = []
        self.population = []
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

    # Initialize population
    def init_population(self, pop_size = None):
        i = 0
        while i < pop_size:
            chromosome = self.init_pop_rnd.uniform(low=0.0, high=1.0, size=self.DIMENSIONS)
            config = self.vector_to_config(chromosome)
            model = Model(chromosome, config)
            # Same Solution Check
            isSame, _ = self.checkSolution(model)
            if not isSame:
                model.solNo = self.solNo
                self.solNo += 1
                self.population.append(model)
                self.allModels[model.solNo] = copy.deepcopy(model)
                self.writePickle(model, model.solNo)
                i += 1
        
        return np.array(self.population)
            
    
    def sample_population(self, size = None):
        '''Samples 'size' individuals'''

        selection = self.sample_pop_rnd.choice(np.arange(len(self.population)), size, replace=False)
        return self.population[selection]
    
    def boundary_check(self, vector):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        projection == The invalid value is truncated to the nearest limit
        random == The invalid value is repaired by computing a random number between its established limits
        reflection == The invalid value by computing the scaled difference of the exceeded bound multiplied by two minus

        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        
        if self.boundary_fix_type == 'projection':
            vector = np.clip(vector, 0.0, 1.0)
        elif self.boundary_fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        elif self.boundary_fix_type == 'reflection':
            vector[violations] = [0 - v if v < 0 else 2 - v if v > 1 else v for v in vector[violations]]

        return vector

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

    def init_eval_pop(self):
        '''
            Creates new population of 'pop_size' and evaluates individuals.
        '''
        print("Start Initialization...")
        self.population = self.init_population(self.pop_size)
        self.best_arch = self.population[0]

        for i in range(self.pop_size):
            model = self.population[i]
            model.fitness = self.f_objective(model)
            self.writePickle(model, model.solNo)
            
            if model.fitness >= self.best_arch.fitness:
                self.best_arch = model

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation(self, current=None, best=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand1(r1.chromosome, r2.chromosome, r3.chromosome)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5)
            mutant = self.mutation_rand2(r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome, r5.chromosome)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_rand1(best, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4)
            mutant = self.mutation_rand2(best, r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_currenttobest1(current, best.chromosome, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_currenttobest1(r1.chromosome, best.chromosome, r2.chromosome, r3.chromosome)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.DIMENSIONS) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.DIMENSIONS)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''
            Performs the exponential crossover of DE
        '''
        n = self.crossover_rnd.randint(0, self.DIMENSIONS)
        L = 0
        while ((self.crossover_rnd.rand() < self.crossover_prob) and L < self.DIMENSIONS):
            idx = (n+L) % self.DIMENSIONS
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''
            Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring
    
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


    def evolve_generation(self):
        '''
            Performs a complete DE evolution: mutation -> crossover -> selection
        '''
        trials = []
        Pnext = [] # Next population

        generationBest = max(self.population, key=lambda x: x.fitness)

        # mutation -> crossover
        for j in range(self.pop_size):
            target = self.population[j].chromosome
            mutant = copy.deepcopy(target)
            mutant = self.mutation(current=target, best=generationBest)
            trial = self.crossover(target, mutant)
            trial = self.boundary_check(trial)
            config = self.vector_to_config(trial)
            model = Model(trial, config)
            self.solNo += 1
            model.solNo = self.solNo
            trials.append(model)
        
        trials = np.array(trials)

        # selection
        for j in range(self.pop_size):
            target = self.population[j]
            mutant = trials[j]

            isSameSolution, sol = self.checkSolution(mutant)
            if isSameSolution:
                print("SAME SOLUTION")
                mutant = sol
            else:
                self.f_objective(mutant)
                self.writePickle(mutant, mutant.solNo)
                self.allModels[mutant.solNo] = copy.deepcopy(mutant)

            # Check Termination Condition
            if self.totalTrainedModel > self.MAX_SOL: 
                return
            #######

            if mutant.fitness >= target.fitness:
                Pnext.append(mutant)
                # Best Solution Check
                if mutant.fitness >= self.best_arch.fitness:
                    self.best_arch = mutant
            else:
                Pnext.append(target)

        self.population = np.array(Pnext)

    def run(self, seed):
        self.seed = seed
        self.solNo = 0
        self.generation = 0
        self.totalTrainedModel = 0
        print(self.mutation_strategy)
        self.reset()
        self.init_eval_pop()

        while self.totalTrainedModel < self.MAX_SOL:
            self.evolve_generation()
            print(f"Generation:{self.generation}, Best: {self.best_arch.fitness}, {self.best_arch.solNo}")
            self.generation += 1     
        

if __name__ == "__main__":
    device = torch.device('cuda')
	

    de = DE(pop_size=20, mutation_factor=0.5, crossover_prob=0.5, seed=42, dataset="CIFARFS")
    de.run(42)
