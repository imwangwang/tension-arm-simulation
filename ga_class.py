# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:57:41 2017

@author: Rohit
"""

import random
import numpy as np
from deap import tools
from deap import base, creator

class GenticAlgorithm(object):
    
    def __init__(self,param=None):
        self.__TOTAL_POP = 16
        self.__SIBLINGS_PER_COUPLE = 1 #total children = CHILD_PER_COUPLE*2
        self.__BEST = 2
        self.__TOLERANCE = 1
        self.__LOW = [-100,-100]
        self.__HIGH = [100,100]
        self.__TOTAL_GENERATIONS = 5000
        self.__R = np.array([[-2,0],[0,-1.5]])
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.__toolbox = base.Toolbox()
        
        self.__toolbox.register("attribute_0", random.randint, self.__LOW[0], self.__HIGH[0])
        self.__toolbox.register("attribute_1", random.randint, self.__LOW[1], self.__HIGH[1])
        self.__toolbox.register("individual", tools.initCycle, creator.Individual,
                         (self.__toolbox.attribute_0,self.__toolbox.attribute_1,self.__toolbox.attribute_2), n=1)
        self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.individual)
        self.__toolbox.register("mate", self.__mate)
        self.__toolbox.register("mutate",self.__mutate,k=3)
        self.__toolbox.register("select", tools.selBest, k=2)
        self.__toolbox.register("evaluate", self.__evaluate)
        self.__toolbox.register("add_new_ind",self.__add_new_ind)

        
    def constrained_individual(self):
        individual = []
        for i in range(2):
            individual.append(random.randint(self.__LOW[i],self.__HIGH[i]))
        return individual


    def __evaluate(self,individual,true_angles):
        pred_angles = np.dot(self.__R,np.array(individual))
        loss = np.linalg.norm(true_angles-pred_angles)
        return loss,0

    
    def __scaling(self,individual,direction='shrink'):
        for i in range(len(individual)):
            if direction == 'shrink':
                scaled_val = int((individual[i]-self.__LOW[i])*255//(self.__HIGH[i]-self.__LOW[i]))
            else:
                scaled_val = (individual[i]/255)*(self.__HIGH[i]-self.__LOW[i]) + self.__LOW[i]
            individual[i] = scaled_val
        return 
    
    def __bit_crossover(self,ind1,ind2):
        sibling1 = self.__toolbox.clone(ind1)
        sibling2 = self.__toolbox.clone(ind2)
        
        self.__scaling(sibling1,direction='shrink')
        self.__scaling(sibling2,direction='shrink')
        
        swap_index = random.randint(1,7)
        mask_1 = (0x01 << swap_index) - 1
        mask_2 = 0xFF - mask_1
        for i in range(len(ind1)):
            sibling1[i] = (sibling1[i] & mask_1) | (sibling2[i] & mask_2)
            sibling2[i] = (sibling1[i] & mask_2) | (sibling2[i] & mask_1)
        self.__scaling(sibling1,direction='restore')
        self.__scaling(sibling2,direction='restore')
        
        return sibling1, sibling2
        
    def __mate(self,population):
        children = []
        for ind1, ind2 in zip(population[::2],population[1::2]):
            for i in range(self.__SIBLINGS_PER_COUPLE):
                child1, child2 = self.__bit_crossover(ind1, ind2)    
                children.append(child1)
                children.append(child2)
        for child in children:
            del child.fitness.values
        return population + children
    
    def __add_new_ind(self,population):
        new_ind = self.__toolbox.population(n=self.__TOTAL_POP-len(population))
        for ind in new_ind:
            del ind.fitness.values
        return population+new_ind
    
    def __mutate(self,population,k):
        mutations = self.__toolbox.clone(population[:k])
        for ind in mutations:
            self.__scaling(ind,direction='shrink')
            for i in range(len(ind)):
                rand_element = random.randint(0,255)
                ind[i] = ind[i]^rand_element
            self.__scaling(ind,direction='expand')
            del ind.fitness.values
        return population+mutations

    
    def run(self,prev_best = None, true_angles=[np.pi, 2*np.pi]):
        mean_error = []
        std = []
        min_error = []
        top_candidate = []
        
        pop = self.__toolbox.population(n=self.__TOTAL_POP)
        if (prev_best is not None):
            for i in range(len(pop[0])):
                pop[0][i] = prev_best[i]
        fitnesses = [self.__toolbox.evaluate(ind,true_angles) for ind in pop]
    #    fitnesses = [toolbox.evaluate(ind) for ind in pop]
        for ind, fit_value in zip(pop,fitnesses):
            ind.fitness.values = fit_value
        n_gen = 0
        while n_gen < self.__TOTAL_GENERATIONS:
            n_gen += 1
            offspring = self.__toolbox.select(pop,k=self.__BEST)
            offspring = list(map(self.__toolbox.clone,offspring))
            
            offspring = self.__toolbox.mate(offspring)
            offspring = self.__toolbox.mutate(offspring,k=self.__BEST)
            offspring = self.__toolbox.add_new_ind(offspring)
            
            for ind in offspring:
                if ind.fitness.valid == False:
                    ind.fitness.values = self.__toolbox.evaluate(ind,true_angles)
            
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]
            mean_error.append(np.mean(fits))
            std.append(np.std(fits))
            
            top_candidate = self.__toolbox.select(pop,k=1)[0]
            min_fit = top_candidate.fitness.values[0]
            min_error.append(min_fit)
            if min_fit <= self.__TOLERANCE:
                break
            
        return top_candidate