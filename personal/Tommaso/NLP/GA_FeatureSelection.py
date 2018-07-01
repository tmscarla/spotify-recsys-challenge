#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random

from deap import base
from deap import creator
from deap import tools

import time
import pandas as pd
import numpy as np
import random

from utils.evaluator import Evaluator
from utils.datareader import Datareader
from utils.post_processing import eurm_to_recommendation_list
from recommenders.similarity.tversky import tversky_similarity
from recommenders.similarity.dot_product import dot_product
from sklearn.utils.sparsefuncs import inplace_csr_column_scale



def randomMutationCustom(individual, indpb):
    """Flip the value of the attributes of the input individual and return the
    mutant. The *individual* is expected to be a :term:`sequence` and the values of the
    attributes shall stay valid after the ``not`` operator is called on them.
    The *indpb* argument is the probability of each attribute to be
    flipped. This mutation is usually applied on boolean individuals.

    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be flipped.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.random()

    return individual,


class GA_FeatureSelection():

    def __init__(self, UCM, URM_train, test_playlists_indices, logFile, bestIndividualFile, mode="selection",
                 numGenerations=30, populationSize=30, initialRandomDistribution=np.random.uniform(0, 1),
                 verbose=True):

        self.UCM = UCM
        self.URM_train = URM_train
        self.test_playlists_indices = test_playlists_indices.astype(np.int)
        self.logFile = open(logFile, "a")
        self.bestIndividualFile = open(bestIndividualFile, "a")
        self.initialRandomDistribution = initialRandomDistribution
        self.verbose = verbose
        self.top = 0

        self.current = 0

        self.evaluator = Evaluator(Datareader(mode='offline', only_load=True, verbose=False))

        self.NUM_VARIABLES = UCM.shape[1]

        if (mode == "weighting" or mode == "selection"):
                self.mode = mode

        # Crossover probability
        self.CXPB = 0.5

        # Mutation probability
        self.MUTPB = 0.2

        # Number of generations for which the evolution runs
        self.NGEN = numGenerations

        self.POPULATION_SIZE = populationSize


    def writeOnLogFile(self, stringToLog):
        self.logFile.write(stringToLog + "\n")
        self.logFile.flush()

    def writeOnBestIndividualFile(self, stringToLog):
        self.bestIndividualFile.write(stringToLog + "\n")
        self.bestIndividualFile.flush()

    # Set the max number of features
    def isIndividualAccettable(self, individual):
        return np.sum(np.array(individual)) < 10000

    def fitnessFunction(self, individual):

        # Convert list into a numpy array
        individual = np.array(individual)

        # Make a copy of the UCM and filter it for each column
        if self.verbose:
            print('Filtering UCM...')
        start = time.time()
        UCM_filtered = self.UCM.copy()
        UCM_filtered = UCM_filtered.astype(np.float64)
        inplace_csr_column_scale(UCM_filtered, individual)
        if self.verbose:
            print('UCM filtered in', time.time() - start, 'sec')

        # Compute similarity
        if self.verbose:
            print('Computing similarity...')
        start = time.time()
        similarity = tversky_similarity(UCM_filtered, shrink=200, alpha=0.1,
                                        beta=1, target_items=self.test_playlists_indices,
                                        binary=False)
        similarity = similarity.tocsr()
        if self.verbose:
            print('Similarity computed in', time.time() - start, 'sec')

        # Compute eurm
        if self.verbose:
            print('Computing eurm...')
        start = time.time()
        eurm = dot_product(similarity, self.URM_train, k=500)
        if self.verbose:
            print('eurm computed in', time.time() - start, 'sec')
            print('Converting eurm in csr...')
        start = time.time()
        eurm = eurm.tocsr()
        eurm = eurm[self.test_playlists_indices, :]
        if self.verbose:
            print('eurm converted in', time.time() - start, 'sec')

        # Evaluate
        rec_list = eurm_to_recommendation_list(eurm)
        print('current', self.current)

        score_cat_1 = self.evaluator.evaluate_single_metric(rec_list, name='Genetic', metric='prec',
                                                            level='track', cat=1, verbose=False)
        score_cat_2 = self.evaluator.evaluate_single_metric(rec_list, name='Genetic', metric='prec',
                                                            level='track', cat=2, verbose=False)
        score = (score_cat_1 + score_cat_2) / 2

        self.current += 1

        if self.verbose:
            print(score)

        print("Numfeatures {}".format(np.sum(individual)))
        print('\n')

        return score,

    def setupParameters(self):

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Attribute generator
        # define 'attr_bool' to be an attribute ('gene')
        # which corresponds to integers sampled uniformly
        # from the range [0,1] (i.e. 0 or 1 with equal
        # probability)

        # Structure initializers
        # define 'individual' to be an individual
        # consisting of 100 'attr_bool' elements ('genes')

        if (self.mode == "weighting"):
            self.toolbox.register("attr_float", self.initialRandomDistribution)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_float, self.NUM_VARIABLES)

        elif (self.mode == "selection"):
            # self.toolbox.register("attr_bool", random.randint, 0, 1)
            self.toolbox.register("attr_bool", self.initialRandomDistribution)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_bool, self.NUM_VARIABLES)

        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # ----------
        # Operator registration
        # ----------
        # register the goal / fitness function
        self.toolbox.register("evaluate", self.fitnessFunction)
        # self.toolbox.decorate("evaluate", tools.DeltaPenality(self.isIndividualAccettable, -1.0))

        # register the crossover operator
        self.toolbox.register("mate", tools.cxTwoPoint)

        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        if self.mode == "weighting":
            self.toolbox.register("mutate", randomMutationCustom, indpb=0.05)

        elif self.mode == "selection":
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        # self.toolbox.register("select", tools.selRandom)


    def main(self):

        self.start_time = time.time()

        random.seed(64)

        self.setupParameters()

        self.writeOnLogFile(time.strftime("%Y-%m-%d %H:%M") + "\n")
        self.writeOnBestIndividualFile(time.strftime("%Y-%m-%d %H:%M") + "\n")

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = self.toolbox.population(n=self.POPULATION_SIZE)

        print("Start of evolution")

        self.current = 0

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop))

        # Begin the evolution
        for g in range(self.NGEN):
            print("-- Generation %i --" % g)

            self.writeOnLogFile("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < self.CXPB:
                    self.toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < self.MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            # Update top value
            if max(fits) > self.top:
                self.top = max(fits)

                # Write on log file
                best_ind = tools.selBest(pop, 1)[0]
                self.writeOnBestIndividualFile('GEN ' + str(g) + ' | ' + str(self.top))
                self.writeOnBestIndividualFile("%s" % best_ind + '\n')

            print("  Top %s" % self.top)
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

            self.writeOnLogFile("  Top %s\n" % self.top +
                                "  Min %s\n" % min(fits) +
                                "  Max %s\n" % max(fits) +
                                "  Avg %s\n" % mean +
                                "  Std %s\n" % std)

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values[0]))

        print("Elapsed time" + str(time.time()-self.start_time))
