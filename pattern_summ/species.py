import random
import collections
import itertools
import multiprocessing
import numpy as np

from .util import get_random_seed_generator


class Species:
    def __init__(self, random_seed_generator=None, representative=None, prob_mutate=0.5, n_pools=None):
        self.n_pools = n_pools or max(1, multiprocessing.cpu_count() - 1)
        self.representative = representative
        self.prob_mutate = prob_mutate
        self.random_seed_generator = random_seed_generator or get_random_seed_generator(1, 99999)

        self.population = []
        self._fitness = []

        if self.representative:
            self.add(self.representative)

    def get_distance(self, pattern):
        return self.representative - pattern

    def eliminate_to(self, n_survivors):
        assert self._fitness is not None

        if type(self._fitness) is not np.array:
            self._fitness = np.array(self._fitness)

        ranking = (-self._fitness).argsort().argsort()  # type: np.ndarray
        self.population = list(itertools.compress(self.population, ranking < n_survivors))

    @staticmethod
    def _reproduce(self_population, couples, prob_mutate, random_seed):
        np.random.seed(random_seed)
        random.seed(random_seed)

        population = []
        for i, j in couples:
            a, b = self_population[i], self_population[j]
            offspring = a.crossover(b).copy()

            if not len(offspring):
                continue

            if np.random.random() < prob_mutate:
                offspring.mutate()

            if len(offspring):
                population.append(offspring)

        return population

    def reproduce(self, population_size):
        n_couples = int(population_size) - len(self)
        if n_couples == 0:
            return []

        couples = np.random.randint(0, len(self), (n_couples, 2))

        couples_set = []
        size = int(np.ceil(len(couples) / self.n_pools))
        for i in range(0, len(couples), size):
            couples_set.append(couples[i: i + size])

        pool = multiprocessing.Pool(self.n_pools)
        data_generator = zip(itertools.repeat(self.population),
                             couples_set, itertools.repeat(self.prob_mutate),
                             self.random_seed_generator)
        population = sum(pool.starmap(self._reproduce, data_generator), [])
        pool.close()

        self.population += population
        # self.population = list(set(self.population))
        return self.population

    def add(self, pattern):
        self.population.append(pattern)
        if type(self._fitness) is not list:
            self._fitness = list(self._fitness)
        self._fitness.append(pattern.fitness)

    @property
    def convergence(self):
        return collections.Counter(self.population).most_common()[0][1] / len(self)

    @property
    def fitness(self):
        if type(self._fitness) is not np.array:
            self._fitness = np.array(self._fitness)
        return self._fitness.mean()

    def __len__(self):
        return len(self.population)

    def __lt__(self, other):
        return self.fitness < other.fitness
