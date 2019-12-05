import random
import functools
import itertools
import multiprocessing
import gc
import numpy as np
import pandas as pd

from .pattern import Pattern, WordSequence
from .data_manager import DataManager
from .species import Species

from .util import get_random_seed_generator


class PatternEvolution:
    def __init__(self, data, population_size, n_survivors,
                 prob_mutate=0.9, join_thresh=0.6, alpha=2, beta=1, n_pools=None, random_seed=None):

        random.seed(random_seed)
        np.random.seed(random_seed)
        self.random_seed_generator = get_random_seed_generator(1, 99999, random_seed)

        self.data_manager = DataManager(pd.Series(data).reset_index(drop=True))
        self.population_size = population_size
        self.n_survivors = n_survivors
        self.prob_mutate = prob_mutate
        self.join_thresh = join_thresh

        self.tightness_alpha = alpha
        self.tightness_beta = beta

        self.last_fitness = np.array([])
        self.pattern_scores = {}

        self.n_pools = n_pools or max(1, multiprocessing.cpu_count() - 1)
        self.generation = 0
        self.species = None
        self.generate_species()

    def make_species(self, pattern):
        return Species(representative=pattern, prob_mutate=self.prob_mutate, n_pools=self.n_pools,
                       random_seed_generator=self.random_seed_generator)

    def specify(self, population):
        population = sorted(population, reverse=True)

        self.species = []
        repr_pattern = []
        repr_indexes = []

        distances = np.ones([len(population)] * 2)
        # pool = multiprocessing.Pool(self.n_pools)
        for i, pattern in enumerate(population):

            if repr_indexes:
                if repr_indexes[-1] == i - 1:
                    # dist = pool.map(population[i-1].__sub__, population[i:])
                    distances[i - 1, i:] = [population[i - 1] - p for p in population[i:]]

                dist_species = distances[repr_indexes, i]
                index = int(np.argmin(dist_species))

                min_distance = dist_species[index]
                best_species = self.species[index]

                if min_distance <= self.join_thresh:
                    best_species.add(pattern)
                else:
                    species = self.make_species(pattern)
                    self.species.append(species)
                    repr_pattern.append(species.representative)
                    repr_indexes.append(i)
            else:
                species = self.make_species(pattern)
                self.species.append(species)
                repr_pattern.append(species.representative)
                repr_indexes.append(i)

        # pool.close()

        self.last_fitness = []
        for species in self.species:
            self.last_fitness.append(species.fitness)
        self.last_fitness = np.array(self.last_fitness)

    def generate_species(self):
        population = self.generate_population()
        population = self.calculate_fitness(population)
        self.specify(population)

    @staticmethod
    def _generate_population(indexes, random_seed, data_manager,
                             tightness_alpha, tightness_beta):
        np.random.seed(random_seed)
        random.seed(random_seed)

        result = []
        for i, j in indexes:
            a = WordSequence(data_manager.get(i))
            b = WordSequence(data_manager.get(j))
            intersection = a.intersection(b)
            if len(intersection):
                pattern = Pattern(intersection, tightness_alpha, tightness_beta)
                pattern.action = ['created']
                result.append(pattern)
        return result

    def generate_population(self, population_size=None):
        population_size = population_size or self.population_size

        result = []
        r = population_size
        while r:
            indexes = np.random.randint(0, self.data_manager.size, (r, 2))

            indexes_set = []
            size = int(np.ceil(len(indexes) / self.n_pools))
            for i in range(0, len(indexes), size):
                indexes_set.append(indexes[i: i + size])

            pool = multiprocessing.Pool(self.n_pools)
            f = functools.partial(self._generate_population, data_manager=self.data_manager,
                                  tightness_alpha=self.tightness_alpha,
                                  tightness_beta=self.tightness_beta)
            result += sum(pool.starmap(f, zip(indexes_set, self.random_seed_generator)), [])
            pool.close()

            r = population_size - len(result)

        return result

    @staticmethod
    def _calculate_fitness(population, random_seed, data_manager):
        np.random.seed(random_seed)
        random.seed(random_seed)

        patterns = []
        for pattern in population:
            pattern.calculate_score(data_manager)
            patterns.append(pattern)

        del data_manager
        gc.collect()

        return patterns

    def calculate_fitness(self, population):
        population_pending = []
        population_done = []
        for pattern in population:
            if pattern.fitness is None:
                population_pending.append(pattern)
            else:
                population_done.append(pattern)

        _population_pending = []
        for pattern in population_pending:
            scores = self.pattern_scores.get(pattern, None)
            if scores:
                pattern._tightness = scores['tightness']
                pattern.fitness = scores['fitness']
                pattern.accuracy = scores['accuracy']
                pattern.match_indexes = scores['match_indexes']
                pattern.match_count = scores['match_count']

                population_done.append(pattern)
            else:
                _population_pending.append(pattern)
        population_pending = _population_pending

        # re-sampling
        indices = []
        for pattern in population_done:
            if pattern.match_count > 0:
                i = np.random.choice(np.argwhere(pattern.match_indexes).T[0])
                indices.append(i)
        samples = list(self.data_manager.get(indices))
        for pattern, sample in zip(population_done, samples):
            if pattern.match_count > 0:
                pattern.sample = sample

        if population_pending:
            population_set = []
            size = int(np.ceil(len(population_pending) / self.n_pools))

            for i in range(0, len(population_pending), size):
                population_set.append(population_pending[i: i + size])

            pool = multiprocessing.Pool(self.n_pools)
            f = functools.partial(self._calculate_fitness, data_manager=self.data_manager)
            population_processed = sum(pool.starmap(f, zip(population_set, self.random_seed_generator)), [])
            pool.close()

            for pattern in population_processed:
                if pattern in self.pattern_scores:
                    continue

                scores = {
                    'tightness': pattern.tightness,
                    'fitness': pattern.fitness,
                    'accuracy': pattern.accuracy,
                    'match_indexes': pattern.match_indexes,
                    'match_count': pattern.match_count,
                }
                self.pattern_scores[pattern.copy()] = scores

            population_done += population_processed

        return population_done

    def eliminate(self):
        fitness_pct = self.last_fitness / self.last_fitness.sum()
        species_survivors = np.floor(fitness_pct * self.n_survivors)

        selector = []
        for species, n in zip(self.species, species_survivors):
            species.eliminate_to(n)
            selector.append(len(species) > 0)
        species = [species for species in self.species if len(species) > 0]

        fitness_pct = fitness_pct[selector]
        fitness_pct /= fitness_pct.sum()

        return species, fitness_pct

    def reproduce(self, species, ratio):
        species_filter = [s for s in species if len(s) > 1]
        ratio = ratio[[len(s) > 1 for s in species]]

        species_population_size = np.ceil(ratio * self.population_size)

        population = sum((s.population for s in species if len(s) <= 1), [])
        population += sum(itertools.starmap(Species.reproduce,
                                            zip(species_filter, species_population_size)), [])

        availability = self.population_size - len(population)
        if availability > 0:
            population += self.generate_population(availability)

        population = self.calculate_fitness(population)
        population = [pattern for pattern in population if pattern.fitness > 0]
        return population

    def step(self):
        species, fitness_pct = self.eliminate()
        population = self.reproduce(species, fitness_pct)
        self.specify(population)

        self.generation += 1
        return self.last_fitness
