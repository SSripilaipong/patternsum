import random
import functools
import multiprocessing
import gc
import numpy as np


from pattern_summ.core.species import Species
from pattern_summ.util import get_random_seed_generator
from pattern_summ.pattern import WordSequence, Pattern
from .score import ScoreManager


class Optimizer:
    def __init__(self, data_manager, population_size, n_survivors, tightness_alpha=2, tightness_beta=1,
                 join_thresh=0.5, prob_mutate=0.9, prob_mutate_add=0.20, prob_mutate_merge=0.30,
                 prob_mutate_split=0.30, prob_mutate_drop=0.20, n_pools=None, random_seed=None):
        self.data_manager = data_manager
        self.population_size = population_size
        self.n_survivors = n_survivors
        self.tightness_alpha = tightness_alpha
        self.tightness_beta = tightness_beta
        self.join_thresh = join_thresh

        self.prob_mutate = prob_mutate
        self.prob_mutate_add = prob_mutate_add
        self.prob_mutate_merge = prob_mutate_merge
        self.prob_mutate_split = prob_mutate_split
        self.prob_mutate_drop = prob_mutate_drop

        self.n_pools = n_pools or max(1, multiprocessing.cpu_count() - 1)
        self.random_seed = random_seed
        self.random_seed_generator = None

        self.fitness = np.array([])
        self.species = []
        self.population = []
        self.generation = 0

        self.pattern_scores = ScoreManager()

    def initialize(self):
        if self.random_seed:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        self.random_seed_generator = get_random_seed_generator(1, 99999, self.random_seed)

        self.generate_population(self.population_size)
        self.calculate_fitness()
        self.specify()

        self.generation = 0

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

        if self.random_seed:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        self.random_seed_generator = get_random_seed_generator(1, 99999, self.random_seed)

    def make_species(self, pattern):
        return Species(ancestor=pattern, prob_mutate=self.prob_mutate,
                       prob_mutate_add=self.prob_mutate_add, prob_mutate_merge=self.prob_mutate_merge,
                       prob_mutate_split=self.prob_mutate_split, prob_mutate_drop=self.prob_mutate_drop,
                       generation=self.generation, n_pools=self.n_pools,
                       random_seed_generator=self.random_seed_generator)

    def specify(self):
        self.population = sorted(self.population, reverse=True)

        self.fitness = np.array([])
        self.species = []
        repr_pattern = []
        repr_indexes = []

        distances = np.ones([len(self.population)] * 2)
        for i, pattern in enumerate(self.population):

            if repr_indexes:
                if repr_indexes[-1] == i - 1:
                    distances[i - 1, i:] = [self.population[i - 1] - p for p in self.population[i:]]

                dist_species = distances[repr_indexes, i]
                index = int(np.argmin(dist_species))

                min_distance = dist_species[index]
                best_species = self.species[index]

                if min_distance <= self.join_thresh:
                    best_species.add(pattern)
                else:
                    species = self.make_species(pattern)
                    self.species.append(species)
                    repr_pattern.append(species.ancestor)
                    repr_indexes.append(i)
            else:
                species = self.make_species(pattern)
                self.species.append(species)
                repr_pattern.append(species.ancestor)
                repr_indexes.append(i)

        self.fitness = []
        for species in self.species:
            self.fitness.append(species.fitness)

        self.fitness = np.array(self.fitness)

    def generate_population(self, population_size):
        result = []
        r = population_size
        while r:
            indexes = np.random.randint(0, self.data_manager.size, (r, 2))

            indexes_set = []
            size = int(np.ceil(len(indexes) / self.n_pools))
            for i in range(0, len(indexes), size):
                indexes_set.append(indexes[i: i + size])

            pool = multiprocessing.Pool(self.n_pools)
            f = functools.partial(_generate_population, data_manager=self.data_manager,
                                  tightness_alpha=self.tightness_alpha,
                                  tightness_beta=self.tightness_beta, generation=self.generation)
            result += sum(pool.starmap(f, zip(indexes_set, self.random_seed_generator)), [])
            pool.close()

            r = population_size - len(result)

        self.population = result

    def _calculate_fitness(self, patterns):
        population_set = []
        size = int(np.ceil(len(patterns) / self.n_pools))

        for i in range(0, len(patterns), size):
            population_set.append(patterns[i: i + size])

        pool = multiprocessing.Pool(self.n_pools)
        f = functools.partial(_calculate_fitness, data_manager=self.data_manager)
        population_processed = sum(pool.starmap(f, zip(population_set, self.random_seed_generator)), [])
        pool.close()

        return population_processed

    def refresh_pattern_sample(self, patterns):
        indices = []
        for pattern in patterns:
            if pattern.match_count > 0:
                i = np.random.choice(np.argwhere(pattern.match_indexes).T[0])
                indices.append(i)

        samples = list(self.data_manager.get(indices))
        for pattern, sample in zip(patterns, samples):
            if pattern.match_count > 0:
                pattern.sample = sample

    def calculate_fitness(self):
        population_pending = [pattern for pattern in self.population if pattern.fitness is None]
        population_done = [pattern for pattern in self.population if pattern.fitness is not None]

        population_matched, population_pending = self.pattern_scores.query(population_pending)
        population_done.extend(population_matched)

        self.refresh_pattern_sample(population_done)

        if population_pending:
            population_processed = self._calculate_fitness(population_pending)
            self.pattern_scores.update(population_processed)

            population_done += population_processed

        self.population = population_done

    def eliminate(self):
        fitness_pct = self.fitness / self.fitness.sum()
        species_survivors = np.floor(fitness_pct * self.n_survivors)

        selector = []
        for species, n in zip(self.species, species_survivors):
            species.eliminate_to(n)
            selector.append(len(species) > 0)
        species = [species for species in self.species if len(species) > 0]

        fitness_pct = fitness_pct[selector]
        fitness_pct /= fitness_pct.sum()

        self.species = species
        self.fitness = self.fitness[selector]

    def reproduce(self):
        species_filter = [s for s in self.species if len(s) > 1]
        ratio = self.fitness[[len(s) > 1 for s in self.species]]
        ratio /= ratio.sum()

        species_population_size = np.ceil(ratio * self.population_size)

        self.population = sum((s.population for s in self.species if len(s) <= 1), [])
        for sp, si in zip(species_filter, species_population_size):
            self.population += sp.reproduce(si, self.generation)

        availability = self.population_size - len(self.population)
        if availability > 0:
            self.generate_population(availability)

        self.calculate_fitness()
        self.population = [pattern for pattern in self.population if pattern.fitness > 0]
        self.species = []

    def step(self):
        for s in self.species:
            s.generation += 1
        self.generation += 1

        self.eliminate()
        self.reproduce()
        self.specify()


def _generate_population(indexes, random_seed, data_manager, tightness_alpha, tightness_beta, generation):
    np.random.seed(random_seed)
    random.seed(random_seed)

    result = []
    for i, j in indexes:
        a = WordSequence(data_manager.get(i))
        b = WordSequence(data_manager.get(j))
        intersection = a.intersection(b)
        if len(intersection):
            pattern = Pattern(intersection, tightness_alpha, tightness_beta)
            action = {
                'name': 'create',
                'string': (tuple(a), tuple(b)),
                'ws': tuple(pattern),
                'generation': generation,
            }
            pattern.action.append(action)
            result.append(pattern)
    return result


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
