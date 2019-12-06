import pandas as pd

from .data_manager import DataManager
from .optimizer import Optimizer


class PatternSummarization:
    def __init__(self, data, population_size, n_survivors, prob_mutate=0.9, prob_mutate_add=0.20,
                 prob_mutate_merge=0.30, prob_mutate_split=0.30, prob_mutate_drop=0.20,
                 join_thresh=0.6, alpha=2, beta=1,
                 n_best=None, min_acc=0.25, conv_thresh=1,
                 n_pools=None, random_seed=None):
        self.data_manager = DataManager(data=pd.Series(data).reset_index(drop=True))

        self.n_best = n_best
        self.min_acc = min_acc
        self.conv_thresh = conv_thresh

        self.n_pools = n_pools
        self.random_seed = random_seed

        self.optimizer = Optimizer(data_manager=self.data_manager, population_size=population_size,
                                   n_survivors=n_survivors, tightness_alpha=alpha, tightness_beta=beta,
                                   join_thresh=join_thresh, prob_mutate=prob_mutate,
                                   prob_mutate_add=prob_mutate_add, prob_mutate_merge=prob_mutate_merge,
                                   prob_mutate_split=prob_mutate_split, prob_mutate_drop=prob_mutate_drop,
                                   n_pools=self.n_pools, random_seed=self.random_seed)

    @property
    def n_survivors(self):
        return self.optimizer.n_survivors

    @property
    def population_size(self):
        return self.optimizer.population_size

    def _initialize(self, random_seed):
        if random_seed:
            self.optimizer.set_random_seed(random_seed)

        self.optimizer.initialize()

    def evolve(self, generations=1, random_seed=None, reset=False):
        if reset or self.optimizer.generation == 0:
            self._initialize(random_seed)

        for _ in range(generations):
            self.optimizer.step()

    def get_patterns(self):
        species = filter(lambda s: s.convergence >= self.conv_thresh, self.optimizer.species[:self.n_best])
        patterns = map(lambda s: s.representative.copy(), species)
        patterns = filter(lambda p: p.accuracy >= self.min_acc, patterns)
        patterns = tuple(patterns)
        return patterns
