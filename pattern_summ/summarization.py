import pandas as pd

from .data_manager import DataManager
from .evolution import Evolution


class PatternSummarization:
    def __init__(self, data, population_size, n_survivors, prob_mutate=0.9, prob_mutate_add=0.20,
                 prob_mutate_merge=0.30, prob_mutate_split=0.30, prob_mutate_drop=0.20,
                 join_thresh=0.6, alpha=2, beta=1,
                 n_pools=None, random_seed=None):
        self.data_manager = DataManager(data=pd.Series(data).reset_index(drop=True))

        self.n_pools = n_pools
        self.random_seed = random_seed

        self.evolution = Evolution(data_manager=self.data_manager, population_size=population_size,
                                   n_survivors=n_survivors, tightness_alpha=alpha, tightness_beta=beta,
                                   join_thresh=join_thresh, prob_mutate=prob_mutate,
                                   prob_mutate_add=prob_mutate_add, prob_mutate_merge=prob_mutate_merge,
                                   prob_mutate_split=prob_mutate_split, prob_mutate_drop=prob_mutate_drop,
                                   n_pools=self.n_pools, random_seed=self.random_seed)

    @property
    def n_survivors(self):
        return self.evolution.n_survivors

    @property
    def population_size(self):
        return self.evolution.population_size

    def _initialize(self, random_seed):
        if random_seed:
            self.evolution.set_random_seed(random_seed)

        self.evolution.initialize()

    def fit(self, generations=1, random_seed=None, reset=True):
        if reset:
            self._initialize(random_seed)

        for _ in range(generations):
            self.evolution.step()
