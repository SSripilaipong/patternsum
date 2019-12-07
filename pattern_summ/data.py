import numpy as np
import pandas as pd


class DataManager:
    def __init__(self, data):
        self.data = pd.Series(data)
        self.size = self.data.shape[0]

    def get_matches(self, regex, n_sample=1):
        match = self.data.str.match(regex)
        m = np.array([False] * self.size)
        m[self.data[match].index] = True

        data_match = self.data[match]
        if data_match.shape[0] > 0:
            sample = list(data_match.sample(n_sample))
        else:
            sample = []

        return match.mean(), m, sample

    def get(self, i):
        return self.data.loc[i]
