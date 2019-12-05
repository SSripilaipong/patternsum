import re
import numpy as np

from .word_sequence import WordSequence


class Pattern(WordSequence):
    def __init__(self, words, alpha=2, beta=1):
        super().__init__(*words)

        self.alpha = alpha
        self.beta = beta

        self._tightness = None

        self.fitness = None
        self.accuracy = None
        self.match_indexes = None
        self.match_count = None

        self.sample = None
        self.action = []

    def copy(self):
        pattern = Pattern(self.words)

        pattern.alpha = self.alpha
        pattern.beta = self.beta

        pattern._tightness = self._tightness

        pattern.fitness = self.fitness
        pattern.accuracy = self.accuracy
        pattern.match_indexes = self.match_indexes
        pattern.match_count = self.match_count

        pattern.sample = None
        pattern.action = list(self.action)

        return pattern

    def reset(self):
        self._tightness = None
        self.fitness = None
        self.accuracy = None
        self.match_indexes = None
        self.match_count = None

    def update(self):
        self.reset()

        self.calculate_tightness()

    def calculate_score(self, data_manager):
        self.accuracy, self.match_indexes, samples = data_manager.get_matches(self.regex)
        if samples:
            self.sample = samples[0]
        else:
            self.sample = None

        self.match_count = self.match_indexes.sum()
        self.fitness = self.accuracy * self.tightness

        return self.fitness

    def calculate_tightness(self):
        length = sum(map(len, self.words))
        n_words = len(self.words)
        self._tightness = (length ** self.alpha) * (n_words ** -self.beta)
        return self._tightness

    @property
    def tightness(self):
        if self._tightness is None:
            self.calculate_tightness()
        return self._tightness

    def crossover(self, other):
        ws = self.intersection(other)
        pattern = Pattern(ws, self.alpha, self.beta)
        action = {
            'name': 'crossover',
            'parent': (tuple(self), tuple(other)),
            'ws': tuple(pattern),
        }
        pattern.action.append(action)
        return pattern

    def mutate_add(self):
        if not self.sample:
            return

        index = np.random.randint(0, len(self.words) + 1)
        expr = self.get_regex_capture(index)
        for extract in re.findall(expr, self.sample):
            if extract:
                p = np.random.randint(0, len(extract))
                self.insert(index, extract[p])

                action = {
                    'name': 'mutate',
                    'type': 'add',
                    'params': {
                        'sample': self.sample,
                        'index': index,
                        'location': p,
                    },
                    'ws': tuple(self),
                }
                self.action.append(action)
                break

    def mutate_merge(self):
        if len(self) == 1:
            return

        index = np.random.randint(1, len(self))
        merge = ''.join(self[index - 1:index + 1])
        self.pop(index)
        self.pop(index - 1)

        self.insert(index - 1, merge)

        action = {
            'name': 'mutate',
            'type': 'merge',
            'params': {
                'index': index,
            },
            'ws': tuple(self),
        }
        self.action.append(action)

    def mutate_split(self):
        index = np.random.randint(0, len(self))
        if len(self[index]) == 1:
            return

        i = np.random.randint(1, len(self[index]))
        self.split(index, i)

        action = {
            'name': 'mutate',
            'type': 'split',
            'params': {
                'index': index,
                'location': i,
            },
            'ws': tuple(self),
        }
        self.action.append(action)

    def mutate_drop(self):
        if len(self) == 1 and len(self[0]) == 1:
            return

        i = np.random.randint(0, len(self) * 2) / 2

        index = int(i)
        if len(self[index]) <= 1:
            return

        if index == i:
            word = self[index][1:]
        else:
            word = self[index][:-1]
        self[index] = word

        action = {
            'name': 'mutate',
            'type': 'drop',
            'params': {
                'index': i,
            },
            'ws': tuple(self),
        }
        self.action.append(action)

    def mutate(self, prob_add, prob_merge, prob_split, prob_drop):
        prob = np.random.random()

        prob -= prob_add
        if prob <= 0:
            self.mutate_add()
            prob = 999

        prob -= prob_merge
        if prob <= 0:
            self.mutate_merge()
            prob = 999

        prob -= prob_split
        if prob <= 0:
            self.mutate_split()
            prob = 999

        prob -= prob_drop
        if prob <= 0:
            self.mutate_drop()

        self.update()

    def __sub__(self, other):
        assert self.match_indexes is not None
        assert other.match_indexes is not None

        intersection = (self.match_indexes & other.match_indexes).sum()
        total = self.match_count

        if total == 0:
            dist = 1
        else:
            dist = 1 - (intersection / total)

        return dist

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __repr__(self):
        return 'Pattern(' + '...'.join(map(repr, self)) + ')'
