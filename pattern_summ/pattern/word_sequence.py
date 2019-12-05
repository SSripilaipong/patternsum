from pattern_summ.util import keep_punc
import numpy as np


class WordSequence:
    def __init__(self, *words):
        assert all(map(lambda t: type(t) is str, words))
        self.words = list(words)

        self._flatten = None
        self._word_index = None
        self._regex = None
        self._words_keep_punc = None

    def copy(self):
        ws = WordSequence(*self.words)

        ws._flatten = self._flatten
        ws._word_index = self._word_index
        ws._regex = self._regex
        ws._words_keep_punc = self._words_keep_punc

        return ws

    def reset(self):
        self._flatten = None
        self._word_index = None
        self._regex = None
        self._words_keep_punc = None

    def _get_match_score(self, other):
        seq_0, seq_1 = self.flatten, other.flatten
        len_0, len_1 = map(len, (seq_0, seq_1))
        d = np.zeros((len_0 + 1, len_1 + 1))

        for i in range(1, len_0 + 1):
            for j in range(1, len_1 + 1):
                i_, j_ = i - 1, j - 1

                if seq_0[i_] == seq_1[j_]:
                    d[i, j] = 1 + d[i_, j_]
                else:
                    d[i, j] = max(d[i_, j], d[i, j_])
        return d

    def intersection(self, other):
        ind_0, ind_1 = self.word_index, other.word_index
        d = self._get_match_score(other)

        seq_0, seq_1 = self.flatten, other.flatten
        i, j = map(len, (seq_0, seq_1))

        ws = [[]]

        while i > 0 and j > 0:
            i_, j_ = i - 1, j - 1

            m = max([d[i_, j_], d[i_, j], d[i, j_]])
            if m == d[i, j] - 1:
                assert seq_0[i_] == seq_1[j_]

                ws[-1].append(seq_0[i_])
            elif ws[-1]:
                ws.append([])

            if 1 in (ind_0[i_], ind_1[j_]) and ws[-1]:
                ws.append([])

            if d[i_, j_] == m:
                i -= 1
                j -= 1
            elif d[i_, j] == m:
                i -= 1
            elif d[i, j_] == m:
                j -= 1

        if not ws[-1]:
            ws = ws[:-1]

        result = list(reversed(list(map(lambda t: ''.join(reversed(t)), ws))))

        result_adj = []
        w_0, w_1 = map(list, (self.words, other.words))
        i, j = 0, 0

        skip = False
        for w_x, w_y in zip(result, result[1:] + ['']):
            if skip:
                skip = False
                continue

            while w_x not in w_0[i]:
                i += 1
            p = w_0[i].index(w_x) + len(w_x)
            w_0[i] = w_0[i][p:]
            if not w_0:
                i += 1
                result_adj.append(w_x)
                continue
            elif not w_0[i].startswith(w_y):
                result_adj.append(w_x)
                continue

            while w_x not in w_1[j]:
                j += 1
            q = w_1[j].index(w_x) + len(w_x)
            w_1[j] = w_1[j][q:]
            if not w_1:
                j += 1
                result_adj.append(w_x)
                continue
            elif not w_1[j].startswith(w_y):
                result_adj.append(w_x)
                continue

            result_adj.append(w_x + w_y)
            skip = True

        return WordSequence(*result_adj)

    def _calculate_flatten(self):
        self._flatten = sum(map(list, self.words), [])
        self._word_index = [0] * sum(map(len, self.words))

        i = 0
        for s in self.words:
            self._word_index[i] = 1
            i += len(s)

    def split(self, w, c):
        split = list(filter(lambda t: t, (self.words[w][:c], self.words[w][c:])))
        self.words = self.words[:w] + split + self.words[w + 1:]
        self.reset()

    @property
    def flatten(self):
        if not self._flatten:
            self._calculate_flatten()
        return self._flatten

    @property
    def word_index(self):
        if not self._word_index:
            self._calculate_flatten()
        return self._word_index

    @property
    def words_keep_punc(self):
        if not self._words_keep_punc:
            self._words_keep_punc = list(map(keep_punc, self.words))
        return self._words_keep_punc

    @property
    def regex(self):
        if not self._regex:
            self._regex = r'.*?'.join([''] + self.words_keep_punc + [''])
        return self._regex

    def get_regex_capture(self, x):
        words = self.words_keep_punc
        loc = sum(3 + len(words[i]) for i in range(x))
        regex = self.regex
        regex = f'{regex[:loc]}({regex[loc: loc + 3]}){regex[loc + 3:]}'
        return regex

    def insert(self, *args, **kwargs):
        self.words.insert(*args, **kwargs)
        self.reset()

    def pop(self, *args, **kwargs):
        self.words.pop(*args, **kwargs)
        self.reset()

    def __setitem__(self, *args, **kwargs):
        self.words.__setitem__(*args, **kwargs)
        self.reset()

    def __getitem__(self, *args, **kwargs):
        return self.words.__getitem__(*args, **kwargs)

    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)

    def __hash__(self):
        return hash(tuple(self.words))

    def __eq__(self, other):
        return self.words == other.words

    def __repr__(self):
        return f'WordSequence{(*self.words,)}'
