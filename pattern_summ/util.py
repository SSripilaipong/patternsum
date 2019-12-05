import random
import string
import re

punc_pattern = re.compile('([' + string.punctuation + '])')


def keep_punc(r):
    return punc_pattern.sub(r'\\\1', r)


def get_random_seed_generator(low=1, high=99999, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    while True:
        yield random.randint(low, high)
