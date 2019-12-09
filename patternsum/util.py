import random
import string
import re

punc_pattern = re.compile('([' + string.punctuation + '])')


def keep_punc(r):
    return punc_pattern.sub(r'\\\1', r)


def get_random_seed_generator(low=1, high=99999):
    while True:
        yield random.randint(low, high)
