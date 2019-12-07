import random
import numpy as np
import pandas as pd

from pattern_summ import PatternSummarization
from pattern_summ.species import Species


random_seed = 1234


def generate_data():
    string_list = pd.read_csv('data/tnames_mix.csv', encoding='utf-8', squeeze=True)

    filter_บริษัท = string_list.str.startswith('บริษัท')
    filter_บจ = string_list.str.startswith('บจ')
    filter_หจ = string_list.str.startswith('หจ')
    filter_จำกัด = string_list.str.contains('จำกัด')
    filter_มหาชน = string_list.str.contains('มหาชน')

    data_บริษัท_จำกัด = string_list[filter_บริษัท & ~filter_บจ & ~filter_หจ & filter_จำกัด & ~filter_มหาชน]
    data_บริษัท = string_list[filter_บริษัท & ~filter_บจ & ~filter_หจ & ~filter_จำกัด & ~filter_มหาชน]
    data_จำกัด = string_list[~filter_บริษัท & ~filter_บจ & ~filter_หจ & filter_จำกัด & ~filter_มหาชน]
    data_จำกัด_มหาชน = string_list[~filter_บริษัท & ~filter_บจ & ~filter_หจ & filter_จำกัด & filter_มหาชน]

    data_บจ_จำกัด = string_list[~filter_บริษัท & filter_บจ & ~filter_หจ & filter_จำกัด & ~filter_มหาชน]

    pattern_names = ['บริษัท_จำกัด', 'บริษัท', 'จำกัด', 'จำกัด_มหาชน', 'บจ_จำกัด']
    pattern_data = [data_บริษัท_จำกัด, data_บริษัท, data_จำกัด, data_จำกัด_มหาชน, data_บจ_จำกัด]

    thai_characters = [chr(3584 + i) for i in range(1, 90) if i not in (59, 60, 61, 62)]

    np.random.seed(random_seed)
    random_string = pd.Series([''.join([random.choice(thai_characters) for _ in range(length)])
                               for length in np.random.randint(16, 28, 1000)])
    np.random.seed()

    data = pd.concat([data_จำกัด.sample(500, random_state=random_seed),
                      data_บริษัท.sample(500, random_state=random_seed),
                      random_string], axis=0).reset_index(drop=True)
    return data


def main():
    data = generate_data()
    opt = PatternSummarization(data, 300, 120, random_seed=random_seed, alpha=2, beta=1,
                               prob_mutate_add=0.10, prob_mutate_merge=0.10, prob_mutate_split=0.45,
                               prob_mutate_drop=0.35,
                               n_best=10, min_acc=0.20)
    opt.evolve(generations=20, n_no_new_species=3)
    for s in opt.optimizer.species:  # type: Species
        p = s.ancestor
        print(f'    {p.fitness:5.2f} {p.accuracy:5.2f} {p.tightness:5.2f} {s.convergence:5.2f} {p.words}')
    print('n generations:', opt.optimizer.generation)
    print()
    print('result:')
    for p in opt.get_patterns():
        print(f'    {p}')


if __name__ == '__main__':
    main()
