from matplotlib import pyplot as plt
import numpy as np
import pickle
from random import shuffle


results_file = '/home/safoex/Documents/data/results_15_01_2020.pickle'

with open(results_file, 'rb') as f:
    results_dict = pickle.load(f)

cv_results = {}

non_none_iters = {}
models = {
    'meltacchin',
    'melpollo',
    'humana1',
    'humana2'
}

for model in models:
    ress = results_dict[2][0.15][model]
    non_none_iters[model] = [i for i, x in enumerate(ress) if x is not None]


valtest_split = 0.5
splits = {
    'val' : {},
    'test': {}
}

for model in models:
    shuffled_iters = non_none_iters[model].copy()
    shuffle(shuffled_iters)
    l = len(shuffled_iters)
    vals = int(l * valtest_split)
    splits['val'][model] = [i in shuffled_iters[:vals] for i in range(l)]
    splits['test'][model] = [i in shuffled_iters[vals:] for i in range(l)]



def simple_quadratic_metric(results_array, model, split):
    m = 0
    for o, s in zip(results_array, split):
        if s:
            _, _, t1, t2, pred_model, _ = o
            m += (1 - (float(t1) + float(t2)) / 2) ** 2 * (model == pred_model)
    return m


for epoch in range(2, 9):
    cv_results[epoch] = {}
    for t in range(3, 20):
        threshold = t / 20.0
        cv_results[epoch][threshold] = {
            key: sum(simple_quadratic_metric(np.array(results_dict[epoch][threshold][model]), model, splits[key][model]) for model in models)
            for key in ['val', 'test']
        }

print(cv_results)