from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

workdir = '/home/safoex/Documents/data/aae/release2/release2'
classes = {'babyfood': ['meltacchin', 'melpollo'],
           'babymilk': ['humana1', 'humana2']}



all_subclasses = sum([subcl for _, subcl in classes.items()], [])

def global_class_of(local_class, classes):
    for gcl, gcl_list in classes.items():
        if local_class in gcl_list:
            return gcl
    return None


extra_folder = 'local_%.2f' % 0.4

testclass = 'melpollo'

exp_res_pattern = '/home/safoex/Desktop/esafronov/data/%s/av_test_%d/exp_%d/results.pickle'

succeded = 0
improved = [0, 0, 0, 0, 0]
incorrect = 0
total = 0
errors = 0

ambiguity_threshold = 0.4

for av_test_n in range(5):
    for exp_n in range(20):
        try:
            res_class = None
            with open(exp_res_pattern % (testclass, 0, exp_n), 'rb') as f:
                res =pickle.load(f)
                initial_class = res[0][0][4]
                not_found_correct = True
                for i, r in enumerate(res):
                    res_class = r[0][4]
                    ambiguity = (r[0][2] + r[0][3]) / 2
                    #print(ambiguity, res_class)

                    if ambiguity <= ambiguity_threshold:
                        break

                if testclass == res_class:
                    succeded += 1
                    if res_class != initial_class:
                        improved[i] += 1
                    not_found_correct = False

                total += 1
                if not_found_correct:
                    incorrect += 1
        except Exception as e:
            print(e)
            errors += 1
            pass

print('total: ', total)
print('succeded: ', succeded)
print('improved: ', improved)
print('improved total: ', sum(improved))
print('incorrect: ', incorrect)
print('errors: ', errors)