from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

workdir = '/home/safoex/Documents/data/aae/release2/release2'
classes = {'babyfood': ['meltacchin', 'melpollo'],
           'babymilk': ['humana1', 'humana2']}
#
# classes = {'redboxes': ['tiramisu', 'pistacchi'],
#            'yellowboxes': ['cioccolato', 'vaniglia']}
#
# classes = {'yellowboxes': ['cioccolato', 'vaniglia']}
# classes = {'redboxes': ['tiramisu', 'pistacchi']}

# classes = {'babymilk': ['humana1', 'humana2']}

all_subclasses = sum([subcl for _, subcl in classes.items()], [])

def global_class_of(local_class, classes):
    for gcl, gcl_list in classes.items():
        if local_class in gcl_list:
            return gcl
    return None


pattern = "/home/safoex/Documents/data/finals/%s/%d/rad_0.30"

target_thresholds = np.linspace(0.05, 1, 20)
jumps_limit = 4
max_jumps_positive = [np.zeros_like(target_thresholds) for _ in range(jumps_limit)]
max_jumps_total = [np.zeros_like(target_thresholds) for _ in range(jumps_limit)]

total_success = 0
total_success2 = 0
total_success1 = 0
total_successes = [0,0,0,0,0]
total = 0
objs = {}
for n in range(4):
    for assumed_class in classes:
        for testclass in classes[assumed_class]:
            if testclass not in objs:
                objs[testclass] = {
                    'positives' : [np.zeros_like(target_thresholds) for _ in range(jumps_limit)],
                    'total': [np.zeros_like(target_thresholds) for _ in range(jumps_limit)]
                }
            # if assumed_class != 'babyfood':
            #     continue
            # if n == 2:
            #     continue
            panda_dir = pattern % (testclass, n)
            with open(panda_dir + '/results.pickle', 'rb') as f:
                results = pickle.load(f)

            for t, threshold in enumerate(target_thresholds):
                for i, exp in enumerate(results):
                    previous = i
                    target_achieved = False
                    good = False
                    if exp[0][0] is None:
                        break
                    need_to_print = (t == 3)
                    if need_to_print:
                        print()
                    for j in range(jumps_limit):
                        result = exp[j][0]
                        if result is None:
                            break

                        total_successes[j] += (exp[j][0][4] == testclass)
                        ambiguity = (result[2] + result[3]) / 2
                        est_class = result[4]
                        if need_to_print:
                            print(testclass, est_class)
                        if ambiguity < threshold or previous == exp[j][1]:
                            if testclass == est_class:
                                if not target_achieved:
                                    good = True
                                for k in range(j, jumps_limit):
                                # k = j
                                    objs[testclass]['positives'][k][t] += 1
                                    max_jumps_positive[k][t] += 1
                            target_achieved = True
                            for k in range(j, jumps_limit):
                            # k = j
                                objs[testclass]['total'][k][t] += 1
                                max_jumps_total[k][t] += 1
                            break
                        previous = exp[j][1]
                    if good:
                        total_success += 1
                    if target_achieved:
                        total_success2 += 1
                    total += 1

print(total, total_success, total_success / total)
print(total, total_success2, total_success2 / total)
print()
for j in range(jumps_limit):
    print(total, total_successes[j], total_successes[j] / total)

print()
t = int(0.6 * 20)
for j in range(jumps_limit):
    print(max_jumps_positive[j][t], max_jumps_positive[j][t] / max_jumps_total[j][t])


for j in range(3):
    plt.plot(target_thresholds, max_jumps_positive[j]/max_jumps_total[j], label='%d' % j)
plt.legend()
plt.show()

