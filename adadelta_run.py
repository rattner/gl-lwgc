from subprocess import Popen
import os
import sys

gpu=0
eps = [1.0, 8e-1, 3e-1, 2e-2]
epsu = [1.0, 8e-1, 8e-2, 2e-2, 2e-3, 2e-4, 0.0]
lamb = [0.035, 0.045, 0.055]
lr = [0.05, 0.2, 0.35, 0.8]
gamma = [0.75, 0.85, 0.9]

for e in eps:
    for eu in epsu:
        for l in lamb:
            for r in lr:
                for g in gamma:
                    line = 'activate pytorch & python dynamiceval.py' + ' --epsilon=' + str(e) + ' --epsilonu=' + str(eu) + ' --lamb=' + str(l) + ' --lr=' + str(r) + ' --gamma=' + str(g) + ' --gpu=' + str(gpu)
                    os.system(line)
