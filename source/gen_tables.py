import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# seed, time_spent, training_fitness, training_acc, testing_fitness, testing_acc, n_features_used

def load_stats(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = [l.strip().split(',') for l in lines]
    return np.array(data, dtype=np.float)

dataset_names = ['iris', 'hepatitis', 'breastEW', 'multiple_features']
data_k10_20k = np.stack([load_stats('outputs_k10_20k/{}/stats.csv'.format(dataset)) for dataset in dataset_names], axis=0)
data_k100_40k = np.stack([load_stats('outputs_k100_40k/{}/stats.csv'.format(dataset)) for dataset in dataset_names], axis=0)

# removing the seed field
data_k10_20k = data_k10_20k[:, :, 1:]  
data_k100_40k = data_k100_40k[:, :, 1:]

plot_names = ['Tempo de\\\\Treino (s)', 'Fitness\\\\Treino', 'Acurácia\\\\Treino', 'Fitness\\\\Teste', 'Acurácia\\\\Teste', 'Número de\\\\Features']

beautiful_dataset_names = {'iris':'Iris', 'breastEW':'BreastEW', 'hepatitis':'Hepatitis', 'multiple_features':'Multiple Features'}


box_colors = ['darkkhaki', 'royalblue']

table = [] #[' & '.join(['\\multicolumn{{2}}{{|c|}}{}'.format(pn) for pn in plot_names])]

for edn, dn in enumerate(dataset_names):
    
    line = [beautiful_dataset_names[dn]]

    for epn, pn in enumerate(plot_names):

        data = np.array([data_k10_20k[edn, :, epn], data_k100_40k[edn, :, epn]])
        
        values = np.amin(data, axis=1)
        # values = np.amax(data, axis=1)
        line += ['${:.2f}$ & ${:.2f}$'.format(*values)]

        # mean_values = np.mean(data, axis=1)
        # std_values = np.std(data, axis=1)
        # line += ['${:.2f} \\pm {:.2f}$ & ${:.2f} \\pm {:.2f}$'.format(mean_values[0], std_values[0], mean_values[1], std_values[1])]

    line = ' & '.join(line)

    table += [line]

table = '\\\\\n'.join(table)
print(table)