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


for edn, dn in enumerate(dataset_names):
    
    line = [beautiful_dataset_names[dn]]

    data = np.array([data_k10_20k[edn, :, 4], data_k100_40k[edn, :, 4]])
    values = np.argmax(data, axis=1)
    print(dn, values)