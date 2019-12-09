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

plot_names = ['Tempo de\nTreino (s)', 'Fitness\nTreino', 'Acurácia\nTreino', 'Fitness\nTeste', 'Acurácia\nTeste', 'Número de\nFeatures']

beautiful_dataset_names = {'iris':'Iris', 'breastEW':'BreastEW', 'hepatitis':'Hepatitis', 'multiple_features':'Multiple Features'}


box_colors = ['darkkhaki', 'royalblue']

for edn, dn in enumerate(dataset_names):
    
    fig, axs = plt.subplots(1, len(plot_names), sharey=False, figsize=(10, 5))

    for epn, pn in enumerate(plot_names):

        data = [data_k10_20k[edn, :, epn], data_k100_40k[edn, :, epn]]

        bp = axs[epn].boxplot(data, notch=0, sym='m+', vert=1, whis=1.5, widths=0.75)

        axs[epn].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        axs[epn].set_xlabel(pn, fontsize=12)

        num_boxes = 2
        medians = np.empty(num_boxes)
        for i in range(num_boxes):
            box = bp['boxes'][i]
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            box_coords = np.column_stack([boxX, boxY])
            # Alternate between Dark Khaki and Royal Blue
            axs[epn].add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
            # Now draw the median lines back over what we just filled in
            med = bp['medians'][i]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                axs[epn].plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
            # Finally, overplot the sample averages, with horizontal alignment
            # in the center of each box
            axs[epn].plot(np.average(med.get_xdata()), np.average(data[i]),
                    color='w', marker='*', markeredgecolor='k')

        # Due to the Y-axis scale being different across samples, it can be
        # hard to compare differences in medians across the samples. Add upper
        # X-axis tick labels with the sample medians to aid in comparison
        # (just use two decimal places of precision)
        pos = np.arange(num_boxes) + 1
        upper_labels = [str(np.round(s, 2)) for s in medians]
        weights = ['bold', 'semibold']
        for tick, label in zip(range(num_boxes), axs[epn].get_xticklabels()):
            k = tick % 2
            axs[epn].text(pos[tick], 1.02, upper_labels[tick], #1.05
                    transform=axs[epn].get_xaxis_transform(),
                    horizontalalignment='center', fontsize=10,
                    weight=weights[k], color=box_colors[k])

            # adds the mean.. 
            # axs[epn].text(pos[tick], 1.05, str(np.round(np.average(data[tick]), 2)), #1.05
            #         transform=axs[epn].get_xaxis_transform(),
            #         horizontalalignment='center', fontsize=10,
            #         weight=weights[k], color=box_colors[k])

        axs[epn].set_xticklabels([])
        axs[epn].set_xticks([])
            

    plt.tight_layout()
    # plt.show()
    plt.savefig('outputs_plot/boxplot_{}.jpg'.format(dn))
    plt.close()
