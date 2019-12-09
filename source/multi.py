import numpy as np
from sklearn.model_selection import train_test_split
from tree import Tree, Forest, copy_obj
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix


def data_loader(path):
    data = np.load(path)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y

def data_normalizer(X):
    X /= np.amax(X, axis=0)
    X = (X * 2.0) - 1.0
    return X


class MaxEvaluationsExceeded(Exception):
    pass


def forest_accuracy(forest, X, Y):
    outputs = np.zeros([X.shape[0], Y.shape[1]])
    for i in range(X.shape[0]):
        outputs[i] = forest.forward(X[i, :])
    
    assert Y.shape == outputs.shape
    expected = np.argmax(Y, axis=1)
    predicted = np.argmax(outputs, axis=1)
    return (expected == predicted).mean()


def fitness(expected, predicted, count_evals=True):
    assert expected.shape == predicted.shape
    
    global fit_evals
    if count_evals:
        fit_evals += 1
        if fit_evals > MAX_EVALUATIONS:
            raise MaxEvaluationsExceeded()

    expected = np.argmax(expected, axis=1)
    predicted = np.argmax(predicted, axis=1)
    error = (expected != predicted).mean()

    # error = np.square(expected - predicted).mean() # descomentar linha que joga pra 0 ou 1 no arquivo tree.py (linha 63) caso descomentar aqui
    # atualizar error_from_fitness function
    
    return 1.0 / (1.0 + error)
    # return (expected == predicted).mean()

def error_from_fitness(f):
    return 1.0/f - 1.0
    # return 1.0 - f

def evaluate_forest(forest, X_train, Y_train, count_evals=True):
    outputs = np.zeros([X_train.shape[0], Y_train.shape[1]])
    for i in range(X_train.shape[0]):
        outputs[i] = forest.forward(X_train[i, :])
    return fitness(Y_train, outputs, count_evals)

def evaluate_population(population, X_train, Y_train):
    for p in range(len(population)):
        population[p][1] = evaluate_forest(population[p][0], X_train, Y_train, count_evals=False)
    return sorted(population, key=lambda x: x[1], reverse=True)

def one_hot_encoder(classes, n_classes):
    assert len(classes.shape) == 1
    one_hot = np.zeros((classes.size, n_classes))
    one_hot[np.arange(classes.size), classes.astype(np.int64)] = 1.0
    return one_hot

def evaluate_and_confusion_matrix(forest, X, Y):
    outputs = np.zeros([X.shape[0], Y.shape[1]])
    for i in range(X.shape[0]):
        outputs[i] = forest.forward(X[i, :])

    expected = np.argmax(Y, axis=1)
    predicted = np.argmax(outputs, axis=1)
    
    return confusion_matrix(expected, predicted)


def plot_confusion_matrix(cm, classes, normalize=True, filepath=None):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if filepath == None:
        plt.show()
    else:
        plt.savefig(filepath)
    plt.close()


def main(k, epochs, dataset, seed, output_folder):

    X, Y = data_loader('datasets/{}.npy'.format(dataset))
    n_features, n_classes = {'breastEW':(30, 2), 'hepatitis':(19, 2), 'multiple_features':(649, 10), 'iris':(4, 3)}[dataset]  # I put iris here just to test but it was not in the first assignment so we do not need to include it in our results
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, stratify=Y)
    X_train = data_normalizer(X_train) # Yes, should be normalized after the train test split
    X_test = data_normalizer(X_test)
    Y_train = one_hot_encoder(Y_train, n_classes)
    Y_test = one_hot_encoder(Y_test, n_classes)

    stop_error = 0.01
    pr, pm, pc = 0.2, 0.2, 0.6 # Pr%, Pm%, Pc% in the paper
    n_pr, n_pm, n_pc = int(k * pr), int(k * pm), int(k * pc)

    record_fitness = []

    start_t = time.clock()

    population = [[Forest(input_size=X_train.shape[1], n_classes=n_classes), 0.0] for _ in range(k)]

    for e in tqdm(range(epochs)):

        try:

            population = evaluate_population(population, X_train, Y_train)
            record_fitness.append(population[0][1])

            best_error = error_from_fitness(population[0][1])
            if stop_error > best_error:
                # print('Minimum error {} with fitness {} reached at iteration {}'.format(best_error, population[0][1], e))
                break

            next_gen = population[:n_pr]
            population_to_crossover = population[n_pr:]
            new_individuals = []

            # Crossover 
            while len(new_individuals) < len(population_to_crossover):

                fidx_1, fidx_2 = np.random.permutation(len(population_to_crossover))[:2]

                ia = population_to_crossover[fidx_1]
                ib = population_to_crossover[fidx_2]

                idx = np.random.randint(n_classes)

                ta = ia[0].trees[idx]
                tb = ib[0].trees[idx]

                gen_1, gen_2 = Tree.crossover(ta, tb)

                # one possibility
                ia[0].trees[idx] = gen_1
                ib[0].trees[idx] = gen_2

                f1 = evaluate_forest(ia[0], X_train, Y_train)
                f2 = evaluate_forest(ib[0], X_train, Y_train)

                a_appended, b_appended = False, False

                if f1 > ia[1]:
                    new_individuals.append([copy_obj(ia[0]), f1])
                    a_appended = True

                if f2 > ib[1]:
                    new_individuals.append([copy_obj(ib[0]), f2])
                    b_appended = True

                # another possibility
                if not a_appended:
                    ia[0].trees[idx] = gen_2
                    f3 = evaluate_forest(ia[0], X_train, Y_train)
                    if f3 > ia[1]:
                        new_individuals.append([copy_obj(ia[0]), f3])

                if not b_appended:
                    ib[0].trees[idx] = gen_1
                    f4 = evaluate_forest(ib[0], X_train, Y_train)
                    if f4 > ib[1]:
                        new_individuals.append([copy_obj(ib[0]), f4])

                # revert
                ia[0].trees[idx] = ta
                ib[0].trees[idx] = tb
            # end of crossover

            new_individuals = sorted(new_individuals, key=lambda x: x[1], reverse=True)[:len(population_to_crossover)] # remove extra if there is more than the permited..

            next_gen += new_individuals[:n_pc]
            individuals_to_mutate = new_individuals[n_pc:]


            new_individuals = []

            # Mutation 
            for f in individuals_to_mutate:

                while True:

                    idx = np.random.randint(n_classes)

                    t = f[0].trees[idx]

                    t_mutated = Tree.mutate(t)

                    f[0].trees[idx] = t_mutated

                    fit = evaluate_forest(f[0], X_train, Y_train)

                    appended = False
                    if fit > f[1]:
                        new_individuals.append([copy_obj(f[0]), fit])
                        appended = True

                    # revert
                    f[0].trees[idx] = t

                    if appended:
                        break
            # end of mutation

            next_gen += new_individuals

            population = next_gen
        
        except MaxEvaluationsExceeded:
            # print('Maximum number of fitness evaluations reached at iteration {}'.format(e))
            break


    population = evaluate_population(population, X_train, Y_train)
    record_fitness.append(population[0][1])

    time_spent = time.clock() - start_t

    best_forest, training_fitness = population[0]

    best_forest.build_visualization('{}graphviz/'.format(output_folder)) # the best forest is saved. Before running, clean the folder

    training_acc = forest_accuracy(best_forest, X_train, Y_train)

    plt.figure()
    plt.plot(np.arange(len(record_fitness)), record_fitness)
    plt.tight_layout()
    plt.savefig('{}/best_train_fitness_hist.jpg'.format(output_folder))
    plt.close()

    ######## Testing

    testing_fitness = evaluate_forest(best_forest, X_test, Y_test, count_evals=False)
    testing_acc = forest_accuracy(best_forest, X_test, Y_test)

    cm = evaluate_and_confusion_matrix(best_forest, X_test, Y_test)
    plot_confusion_matrix(cm, np.arange(n_classes), filepath='{}/test_cm.jpg'.format(output_folder))

    n_features_used = len(best_forest.get_number_used_features().keys())

    return time_spent, training_fitness, training_acc, testing_fitness, testing_acc, n_features_used


if __name__ == "__main__":

    MAX_EVALUATIONS = 40000 # 40000 in the paper
    k = 100
    epochs = MAX_EVALUATIONS # tava 10

    # max evaluation can be divided by number of features..

    for dataset in tqdm(['iris', 'breastEW', 'hepatitis', 'multiple_features']):
        
        record_stats = []
        
        for seed in tqdm(range(20)):
            fit_evals = 0 # global variable
            random.seed(seed)
            np.random.seed(seed)
            output_folder = 'outputs/{}/{}/'.format(dataset, seed)
            outputs = main(k, epochs, dataset, seed, output_folder)
            record_stats.append((seed,) + outputs)

        with open('outputs/{}/stats.csv'.format(dataset), 'w') as f:
            f.write('\n'.join(['{},{},{},{},{},{},{}'.format(*rs) for rs in record_stats]))
