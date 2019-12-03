import numpy as np
from sklearn.model_selection import train_test_split
from tree import Tree, Forest
import random
import copy
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


def fitness(expected, predicted):
    assert expected.shape == predicted.shape
    expected = np.argmax(expected, axis=1)
    predicted = np.argmax(predicted, axis=1)
    error = (expected != predicted).mean()
    # error = np.square(expected - predicted).mean() # descomentar linha que joga pra 0 ou 1 no arquivo tree.py (linha 63) caso descomentar aqui
    return 1.0 / (1.0 + error), error

fit_evals = 0
def evaluate_forest(forest, X_train, Y_train, count_evals=True):
    global fit_evals
    if count_evals:
        fit_evals += 1
        if fit_evals > MAX_EVALUATIONS:
            raise MaxEvaluationsExceeded()

    outputs = np.zeros([X_train.shape[0], Y_train.shape[1]])
    for i in range(X_train.shape[0]):
        outputs[i] = forest.forward(X_train[i, :])
    return fitness(Y_train, outputs)

def evaluate_population(population, X_train, Y_train):
    error = None
    for p in range(len(population)):
        population[p][1], e = evaluate_forest(population[p][0], X_train, Y_train, count_evals=False)
        if error is None or e < error:
            error = e
    return sorted(population, key=lambda x: x[1], reverse=True), error

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


def main(k, epochs, dataset):

    random.seed(SEED)
    np.random.seed(SEED)

    X, Y = data_loader('datasets/{}.npy'.format(dataset))
    n_features, n_classes = {'breastEW':(30, 2), 'hepatitis':(19, 2), 'multiple_features':(649, 10), 'iris':(4, 3)}[dataset]  # I put iris here just to test but it was not in the first assignment so we do not need to include it in our results
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train = data_normalizer(X_train) # Yes, should be normalized after the train test split
    X_test = data_normalizer(X_test)
    Y_train = one_hot_encoder(Y_train, n_classes)
    Y_test = one_hot_encoder(Y_test, n_classes)

    pr, pm, pc = 0.2, 0.2, 0.6 # Pr%, Pm%, Pc% in the paper
    n_pr, n_pm, n_pc = int(k * pr), int(k * pm), int(k * pc)

    record_fitness = []

    start_t = time.clock()

    population = [[Forest(input_size=X_train.shape[1], n_classes=n_classes), 0.0] for _ in range(k)]

    error = None

    for e in tqdm(range(epochs)):

        try:

            population, e0 = evaluate_population(population, X_train, Y_train)
            record_fitness.append(population[0][1])

            if error is None or e0 < error:
                error = e0

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

                f1, e1 = evaluate_forest(ia[0], X_train, Y_train)
                f2, e2 = evaluate_forest(ib[0], X_train, Y_train)

                error = min([e1, e2, error])

                a_appended = False
                b_appended = False

                if f1 > ia[1]:
                    new_individuals.append([copy.deepcopy(ia[0]), f1])
                    a_appended = True

                if f2 > ib[1]:
                    new_individuals.append([copy.deepcopy(ib[0]), f2])
                    b_appended = True

                # another possibility
                if not a_appended:
                    ia[0].trees[idx] = gen_2
                    f3, e3 = evaluate_forest(ia[0], X_train, Y_train)
                    if e3 < error:
                        error = e3
                    if f3 > ia[1]:
                        new_individuals.append([copy.deepcopy(ia[0]), f3])

                if not b_appended:
                    ib[0].trees[idx] = gen_1
                    f4, e4 = evaluate_forest(ib[0], X_train, Y_train)
                    if e4 < error:
                        error = e4
                    if f4 > ib[1]:
                        new_individuals.append([copy.deepcopy(ib[0]), f4])

                # revert
                ia[0].trees[idx] = ta
                ib[0].trees[idx] = tb
            # end of crossover

            new_individuals = sorted(new_individuals, key=lambda x: x[1], reverse=True)[:len(population_to_crossover)] # remove extra if there is more than the permited..

            next_gen += new_individuals[:n_pc]
            individuals_to_mutate = new_individuals[n_pc:]
            assert len(individuals_to_mutate) == n_pm

            new_individuals = []

            # Mutation 
            for f in individuals_to_mutate:

                while True:

                    idx = np.random.randint(n_classes)

                    t = f[0].trees[idx]

                    t_mutated = Tree.mutate(t)

                    f[0].trees[idx] = t_mutated

                    fit, e5 = evaluate_forest(f[0], X_train, Y_train)
                    if e5 < error:
                        error = e5

                    if fit > f[1]:
                        new_individuals.append([copy.deepcopy(f[0]), fit])
                        break
            # end of mutation

            assert len(new_individuals) == n_pm
            next_gen += new_individuals
            assert len(next_gen) == k

            population = next_gen

            if 0.01 > error:
                break
        
        except MaxEvaluationsExceeded:
            print('Maximum number of fitness evaluations reached at iteration {}'.format(e))
            break


    population, error = evaluate_population(population, X_train, Y_train)
    record_fitness.append(population[0][1])

    print('Time spent: {}'.format(time.clock() - start_t))

    best_forest, training_fitness = population[0]

    best_forest.build_visualization() # the best forest is saved. Before running, clean the folder

    print('Training fitness: ', training_fitness)

    plt.figure()
    plt.plot(np.arange(len(record_fitness)), record_fitness)
    plt.tight_layout()
    plt.show()

    ######## Testing

    testing_fitness, error = evaluate_forest(best_forest, X_test, Y_test, count_evals=False)
    print('Testing fitness: ', testing_fitness)
    print('Testing error: ', error)

    cm = evaluate_and_confusion_matrix(best_forest, X_test, Y_test)
    plot_confusion_matrix(cm, np.arange(n_classes))


if __name__ == "__main__":

    # for reproducibility
    SEED = 0
    MAX_EVALUATIONS = 20000 # 40000 in the paper

    dataset = 'iris'
    k = 10
    epochs = 7
    main(k, epochs, dataset)
