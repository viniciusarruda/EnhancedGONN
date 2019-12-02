import numpy as np
from sklearn.model_selection import train_test_split
from tree import Tree, Forest
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# for reproducibility
SEED = 0

def data_loader(path):

    data = np.load(path)

    X = data[:, :-1]
    Y = data[:, -1]

    return X, Y


def data_normalizer(X):
    
    X /= np.amax(X, axis=0) 
    X = (X * 2.0) - 1.0
    return X


# TODO: should be in other file.. maybe in the tree file that should be called GONN
# Esta fitness eh muito esquisita!
def fitness(expected, predicted):
    assert expected.shape == predicted.shape
    return 1.0 / (1.0 + np.square(expected - predicted).mean())

def evaluate_forest(forest, X_train, Y_train):
    outputs = np.zeros([X_train.shape[0], Y_train.shape[1]])
    for i in range(X_train.shape[0]):
        outputs[i] = forest.forward(X_train[i, :])
    return fitness(Y_train, outputs)


def evaluate_population(population, X_train, Y_train):
    for p in range(len(population)):
        population[p][1] = evaluate_forest(population[p][0], X_train, Y_train)
    return sorted(population, key=lambda x: x[1], reverse=True)

def one_hot_encoder(classes, n_classes):
    
    assert len(classes.shape) == 1
    one_hot = np.zeros((classes.size, n_classes))
    one_hot[np.arange(classes.size), classes.astype(np.int64)] = 1.0
    return one_hot

def main():
    
    random.seed(SEED)
    np.random.seed(SEED)

    dataset = 'iris'

    X, Y = data_loader('datasets/{}.npy'.format(dataset))
    n_features, n_classes = {'breastEW':(30, 2), 'hepatitis':(19, 2), 'multiple_features':(649, 10), 'iris':(4, 3)}[dataset]  # I put iris here just to test but it was not in the first assignment so we do not need to include it in our results
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train = data_normalizer(X_train) # Yes, should be normalized after the train test split
    X_test = data_normalizer(X_test)
    Y_train = one_hot_encoder(Y_train, n_classes)
    Y_test = one_hot_encoder(Y_test, n_classes)

    k = 10
    epochs = 10 #14
    pr = 0.2 # Pr% in the paper
    pm = 0.2 # Pm% in the paper
    pc = 0.6 # Pc% in the paper
    n_pr = int(k * pr)
    n_pc = int(k * pc)
    n_pm = int(k * pm)

    record_fitness = np.zeros((epochs+1, k))

    population = [[Forest(input_size=X_train.shape[1], n_classes=n_classes), 0.0] for _ in range(k)]

    for e in tqdm(range(epochs)):

        population = evaluate_population(population, X_train, Y_train)

        for i in range(k):
            record_fitness[e, i] = population[i][1]

        next_gen = population[:n_pr]
        population = population[n_pr:]
        new_individuals = []

        # Crossover (should not be inside the Forest class ?)
        while len(new_individuals) < len(population):

            fidx_1, fidx_2 = np.random.permutation(len(population))[:2]

            ia = population[fidx_1]
            ib = population[fidx_2]

            idx = np.random.randint(n_classes)

            ta = ia[0].trees[idx]
            tb = ib[0].trees[idx]

            gen_1, gen_2 = Tree.crossover(ta, tb)

            # one possibility
            ia[0].trees[idx] = gen_1
            ib[0].trees[idx] = gen_2

            f1 = evaluate_forest(ia[0], X_train, Y_train)
            f2 = evaluate_forest(ib[0], X_train, Y_train)

            if f1 > ia[1]:  
                new_individuals.append([copy.deepcopy(ia[0]), f1])
            
            if f2 > ib[1]:
                new_individuals.append([copy.deepcopy(ib[0]), f2])

            # another possibility
            ia[0].trees[idx] = gen_2
            ib[0].trees[idx] = gen_1

            f3 = evaluate_forest(ia[0], X_train, Y_train)
            f4 = evaluate_forest(ib[0], X_train, Y_train)

            if f3 > ia[1]:
                new_individuals.append([copy.deepcopy(ia[0]), f3])

            if f4 > ib[1]:
                new_individuals.append([copy.deepcopy(ib[0]), f4])

            # se esta nova arvore gerada acabar entrando em todas as florestas, isso nao iria dimiuir a diversidade nao ?
            # talves se entrou em aguma floresta, nao inserir em outra ?

            # revert
            ia[0].trees[idx] = ta
            ib[0].trees[idx] = tb
        # end of crossover

        new_individuals = sorted(new_individuals, key=lambda x: x[1], reverse=True)[:len(population)] # remove extra if there is more than the permited..   

        next_gen += new_individuals[:n_pc]
        individuals_to_mutate = new_individuals[n_pc:]
        assert len(individuals_to_mutate) == n_pm

        new_individuals = []

        # Mutation (should not be inside the Forest class ?)
        for f in individuals_to_mutate:
            while True:

                idx = np.random.randint(n_classes)

                t = f[0].trees[idx]

                t_mutated = Tree.mutate(t)

                f[0].trees[idx] = t_mutated

                fit = evaluate_forest(f[0], X_train, Y_train)

                if fit > f[1]:  
                    new_individuals.append([copy.deepcopy(f[0]), fit])
                    break
        # end of mutation

        assert len(new_individuals) == n_pm
        next_gen += new_individuals
        assert len(next_gen) == k

        population = next_gen

    population = evaluate_population(population, X_train, Y_train)
    for i in range(k):
        record_fitness[epochs, i] = population[i][1]

    print('max: ', population[0])

    # for e, f in enumerate(population):
    #     f[0].build_visualization(e)

    fig, ax = plt.subplots(nrows=k, ncols=1, sharex=True, gridspec_kw={'hspace': 0})

    for e, row in enumerate(ax):
        row.plot(np.arange(epochs+1), record_fitness[:, e])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    main()