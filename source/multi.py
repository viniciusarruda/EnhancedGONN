import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tree import Tree
from tree import Forest
import random
import copy

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


def main():
    
    random.seed(SEED)
    np.random.seed(SEED)

    dataset = 'iris'

    X, Y = data_loader('datasets/{}.npy'.format(dataset))
    n_features, n_classes = {'breastEW':(30, 2), 'hepatitis':(19, 2), 'multiple_features':(649, 10), 'iris':(4, 3)}[dataset]  # I put iris here just to test but it was not in the first assignment so we do not need to include it in our results
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train = data_normalizer(X_train) # Yes, should be normalized after the train test split
    X_test = data_normalizer(X_test)

    if 2 < n_classes:
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        Y_train = enc.fit_transform([[l] for l in Y_train])
        Y_test = enc.fit_transform([[l] for l in Y_test])

    # TOY
    # testar tbm importando tree_first ao inves de tree
    # t = Tree(input_size=4)
    # t.build_visualization()
    # print(t.forward(np.zeros(4)))
    # exit()

    k = 50
    epochs = 10
    pr = 0.2 # Pr% in the paper
    n_pr = int(k * pr)
    depth = [3, 4, 5, 6]

    population = [[Forest(input_size=X_train.shape[1], depth=depth, n_classes=n_classes), 0.0] for _ in range(k)]

    for e in range(epochs):

        population = evaluate_population(population, X_train, Y_train)

        # pop_fitness = [p[1] for p in population]
        # print(pop_fitness) # just to show

        next_gen = population[:n_pr]
        population = population[n_pr:]

        while len(next_gen) < k:

            fidx_1, fidx_2 = np.random.permutation(len(population))[:2]

            ia = population[fidx_1]
            ib = population[fidx_2]

            idx = np.random.randint(n_classes)

            ta = ia[0].trees[idx]
            tb = ib[0].trees[idx]

            gen_1, gen_2 = Tree.crossover(ta, tb)

            ia[0].trees[idx] = gen_1
            ib[0].trees[idx] = gen_2

            f1 = evaluate_forest(ia[0], X_train, Y_train)
            f2 = evaluate_forest(ib[0], X_train, Y_train)

            if f1 > ia[1]:  # if not, is applied with the same parents or not ? I am changing the parents..
                next_gen.append([copy.deepcopy(ia[0]), f1])
            
            if f2 > ib[1]:
                next_gen.append([copy.deepcopy(ib[0]), f2])

            # another possibility
            ia[0].trees[idx] = gen_2
            ib[0].trees[idx] = gen_1

            f3 = evaluate_forest(ia[0], X_train, Y_train)
            f4 = evaluate_forest(ib[0], X_train, Y_train)

            if f3 > ia[1]:
                next_gen.append([copy.deepcopy(ia[0]), f3])

            if f4 > ib[1]:
                next_gen.append([copy.deepcopy(ib[0]), f4])

            # revert
            ia[0].trees[idx] = ta
            ib[0].trees[idx] = tb

        next_gen = next_gen[:k] # if there is more than k generated

        # Keep only best Pc% from next_gen
        # Apply mutation on worst Pm% and then, append them to next_gen..
        # acho que eh isso que entendi.. 

        population = next_gen

    print('max: ', population[0])

if __name__ == "__main__":
    
    main()