import numpy as np
from sklearn.model_selection import train_test_split
from tree import Tree
import random

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

def main():
    
    # random.seed(SEED)
    # np.random.seed(SEED)

    dataset = 'breastEW'

    X, Y = data_loader('datasets/{}.npy'.format(dataset))
    n_features, n_classes = {'breastEW':(30, 2), 'hepatitis':(19, 2), 'multiple_features':(649, 10), 'iris':(4, 3)}[dataset]  # I put iris here just to test but it was not in the first assignment so we do not need to include it in our results
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train = data_normalizer(X_train) # Yes, should be normalized after the train test split
    X_test = data_normalizer(X_test)

    # TOY
    # testar tbm importando tree_first ao inves de tree
    # t = Tree(input_size=4)
    # t.build_visualization()
    # print(t.forward(np.zeros(4)))
    # exit()

    k = 50

    trees_fitness = np.zeros(k)
    trees = [Tree(input_size=4) for _ in range(k)]

    for e, t in enumerate(trees):

        outputs = np.zeros(X_train.shape[0])

        for i in range(X_train.shape[0]):
            output = t.forward(X_train[i, :4]) # only using the first four features due to the fixed tree

            # TODO: Since I did not understand this part, I kept it simple.. need to check this (it is related on how is the forward of P)

            output = 1.0 if output > 0.5 else 0.0

            outputs[i] = output

        trees_fitness[e] = fitness(Y_train, outputs)

    print(trees_fitness)

    


if __name__ == "__main__":
    
    main()