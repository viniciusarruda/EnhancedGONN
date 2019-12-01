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


def evaluate_tree(tree, X_train, Y_train):

    outputs = np.zeros(X_train.shape[0])

    for i in range(X_train.shape[0]):
        output = tree.forward(X_train[i, :4]) # only using the first four features due to the fixed tree
        # TODO: Since I did not understand this part, I kept it simple.. need to check this (it is related on how is the forward of P)
        outputs[i] = 1.0 if output > 0.5 else 0.0

    return fitness(Y_train, outputs)

def evaluate_population(population, X_train, Y_train):
        
    for p in range(len(population)):
        population[p][1] = evaluate_tree(population[p][0], X_train, Y_train)

    return sorted(population, key=lambda x: x[1], reverse=True)


def main():
    
    random.seed(SEED)
    np.random.seed(SEED)

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
    epochs = 10
    pr = 0.2 # Pr% in the paper
    n_pr = int(k * pr)

    population = [[Tree(input_size=X_train.shape[1], depth=random.choice([3,4,5,6])), 0] for _ in range(k)]

    for e in range(epochs):

        population = evaluate_population(population, X_train, Y_train)

        # pop_fitness = [p[1] for p in population]
        # print(pop_fitness) # just to show

        next_gen = population[:n_pr]
        population = population[n_pr:]

        while len(next_gen) < k:

            idx_1, idx_2 = np.random.permutation(len(population))[:2]

            gen_1, gen_2 = Tree.crossover(population[idx_1][0], population[idx_2][0])

            f1 = evaluate_tree(gen_1, X_train, Y_train)
            f2 = evaluate_tree(gen_2, X_train, Y_train)

            if f1 > population[idx_1][1]:  # if not, is applied with the same parents or not ? I am changing the parents..
                next_gen.append([gen_1, f1])
            
            if f2 > population[idx_2][1]:
                next_gen.append([gen_2, f2])

        next_gen = next_gen[:k] # if there is more than k generated

        # Keep only best Pc% from next_gen
        # Apply mutation on worst Pm% and then, append them to next_gen..
        # acho que eh isso que entendi.. 

        population = next_gen



if __name__ == "__main__":
    
    main()