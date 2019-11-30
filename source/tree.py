import numpy as np
import operator
import random
from graphviz import Graph


class Tree:
    
    def __init__(self, input_size, depth=6):  # como saber ao certo a profundidade ?

        self.input_size = input_size
        D.input_size = input_size
        self.root = P(1, depth)

        # TODO: Here, we need to keep track to all nodes and its types, because
        # for the crossover, only nodes with the same type can be swapped

    def forward(self, input_d):

        # never forget to input the tree
        assert input_d.shape == (self.input_size,)
        D.input_reference = input_d
        return self.root.forward()


    def build_visualization(self):

        viz = Graph(comment='Tree')
        viz.node('O', 'O')
        self.root.build_visualization(viz, 'O')
        # print(viz.source)     
        viz.render('graphviz-output/tree.gv')



class P:

    # count for visualization
    node_counter = 0

    def __init__(self, depth, max_depth):
        # print(depth, max_depth)
        assert depth <= max_depth
        self.left_child = W(depth + 1, max_depth)
        self.right_child = W(depth + 1, max_depth)

    # TODO: very confusing how is the forward of P, need to check this
    def forward(self):
        left_child_out = self.left_child.forward()
        right_child_out = self.right_child.forward()

        return self._sigmoid(left_child_out + right_child_out)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    
    def build_visualization(self, viz, parent_id):

        node_id = 'P{}'.format(P.node_counter)
        viz.node(node_id, node_id)
        P.node_counter += 1
        viz.edge(parent_id, node_id)
        self.left_child.build_visualization(viz, node_id)
        self.right_child.build_visualization(viz, node_id)



class W:

    # count for visualization
    node_counter = 0

    def __init__(self, depth, max_depth):
        # print(depth, max_depth)
        assert depth <= max_depth
        if depth + 1 == max_depth: # I think that is right.. but there is nothing saying this in the paper
            self.left_child = F(depth + 1, max_depth)
            self.right_child = D(depth + 1, max_depth)

        elif depth + 2 == max_depth: # I think that is right.. but there is nothing saying this in the paper
            self.left_child = random.choice([F, A])(depth + 1, max_depth)
            self.right_child = D(depth + 1, max_depth)

        else:
            self.left_child = random.choice([F, A])(depth + 1, max_depth)
            self.right_child = random.choice([D, P])(depth + 1, max_depth)


    def forward(self):
        return self.left_child.forward() * self.right_child.forward()

    def build_visualization(self, viz, parent_id):

        node_id = 'W{}'.format(W.node_counter)
        viz.node(node_id, node_id)
        W.node_counter += 1
        viz.edge(parent_id, node_id)
        self.left_child.build_visualization(viz, node_id)
        self.right_child.build_visualization(viz, node_id)


class A:

    # count for visualization
    node_counter = 0

    def __init__(self, depth, max_depth):
        # print(depth, max_depth)
        assert depth <= max_depth

        if depth + 1 == max_depth: # I think that is right.. but there is nothing saying this in the paper
            self.left_child = F(depth + 1, max_depth)
            self.right_child = F(depth + 1, max_depth)

        else:
            self.left_child = random.choice([F, A])(depth + 1, max_depth)
            self.right_child = random.choice([F, A])(depth + 1, max_depth)

        ops = [('*', operator.mul), ('+', operator.add), ('-', operator.sub), ('%', operator.mod)]
        self.op_symbol, self.op = random.choice(ops)

    def forward(self):  
        return self.op(self.left_child.forward(), self.right_child.forward())


    def build_visualization(self, viz, parent_id):

        node_id = 'A{}({})'.format(A.node_counter, self.op_symbol)
        viz.node(node_id, node_id)
        A.node_counter += 1
        viz.edge(parent_id, node_id)
        self.left_child.build_visualization(viz, node_id)
        self.right_child.build_visualization(viz, node_id)


class F:

    # count for visualization
    node_counter = 0

    def __init__(self, depth, max_depth):
        # print(depth, max_depth)
        assert depth <= max_depth
        self.value = np.random.uniform(low=0.0, high=10.0)

    def forward(self):
        return self.value

    def build_visualization(self, viz, parent_id):

        node_id = 'F{}({})'.format(F.node_counter, str(self.value))
        viz.node(node_id, node_id)
        F.node_counter += 1
        viz.edge(parent_id, node_id)


class D:

    # count for visualization
    node_counter = 0

    input_size = None       # should be initialized before the first use
    input_reference = None  # should be initialized before the first use

    def __init__(self, depth, max_depth):
        # print(depth, max_depth)
        assert depth <= max_depth
        self.idx = random.choice(list(range(D.input_size))) # this is correct ? this is implementing also a feature selection like

    def forward(self):
        return D.input_reference[self.idx]

    def build_visualization(self, viz, parent_id):

        node_id = 'D{}({})'.format(D.node_counter, str(self.idx))
        viz.node(node_id, node_id)
        D.node_counter += 1
        viz.edge(parent_id, node_id)
