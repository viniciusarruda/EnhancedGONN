import numpy as np
import operator
import random
from graphviz import Graph

tree_viz = Graph(comment='Tree')

class Tree:
    
    def __init__(self, input_size):

        self.input_d = np.zeros(input_size)
        self.root = self._gen_random_tree()

        # TODO: Here, we need to keep track to all nodes and its types, because
        # for the crossover, only nodes with the same type can be swapped

    def forward(self, input_d):

        # never forget to input the tree
        self._feed_input(input_d)
        output = self.root.forward()
        return output
        

    def _feed_input(self, input_d):
        self.input_d[:] = input_d[:] # keeps the same memory address

    
    def _gen_random_tree(self, depth=3):

        # tree from the paper gonn in the image fig2a
        # TODO: generate a random tree based on a given depth
        
        f0 = F(value=8.7)
        d1 = D(input_reference=self.input_d, idx=1)
        w1 = W(left_child=f0, right_child=d1)
        tree_viz.edge(w1.node_name, f0.node_name)  # just for visualization
        tree_viz.edge(w1.node_name, d1.node_name)  # just for visualization

        f1 = F(value=3.2)
        d0 = D(input_reference=self.input_d, idx=0)
        w0 = W(left_child=f1, right_child=d0)
        tree_viz.edge(w0.node_name, f1.node_name)  # just for visualization
        tree_viz.edge(w0.node_name, d0.node_name)  # just for visualization

        p1 = P(left_child=w0, right_child=w1)
        tree_viz.edge(p1.node_name, w0.node_name)  # just for visualization
        tree_viz.edge(p1.node_name, w1.node_name)  # just for visualization

        f2 = F(value=1.2)
        w3 = W(left_child=f2, right_child=p1)
        tree_viz.edge(w3.node_name, f2.node_name)  # just for visualization
        tree_viz.edge(w3.node_name, p1.node_name)  # just for visualization

        # ---

        f3 = F(value=7.4)
        d2 = D(input_reference=self.input_d, idx=2)
        w4 = W(left_child=f3, right_child=d2)
        tree_viz.edge(w4.node_name, f3.node_name)  # just for visualization
        tree_viz.edge(w4.node_name, d2.node_name)  # just for visualization

        f4 = F(value=7.0)
        d3 = D(input_reference=self.input_d, idx=3)
        w5 = W(left_child=f4, right_child=d3)
        tree_viz.edge(w5.node_name, f4.node_name)  # just for visualization
        tree_viz.edge(w5.node_name, d3.node_name)  # just for visualization

        p2 = P(left_child=w4, right_child=w5)
        tree_viz.edge(p2.node_name, w4.node_name)  # just for visualization
        tree_viz.edge(p2.node_name, w5.node_name)  # just for visualization

        f5 = F(value=0.5)
        f6 = F(value=9.1)
        a0 = A(left_child=f5, right_child=f6, op_symbol='*')
        tree_viz.edge(a0.node_name, f5.node_name)  # just for visualization
        tree_viz.edge(a0.node_name, f6.node_name)  # just for visualization

        w6 = W(left_child=a0, right_child=p2)
        tree_viz.edge(w6.node_name, a0.node_name)  # just for visualization
        tree_viz.edge(w6.node_name, p2.node_name)  # just for visualization

        # ---

        root = P(left_child=w3, right_child=w6)
        tree_viz.edge(root.node_name, w6.node_name)  # just for visualization
        tree_viz.edge(root.node_name, w3.node_name)  # just for visualization

        tree_viz.node('O', 'O')  # just for visualization
        tree_viz.edge('O', root.node_name)  # just for visualization

        # print(tree_viz.source)                 # just for visualization
        tree_viz.render('graphviz-output/tree.gv')   # just for visualization

        return root


class P:

    # count for visualization
    count = 0

    def __init__(self, left_child, right_child):

        assert isinstance(left_child, W)
        assert isinstance(right_child, W)

        self.left_child = left_child
        self.right_child = right_child

        # visualization code
        self.node_name = 'P{}'.format(P.count)
        tree_viz.node(self.node_name, self.node_name)
        P.count += 1
        # ------------------

    # TODO: very confusing how is the forward of P, need to check this
    def forward(self):
        left_child_out = self.left_child.forward()
        right_child_out = self.right_child.forward()

        return self._sigmoid(left_child_out + right_child_out)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


class W:

    # count for visualization
    count = 0

    def __init__(self, left_child, right_child):

        assert isinstance(left_child, F) or isinstance(left_child, A)
        assert isinstance(right_child, D) or isinstance(right_child, P)

        self.left_child = left_child
        self.right_child = right_child

        # visualization code
        self.node_name = 'W{}'.format(W.count)
        tree_viz.node(self.node_name, self.node_name)
        W.count += 1
        # ------------------

    def forward(self):
        return self.left_child.forward() * self.right_child.forward()


class A:

    # count for visualization
    count = 0

    def __init__(self, left_child, right_child, op_symbol=None):

        assert isinstance(left_child, F) or isinstance(left_child, A)
        assert isinstance(right_child, F) or isinstance(right_child, A)

        self.left_child = left_child
        self.right_child = right_child

        op_str = ['*', '+', '-', '%']
        ops = [operator.mul, operator.add, operator.sub, operator.mod]

        if op_symbol is None:
            op_idx = random.choice(list(range(4)))
            self.op = ops[op_idx]
            op_symbol = op_str[op_idx]
        else:
            assert op_symbol in op_str
            self.op = ops[op_str.index(op_symbol)]

        # visualization code
        self.node_name = 'A{}({})'.format(A.count, op_symbol)
        tree_viz.node(self.node_name, self.node_name)
        A.count += 1
        # ------------------

    def forward(self):  
        return self.op(self.left_child.forward(), self.right_child.forward())


class F:

    # count for visualization
    count = 0

    def __init__(self, value=None):

        if value is None:
            self.value = np.random.uniform(low=0.0, high=10.0)
        else:
            assert isinstance(value, float)
            self.value = value

        # visualization code
        self.node_name = 'F{}({})'.format(F.count, str(self.value))
        tree_viz.node(self.node_name, self.node_name)
        F.count += 1
        # ------------------

    def forward(self):
        return self.value


class D:

    # count for visualization
    count = 0

    def __init__(self, input_reference, idx):

        self.input_reference = input_reference
        self.idx = idx

        # visualization code
        self.node_name = 'D{}({})'.format(D.count, str(self.idx))
        tree_viz.node(self.node_name, self.node_name)
        D.count += 1
        # ------------------

    def forward(self):
        return self.input_reference[self.idx]