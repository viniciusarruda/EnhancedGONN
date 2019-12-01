import numpy as np
import operator
import random
from graphviz import Graph
import copy


class Tree:

    # just for visualization
    tree_counter = 0
    
    def __init__(self, input_size, depth=6):  # como saber ao certo a profundidade ?

        self.input_size = input_size
        D.input_size = input_size
        self.root = P(1, depth, self)

        self.tree_id = Tree.tree_counter
        Tree.tree_counter += 1

        # TODO: Here, we need to keep track to all nodes and its types, because
        # for the crossover, only nodes with the same type can be swapped

    def forward(self, input_d):

        # never forget to input the tree
        assert input_d.shape == (self.input_size,)
        D.input_reference = input_d
        return self.root.forward()


    def build_visualization(self):

        viz = Graph(comment='Tree')
        node_id = 'O({})'.format(self.tree_id)
        viz.node(node_id, node_id)
        self.root.build_visualization(viz, node_id)
        # print(viz.source)     
        viz.render('graphviz-output/tree_{}.gv'.format(self.tree_id))

    def build_nodes(self):
        self.nodes = {'P':[], 'W':[], 'A':[], 'F':[], 'D':[]}
        self.root.build_nodes(self.nodes)
    

    @staticmethod
    def crossover(t1, t2):

        t1, t2 = copy.deepcopy(t1), copy.deepcopy(t2)
        
        if np.random.uniform() < 0.9:
            node_type = random.choice(['P', 'W', 'A'])
        else:
            node_type = random.choice(['F', 'D'])

        node_type = 'P'

        # should be optimized, but do not spend your time with this... we are late
        t1.build_nodes() # do not forget to reconstruct this before this operation
        t2.build_nodes() # do not forget to reconstruct this before this operation
        assert len(t1.nodes[node_type]) > 0 and len(t2.nodes[node_type]) > 0

        idx_1 = np.random.randint(low=0, high=len(t1.nodes[node_type]))
        idx_2 = np.random.randint(low=0, high=len(t2.nodes[node_type]))

        t1_node, t2_node = t1.nodes[node_type][idx_1], t2.nodes[node_type][idx_2]

        t2_node_parent = t2_node.parent

        if node_type == 'P' and isinstance(t1_node.parent, Tree): # if the parent is a Tree, of course the node_type is 'P'
            t1_node.parent.root = t2_node
        elif t1_node.parent.left_child == t1_node:
            t1_node.parent.left_child = t2_node
        else:
            t1_node.parent.right_child = t2_node
        t2_node.parent = t1_node.parent

        if node_type == 'P' and isinstance(t2_node_parent, Tree): # if the parent is a Tree, of course the node_type is 'P'
            t2_node_parent.root = t1_node
        elif t2_node_parent.left_child == t2_node:
            t2_node_parent.left_child = t1_node
        else:
            t2_node_parent.right_child = t1_node
        t1_node.parent = t2_node_parent

        return t1, t2    
    


class P:

    # count for visualization
    node_counter = 0

    def __init__(self, depth, max_depth, parent):
        # print(depth, max_depth)
        assert depth <= max_depth
        self.parent = parent
        self.left_child = W(depth + 1, max_depth, self)
        self.right_child = W(depth + 1, max_depth, self)
        self.node_id = P.node_counter
        P.node_counter += 1

    # TODO: very confusing how is the forward of P, need to check this
    def forward(self):
        left_child_out = self.left_child.forward()
        right_child_out = self.right_child.forward()

        return self._sigmoid(left_child_out + right_child_out)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    
    def build_visualization(self, viz, parent_id):

        node_id = 'P{}'.format(self.node_id)
        viz.node(node_id, node_id)
        viz.edge(parent_id, node_id)
        self.left_child.build_visualization(viz, node_id)
        self.right_child.build_visualization(viz, node_id)

    def build_nodes(self, nodes):
        nodes['P'].append(self)
        self.left_child.build_nodes(nodes)
        self.right_child.build_nodes(nodes)




class W:

    # count for visualization
    node_counter = 0

    def __init__(self, depth, max_depth, parent):
        # print(depth, max_depth)
        assert depth <= max_depth
        self.parent = parent
        self.node_id = W.node_counter
        W.node_counter += 1

        if depth + 1 == max_depth: # I think that is right.. but there is nothing saying this in the paper
            self.left_child = F(depth + 1, max_depth, self)
            self.right_child = D(depth + 1, max_depth, self)

        elif depth + 2 == max_depth: # I think that is right.. but there is nothing saying this in the paper
            self.left_child = random.choice([F, A])(depth + 1, max_depth, self)
            self.right_child = D(depth + 1, max_depth, self)

        else:
            self.left_child = random.choice([F, A])(depth + 1, max_depth, self)
            self.right_child = random.choice([D, P])(depth + 1, max_depth, self)


    def forward(self):
        return self.left_child.forward() * self.right_child.forward()

    def build_visualization(self, viz, parent_id):

        node_id = 'W{}'.format(self.node_id)
        viz.node(node_id, node_id)
        viz.edge(parent_id, node_id)
        self.left_child.build_visualization(viz, node_id)
        self.right_child.build_visualization(viz, node_id)

    def build_nodes(self, nodes):
        nodes['W'].append(self)
        self.left_child.build_nodes(nodes)
        self.right_child.build_nodes(nodes)


class A:

    # count for visualization
    node_counter = 0

    def __init__(self, depth, max_depth, parent):
        # print(depth, max_depth)
        assert depth <= max_depth
        self.parent = parent
        self.node_id = A.node_counter
        A.node_counter += 1

        if depth + 1 == max_depth: # I think that is right.. but there is nothing saying this in the paper
            self.left_child = F(depth + 1, max_depth, self)
            self.right_child = F(depth + 1, max_depth, self)

        else:
            self.left_child = random.choice([F, A])(depth + 1, max_depth, self)
            self.right_child = random.choice([F, A])(depth + 1, max_depth, self)

        ops = [('*', operator.mul), ('+', operator.add), ('-', operator.sub), ('%', self.div)]
        self.op_symbol, self.op = random.choice(ops)

    def div(self, dividend, divisor):
        if 1.0e-15 >= abs(divisor):
            return 0
        else:
            return dividend / divisor

    def forward(self):  
        return self.op(self.left_child.forward(), self.right_child.forward())


    def build_visualization(self, viz, parent_id):

        node_id = 'A{}({})'.format(self.node_id, self.op_symbol)
        viz.node(node_id, node_id)
        viz.edge(parent_id, node_id)
        self.left_child.build_visualization(viz, node_id)
        self.right_child.build_visualization(viz, node_id)

    def build_nodes(self, nodes):
        nodes['A'].append(self)
        self.left_child.build_nodes(nodes)
        self.right_child.build_nodes(nodes)


class F:

    # count for visualization
    node_counter = 0

    def __init__(self, depth, max_depth, parent):
        # print(depth, max_depth)
        assert depth <= max_depth
        self.parent = parent
        self.node_id = F.node_counter
        F.node_counter += 1
        self.value = np.random.uniform(low=0.0, high=10.0)

    def forward(self):
        return self.value

    def build_visualization(self, viz, parent_id):

        node_id = 'F{}({})'.format(self.node_id, str(self.value))
        viz.node(node_id, node_id)
        viz.edge(parent_id, node_id)

    def build_nodes(self, nodes):
        nodes['F'].append(self)

class D:

    # count for visualization
    node_counter = 0

    input_size = None       # should be initialized before the first use
    input_reference = None  # should be initialized before the first use

    def __init__(self, depth, max_depth, parent):
        # print(depth, max_depth)
        assert depth <= max_depth   
        self.parent = parent
        self.node_id = D.node_counter
        D.node_counter += 1
        self.idx = random.choice(list(range(D.input_size))) # this is correct ? this is implementing also a feature selection like

    def forward(self):
        return D.input_reference[self.idx]

    def build_visualization(self, viz, parent_id):

        node_id = 'D{}({})'.format(self.node_id, str(self.idx))
        viz.node(node_id, node_id)
        viz.edge(parent_id, node_id)

    def build_nodes(self, nodes):
        nodes['D'].append(self)
