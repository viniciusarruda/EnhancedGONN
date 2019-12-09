import os
import numpy as np
import random
from graphviz import Digraph
import copy
import scipy.special

OBJ_IDS = 0

def get_id():
    global OBJ_IDS
    OBJ_IDS += 1
    return str(OBJ_IDS)


def copy_obj(obj):
    new_obj = copy.deepcopy(obj)
    new_obj.update_copy()
    return new_obj


class NotExpectedError(Exception):
    pass


class Forest:

    def __init__(self, input_size, n_classes):
        self.n_classes = n_classes
        self.trees = [Tree(input_size=input_size, depth=np.random.choice([3, 4, 5, 6])) for _ in range(n_classes)]
        self.outputs = np.zeros(n_classes)
        self.forest_id = get_id()

    def __del__(self):
        for t in self.trees:
            del t
        del self

    def forward(self, input_d):
        for p in range(self.n_classes):
            self.outputs[p] = self.trees[p].forward(input_d)
        return self.outputs

    def build_visualization(self, folder_name):
        os.makedirs(folder_name)
        for e, t in enumerate(self.trees):
            t.build_visualization(e, folder_name)

    def get_number_used_features(self):
        feature_idxs = []
        for t in self.trees:
            feature_idxs += t.get_number_used_features()

        feature_dict = {}
        for item in feature_idxs:
            try:
                feature_dict[item] += 1
            except:
                feature_dict[item] = 1

        return feature_dict

    def update_copy(self):
        self.forest_id = get_id()
        for t in self.trees:
            t.update_copy()

class Tree:

    def __init__(self, input_size, depth):

        self.input_size = input_size
        D.input_size = input_size
        self.root = P(1, depth, self)
        self.tree_id = get_id()

    def __del__(self):
        del self.root
        del self

    def forward(self, input_d):
        # never forget to input the tree
        assert input_d.shape == (self.input_size,)
        D.input_reference = input_d
        # return 1.0 if 0.5 < self.root.forward() else 0.0
        return self.root.forward()

    def build_visualization(self, class_idx, folder):
        viz = Digraph(comment='Tree')
        viz.node(self.tree_id, 'O')
        self.root.build_visualization(viz, self.tree_id)
        # print(viz.source)
        viz.render('{}tree_class_{}_tree_id_{}.gv'.format(folder, class_idx, self.tree_id))

    def build_nodes(self):
        self.nodes = {'P':[], 'W':[], 'A':[], 'F':[], 'D':[]}
        self.root.build_nodes(self.nodes)

    def get_number_used_features(self):
        self.build_nodes()
        return [d.idx for d in self.nodes['D']]

    def update_copy(self):
        self.tree_id = get_id()
        self.root.update_copy()

    @staticmethod
    def crossover(p1, p2):

        t1, t2 = copy_obj(p1), copy_obj(p2)

        # choosing the node_type to crossover
        t1.build_nodes() # do not forget to reconstruct this before this operation
        t2.build_nodes() # do not forget to reconstruct this before this operation

        existing_nodes_t1 = [key for key in t1.nodes if len(t1.nodes[key]) > 0]
        existing_nodes_t2 = [key for key in t2.nodes if len(t2.nodes[key]) > 0]
        existing_nodes = list(set(existing_nodes_t1) & set(existing_nodes_t2))

        function_set = [node_type for node_type in ['P', 'W', 'A'] if node_type in existing_nodes]
        terminal_set = [node_type for node_type in ['F', 'D'] if node_type in existing_nodes]

        if len(function_set) > 0 and np.random.uniform() < 0.9:
            node_type = random.choice(function_set)
        elif len(terminal_set) > 0:
            node_type = random.choice(terminal_set)
        else:
            raise NotExpectedError()
        # node_type chosen

        idx_1 = np.random.randint(low=0, high=len(t1.nodes[node_type]))
        idx_2 = np.random.randint(low=0, high=len(t2.nodes[node_type]))

        t1_node, t2_node = t1.nodes[node_type][idx_1], t2.nodes[node_type][idx_2]

        t2_node_parent = t2_node.parent

        
        if isinstance(t1_node.parent, Tree):
            t1_node.parent.root = t2_node
        elif t1_node.parent.left_child.node_id == t1_node.node_id:
            t1_node.parent.left_child = t2_node
        elif t1_node.parent.right_child.node_id == t1_node.node_id:
            t1_node.parent.right_child = t2_node
        else:
            msg = '{}, {}, {}, {}'.format(node_type, t1_node.parent.left_child.node_id, t1_node.parent.right_child.node_id, t1_node.node_id)
            raise NotExpectedError(msg)
        t2_node.parent = t1_node.parent

        if isinstance(t2_node_parent, Tree): 
            t2_node_parent.root = t1_node
        elif t2_node_parent.left_child.node_id == t2_node.node_id:
            t2_node_parent.left_child = t1_node
        elif t2_node_parent.right_child.node_id == t2_node.node_id:
            t2_node_parent.right_child = t1_node
        else:
            msg = '{}, {}, {}, {}'.format(node_type, t2_node_parent.left_child.node_id, t2_node_parent.right_child.node_id, t2_node.node_id)
            raise NotExpectedError(msg)
        t1_node.parent = t2_node_parent

        return t1, t2


    @staticmethod
    def mutate(t):

        t = copy_obj(t)

        # choosing the node_type to mutate
        t.build_nodes() # do not forget to reconstruct this before this operation

        existing_nodes = [key for key in t.nodes if len(t.nodes[key]) > 0]

        function_set = [node_type for node_type in ['P', 'W', 'A'] if node_type in existing_nodes]
        terminal_set = [node_type for node_type in ['F', 'D'] if node_type in existing_nodes]

        if len(terminal_set) > 0 and np.random.uniform() < 0.9:
            node_type = random.choice(terminal_set)
        elif len(function_set) > 0:
            node_type = random.choice(function_set)
        else:
            raise NotExpectedError()
        # node_type chosen

        idx = np.random.randint(low=0, high=len(t.nodes[node_type]))
        t_node = t.nodes[node_type][idx]

        max_depth = np.random.choice([3, 4, 5, 6])
        new_sub_tree = {'P': P, 'W': W, 'A': A, 'F': F, 'D': D}[node_type](1, max_depth, t_node.parent)

        if isinstance(t_node.parent, Tree):
            t_node.parent.root = new_sub_tree
        elif t_node.parent.left_child.node_id == t_node.node_id:
            t_node.parent.left_child = new_sub_tree
        elif t_node.parent.right_child.node_id == t_node.node_id:
            t_node.parent.right_child = new_sub_tree
        else:
            msg = '{}, {}, {}, {}'.format(node_type, t_node.parent.left_child.node_id, t_node.parent.right_child.node_id, t_node.node_id)
            raise NotExpectedError(msg)
        # new_sub_tree.parent = t_node.parent # already setted on creation

        return t


class P:

    def __init__(self, depth, max_depth, parent):
        assert depth <= max_depth
        self.parent = parent
        self.left_child = W(depth + 1, max_depth, self)
        self.right_child = W(depth + 1, max_depth, self)
        self.node_id = get_id()

    def __del__(self):
        del self.left_child
        del self.right_child
        del self

    def forward(self):
        left_child_out = self.left_child.forward()
        right_child_out = self.right_child.forward()
        return self._sigmoid(left_child_out + right_child_out)

    def _sigmoid(self, x):
        return scipy.special.expit(x) # safer than 1.0 / (1.0 + np.exp(-x))

    def build_visualization(self, viz, parent_id):
        viz.node(self.node_id, 'P')
        viz.edge(parent_id, self.node_id)
        self.left_child.build_visualization(viz, self.node_id)
        self.right_child.build_visualization(viz, self.node_id)

    def build_nodes(self, nodes):
        nodes['P'].append(self)
        self.left_child.build_nodes(nodes)
        self.right_child.build_nodes(nodes)

    def update_copy(self):
        self.node_id = get_id()
        self.left_child.update_copy()
        self.right_child.update_copy()


class W:

    def __init__(self, depth, max_depth, parent):
        assert depth <= max_depth
        self.parent = parent
        self.node_id = get_id()

        if depth + 1 == max_depth: 
            self.left_child = F(depth + 1, max_depth, self)
            self.right_child = D(depth + 1, max_depth, self)

        elif depth + 2 == max_depth: 
            self.left_child = random.choice([F, A])(depth + 1, max_depth, self)
            self.right_child = D(depth + 1, max_depth, self)

        else:
            self.left_child = random.choice([F, A])(depth + 1, max_depth, self)
            self.right_child = random.choice([D, P])(depth + 1, max_depth, self)

        self.visited = False

    def __del__(self):
        del self.left_child
        del self.right_child
        del self

    def forward(self):
        return self.left_child.forward() * self.right_child.forward()

    def build_visualization(self, viz, parent_id):
        viz.node(self.node_id, 'W')
        viz.edge(parent_id, self.node_id)
        self.left_child.build_visualization(viz, self.node_id)
        self.right_child.build_visualization(viz, self.node_id)

    def build_nodes(self, nodes):
        nodes['W'].append(self)
        self.left_child.build_nodes(nodes)
        self.right_child.build_nodes(nodes)

    def update_copy(self):
        self.node_id = get_id()
        self.left_child.update_copy()
        self.right_child.update_copy()


class A:

    def __init__(self, depth, max_depth, parent):
        assert depth <= max_depth
        self.parent = parent
        self.node_id = get_id()

        if depth + 1 == max_depth:
            self.left_child = F(depth + 1, max_depth, self)
            self.right_child = F(depth + 1, max_depth, self)
        else:
            self.left_child = random.choice([F, A])(depth + 1, max_depth, self)
            self.right_child = random.choice([F, A])(depth + 1, max_depth, self)

        ops = [('*', lambda a,b: a * b),
               ('+', lambda a,b: a + b),
               ('-', lambda a,b: a - b),
               ('%', lambda a,b: 0.0 if 1.0e-15 >= np.abs(b) else a / b)]

        self.op_symbol, self.op = random.choice(ops)

    def __del__(self):
        del self.left_child
        del self.right_child
        del self

    def forward(self):
        return self.op(self.left_child.forward(), self.right_child.forward())

    def build_visualization(self, viz, parent_id):
        viz.node(self.node_id, self.op_symbol)
        viz.edge(parent_id, self.node_id)
        self.left_child.build_visualization(viz, self.node_id)
        self.right_child.build_visualization(viz, self.node_id)

    def build_nodes(self, nodes):
        nodes['A'].append(self)
        self.left_child.build_nodes(nodes)
        self.right_child.build_nodes(nodes)

    def update_copy(self):
        self.node_id = get_id()
        self.left_child.update_copy()
        self.right_child.update_copy()



class F:

    def __init__(self, depth, max_depth, parent):
        assert depth <= max_depth
        self.parent = parent
        self.node_id = get_id()
        self.value = np.random.uniform(low=0.0, high=10.0)

    def forward(self):
        return self.value

    def build_visualization(self, viz, parent_id):
        viz.node(self.node_id, '{:.2f}'.format(self.value))
        viz.edge(parent_id, self.node_id)

    def build_nodes(self, nodes):
        nodes['F'].append(self)

    def update_copy(self):
        self.node_id = get_id()


class D:

    input_size = None       # should be initialized before the first use
    input_reference = None  # should be initialized before the first use

    def __init__(self, depth, max_depth, parent):
        assert depth <= max_depth
        self.parent = parent
        self.node_id = get_id()
        self.idx = random.choice(list(range(D.input_size)))

    def forward(self):
        return D.input_reference[self.idx]

    def build_visualization(self, viz, parent_id):
        viz.node(self.node_id, 'D{}'.format(self.idx))
        viz.edge(parent_id, self.node_id)

    def build_nodes(self, nodes):
        nodes['D'].append(self)

    def update_copy(self):
        self.node_id = get_id()
