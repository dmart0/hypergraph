import numpy as np
import itertools
import random
import tensorly as tl
from sparserc import sparserc_solver
from ortho import orthogonal_decomposition
import matplotlib.pyplot as plt
import matlab.engine
from scipy.linalg import polar
from scipy.linalg import orth


class Hypergraph:
    def __init__(self, n):
        self.n = n # number of nodes
        self.adj = [] # list of edges, a[i][j] = 1 <=> hyperedge i contains node j, otherwise 0
        self.m = 0 # number of edges
        self.mce = 0 # number of nodes in largest edge
        self.tensor = np.zeros(0)
        self.weights = np.zeros(0)
        self.factor_matrix = np.zeros(0)
        self.signal = np.zeros(n)

    def add_edge(self, nodes=set(), mce=-1, num_nodes=-1):

        if mce < 0:
            mce = self.n

        if isinstance(nodes, int):
            nodes = {nodes}

        new_edge = [0] * self.n
        new_edge = np.array(new_edge)

        if len(nodes) == 0:
            #random edge
            while True:

                # How many nodes are in each edge
                number_nonzero = random.randint(2, mce)
                if num_nodes > 0:
                    number_nonzero = num_nodes

                indices = np.random.choice(self.n, size=number_nonzero, replace=False)
                new_edge[indices] = 1
                duplicate = False
                for row in self.adj:
                    if np.array_equal(new_edge, row):
                        duplicate = True
                        continue

                if (duplicate):
                    new_edge = np.zeros(self.n, dtype='int')
                    continue
                self.mce = max(self.mce, number_nonzero)
                break
        else:
            # custom edge
            for node in nodes:
                new_edge[node-1] = 1 # -1 since in foundations of HGSP numbering starts with 1
            self.mce = max(self.mce, len(nodes))

        self.adj.append(new_edge)

        self.m += 1

    def random_edges(self, number_of_edges, mce=-1):
        if mce < 0:
            mce = self.n
        for i in range(0, number_of_edges):
            self.add_edge(mce=mce)

    def add_edges(self, *edges):
        for edge in edges:
            self.add_edge(edge)

    def get_tensor(self):

        shape = (self.n,) * self.mce
        new_tensor = np.zeros(shape, dtype='float')

        for row in self.adj:
            nodes_in_edge = np.nonzero(row)[0]
            permutations = itertools.product(nodes_in_edge, repeat=self.mce)
            result = [p for p in permutations if set(p) == set(nodes_in_edge)]
            value = 1.0 * len(nodes_in_edge) / len(result)
            for idx in result:
                new_tensor[idx] = value

        self.tensor = new_tensor


    def reconstruct_tensor(self, V, weights=None):
        # cheating a little bit because we know mce
        mce = self.mce
        n = self.n

        # eigenvalues here, default is 1/i
        if weights is None:
            weights = [1 / i for i in range(1, n+1)]

        first_vec = V[:, 0]
        result = first_vec
        for _ in range(mce-1):
            result = np.outer(result, first_vec).reshape(*result.shape, -1)
        F = weights[0] * result
        for i in range(1, n):
            current_vec = V[:, i]
            result = current_vec
            for _ in range(mce-1):
                result = np.outer(result, current_vec).reshape(*result.shape, -1)
            F = F + weights[i] * result
        return F


# Example from paper
##hg = Hypergraph(7)
##hg.add_edges((1,4,6), (2,3,7), (5,6,7))

# Small example
#hg = Hypergraph(4)
#hg.add_edges((1,2), (2,3), (3,4), (1,2,3))#, (1,2,3,4))

# Empty example
#hg = Hypergraph(10)
