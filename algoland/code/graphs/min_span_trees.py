"""
MINIMUM SPANNING TREES
======================
Implementation of minimum spanning tree algorithms:

* Kruskal's algorithm
* Prim's algorithm

Given a weighted graph G = (V,E), __minimum spanning trees__ (MSTs) are trees where:
* the nodes of the tree are vertices in V (each appearing once in the MST);
* the edges v~w of the MST are edges v~w in E;
* the sum of the edge weights in the MST are minimal, i.e. there is no other tree
  with a lower total summed weight.

Let |E|/|V| be the "edge density" or inverse-sparsity of the graph; in general, you will
want to use:
* Kruskal's algorithm if you have low edge density, or sparse graphs; and
* Prim's algoritm if you have dense graphs.
The main reason is due to the differences in runtime complexity; see below.

KRUSKAL'S ALGORITHM
-------------------
The core idea is that this is a greedy algorithm which relies on sorting to
fetch the lowest-cost edges at each iteration, building the MST over a number
of iterations. The underlying data structure that Kruskal uses is a disjoint-set
structure to keep track of the vertices observed so far (so that we don't
redundantly add a single node multiple times).
1) Sort edges by weight into a list of edges S.
2) Until we construct a full MST, do:
  (a) fetch smallest weight edge from S;
  (b) 

Complexity: O(E log(V)) runtime, O(E+V) space.

PRIM'S ALGORITHM
----------------

Complexity: O(E + V log(V)) runtime, O(E+V) space.
"""
class Graph(object):
    """
    Dummy class representing a graph.
    """
    def __init__(self, adjacencies, vertices):
        self.vertices = vertices # set of vertices
        self.adjacencies = adjacencies # dict from vertices to list of vertices
        self.edges = [] # [ (v,w,weight) for (v~>w) in Edges ]
        for v,nbrs in self.adjacencies.items():
            for nbr,weight in nbrs:
                self.edges.append((v,nbr,weight))

class DisjointSet(object):
    """
    Implements disjoint set structure, used for union-find algos.
    """
    def __init__(self, values):
        pass

    def same_set(self, x, y):
        # Return True if x,y are merged within the same set.
        return (self.find(x) == self.find(y))

    def union(self, x, y):
        pass # TODO

    def find(self, x):
        # Return representative of the set that holds x.
        pass # TODO

                
def kruskal(graph):
    """
    Computes a minimum spanning tree on the input graph.
    """
    sorted_edges = sorted(graph.edges, key=lambda tpl: tpl[-1], reversed=True)
    tot_cost = 0
    dset = DisjointSet(graph.vertices)
    mst = []
    
    for v,w,weight in sorted_edges:
        if not dset.same_set(v,w):
            tot_cost += weight
            mst.append((v,w,weight))
            dset.union(v,w)

    return tot_cost, mst


import heapq
def prim(graph, source):
    """
    Computes a minimum spanning tree on the input graph; compared to Kruskal, this is
    a bit more involved and requires a priority queue (or min-heap), but is better when
    you have a dense/non-sparse graph with tons of edges.
    """
    # put all the edges in a min-weight priority queue:
    edges = copy(graph.edges)
    # TODO: heapify w/r/t last value, the weight. heapq doesnt support custom comparators, so replace this with custom
    # priority queue implementation later on at some point.
    # Ideally, we should be using a fibonacci heap here.
    heapq.heapify(edges)

    # create set to track nodes we've already seen:
    visited = { v: False for v in graph.vertices }


    # TODO: FINISH THIS ALGORITHM
    tot_cost = 0
