import tree
import markov

import numpy as np

def bin_tree_build(affinity,bal_constant=1.0):
    """
    Takes a static, square, symmetric nxn affinity on n nodes and 
    applies the second eigenvector binary cut algorithm to it.
    """
    
    _,n = affinity.shape

    root = tree.ClusterTreeNode(range(n))
    queue = [root]

    while max([x.size for x in queue]) > 1:
        new_queue = []
        for node in queue:
            if node.size > 2:
                #cut it
                cut = dyadic_eigen_cut(node,affinity,bal_constant)
                node.create_subclusters(cut)
            else:
                #make the singletons
                node.create_subclusters(np.arange(node.size))
            new_queue.extend(node.children)
        queue = new_queue

    root.make_index()                
    return root    

def dyadic_eigen_cut(node,affinity,bal_constant):
    """
    Returns the cut of the affinity matrix corresponding to the elements
    in node, under the condition of bal_constant.
    """ 
    new_data = affinity[node.elements,:][:,node.elements]
    
    vecs,_ = markov.markov_eigs(new_data, 2)
    eig = vecs[:,1]
    eig_sorted = eig.argsort().argsort()
    cut_loc = node.size/2
    labels = eig_sorted < cut_loc
    
    return labels
    
    
    