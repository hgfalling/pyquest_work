import tree
import numpy as np
import scipy.cluster as spcl
import scipy.sparse.linalg as spsl

"""
Cut functions are of the form:

cut_function(tree_node,data,**kwargs):
tree_node is the node whose n elements we are to cut into pieces.
data is the overall database, so we subselect data[tree_node.elements,:] often.
kwargs contains additional necessary information for the cut function.

The returned value is a numpy array of n integers from 0 to the number of 
clusters indicating the cluster than the ith node element belongs to.
"""

def kmeans_tree(eigvecs,eigvals,max_levels=0):
    
    if np.std(eigvecs[:,0]) == 0.0:
        eigvecs = eigvecs[:,1:]
        eigvals = eigvals[1:]
        
    d,n = np.shape(eigvecs)
    
    if max_levels == 0:
        max_levels = min(n,np.floor(np.log(d/5.0)/np.log(2.0))+2)

    vecs = eigvecs.dot(np.diag(eigvals))

    root = tree.ClusterTreeNode(range(d))
    queue = [root]

    while max([x.size for x in queue]) > 1:
        #work on one level at a time to match how the matlab trees are created
        new_queue = []
        for node in queue:
            if node.size >= 4 and node.level + 1 < max_levels:
                #cut it
                cut = kmeans_cut(node,vecs)
                node.create_subclusters(cut)
            else:
                #make the singletons
                node.create_subclusters(np.arange(node.size))
            new_queue.extend(node.children)
        queue = new_queue

    root.make_index()                
    return root

def median_tree(eigvecs,eigvals,max_levels=0):
    
    if np.std(eigvecs[:,0]) == 0.0:
        eigvecs = eigvecs[:,1:]
        eigvals = eigvals[1:]
        
    d,n = np.shape(eigvecs)
    
    if max_levels == 0:
        max_levels = min(n,np.floor(np.log(d/5.0)/np.log(2.0))+2)

    vecs = eigvecs.dot(np.diag(eigvals))

    root = tree.ClusterTreeNode(range(d))
    queue = [root]

    while max([x.size for x in queue]) > 1:
        #work on one level at a time to match how the matlab trees are created
        new_queue = []
        for node in queue:
            if node.size >= 4 and node.level + 1 < max_levels:
                #cut it
                cut = median_cut(node,vecs)
                node.create_subclusters(cut)
            else:
                #make the singletons
                node.create_subclusters(np.arange(node.size))
            new_queue.extend(node.children)
        queue = new_queue

    root.make_index()                
    return root

def eigen_tree(data,row_tree,affinity_function,alpha=1.0,beta=0.0,**kwargs):
    
    d,n = np.shape(data)
    
    kwargs["row_tree"] = row_tree
    
    if "max_levels" in kwargs:
        max_levels = kwargs["max_levels"]
    else:
        max_levels = np.floor(np.log(n)/np.log(2.0))


    root = tree.ClusterTreeNode(range(n))
    queue = [root]

    while max([x.size for x in queue]) > 1:
        #work on one level at a time to match how the matlab trees are created
        new_queue = []
        for node in queue:
            if node.size >= 4 and node.level + 1 < max_levels:
                #cut it
                cut = eigen_cut(node,data,affinity_function,**kwargs)
                node.create_subclusters(cut)
            else:
                #make the singletons
                node.create_subclusters(np.arange(node.size))
            new_queue.extend(node.children)
        queue = new_queue

    root.make_index()                
    return root

def make_markov(data,thres=1e-8):
    d_mat = data*(data > thres)
    rowsums = np.sum(d_mat,axis=1) + 1e-15
    p_mat = d_mat/(np.outer(rowsums,rowsums))
    d_mat2 = np.sqrt(np.sum(p_mat,axis=1)) + 1e-15
    p_mat = p_mat/(np.outer(d_mat2,d_mat2))
    return p_mat

def markov_eigs(data,n_eigs,normalize=True,thres=1e-8):
    n = np.shape(data)[0]
    n_eigs = min(n_eigs,n)
    p_mat = make_markov(data,thres)
    [vectors,singvals,_] = spsl.svds(p_mat,n_eigs)
    y = np.argsort(-singvals)
    eigenvalues = singvals[y]
    eigenvectors = vectors[:,y]
    
    if normalize:
        n_mat = np.hstack([np.reshape([eigenvectors[:,0]],[-1,1])]*n_eigs)
        eigenvectors /= n_mat
        n_mat2 = np.vstack([np.sign(eigenvectors[0,1:])]*n)
        n_mat2[n_mat2==0] = 1.0
        eigenvectors[:,1:] *= n_mat2
        
    return eigenvectors, eigenvalues

def kmeans_cut(node,data,k=2):
    
    cut_data = spcl.vq.whiten(data[node.elements,:])
    _,labels = spcl.vq.kmeans2(cut_data,k,iter=20,thresh=1e-10,minit='points')

    return labels

def median_cut(node,data):
    cut_data = data[node.elements,node.level-1]
    cut_loc = np.median(cut_data)
    labels = np.ones(len(node.elements))*(cut_data > cut_loc)

    return labels

def eigen_cut(node,data,affinity_function,noise=0.0,**kwargs):
    if "emd" in kwargs:
        kwargs["emd"] = kwargs["emd"][node.elements,:][:,node.elements]
    else:
        kwargs["data"] = data[:,node.elements]
    affinity = affinity_function(**kwargs)
    try:
        vecs,_ = markov_eigs(affinity,2)
    except:
        print affinity
        print kwargs["emd"]
        print node.elements
        raise
    eig = vecs[:,1]
    eig_sorted = np.sort(eig)
    n = len(eig_sorted)
    rnoise = np.random.uniform(-noise,noise)
    if noise == 0.0:
        cut_loc = np.median(eig_sorted)
    else:
        cut_loc = eig_sorted[int((n/2)+(rnoise*n))]
    labels = np.ones(n)*(eig > cut_loc)
    
    return labels

def binary_tree(data,cut_function,max_levels=0,node_min_size=4,**kwargs):
    """
    Generic function for generating a binary tree.
    data is the original data matrix OR alternatively, some other matrix
    which is expected by the cut_function.
    cut_function is a cut function as described at the top of this file.
    kwargs can contain all kinds of stuff for cut_function.
    
    The function works as follows:
    It creates a tree_root, which has a elements of length # of columns. 
    Then it calls cut_function(tree_root,data,**kwargs)
    It takes the resulting partition and creates the appropriate children.
    Then it iterates over the children (breadth-first), generating 
    appropriate children on the way down.

    Some default arguments are calculated or supplied:
    
    max_levels = np.floor(np.log(cols)/np.log(2.0))
    alpha = 1.0 (level weighting coefficient)
    beta = 0.0 (for EMD)
    """

    _,cols = np.shape(data)
    
    if max_levels==0:
        max_levels = np.floor(np.log(cols)/np.log(2.0))

    root = tree.ClusterTreeNode(range(cols))
    queue = [root]

    while max([x.size for x in queue]) > 1:
        #work on one level at a time to match how the matlab trees are created
        new_queue = []
        for node in queue:
            if node.size >= node_min_size and node.level + 1 < max_levels:
                #cut it
                cut = cut_function(node,data,**kwargs)
                node.create_subclusters(cut)
            else:
                #make the singletons
                node.create_subclusters(np.arange(node.size))
            new_queue.extend(node.children)
        queue = new_queue

    root.make_index()                
    return root

    