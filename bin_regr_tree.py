import sklearn.linear_model as sklm
import numpy as np
import dual_affinity
import markov
import tree

def break_node(train_data,col_tree_node,row_tree,regressors=None,
               alpha=0.5,beta=1.0):
    """
    First calculates the EMD on the columns of train_data 
    in col_tree_node.elements using row_tree. Converts that to an affinity.
    Calculates the second eigenvector of the markov matrix based on that
    affinity.
    Then fits a linear model using the rows in regressors (all if it's None)
    and uses the LASSO path to identify the best five rows.
    Splits the node using the predicted eigenvector values.
    """
    
    col_indices = col_tree_node.elements
    node_data = train_data[:,col_indices].astype(np.float64)
    col_emd = dual_affinity.calc_emd(node_data,row_tree,alpha,beta)
    col_aff = dual_affinity.emd_dual_aff(col_emd)
    
    vecs,_ = markov.markov_eigs(col_aff,2)
    eig = vecs[:,1]
    
    if regressors is None:
        regressors = range(row_tree.size)
    
    _,active,_ = sklm.lars_path(node_data[regressors,:].T,eig,max_iter=10)
    
    regr_indices = active[0:5]
    
    lm = sklm.LinearRegression()
    lm.fit(node_data[regr_indices,:].T,eig)
    pred_eigs = lm.predict(node_data[regr_indices,:].T)
    
    labels = pred_eigs > 0.0
    partition = labels*np.ones(labels.shape[0])
    col_tree_node.create_subclusters(partition)
    return np.array([regressors[x] for x in regr_indices]),lm

def process_node(train_data,row_tree,node_list,regressors=None):
    node = node_list.pop(0)
    if regressors is None:
        regressors = range(row_tree.size)

    if node.size < 12:
        node.create_subclusters(range(node.size))
    else:
        active,lm = break_node(train_data,node,row_tree,regressors)
        node.lm = lm
        node.active = active
        node_list.extend(node.children)

def mtree(train_data,row_tree,regressors=None):
    root = tree.ClusterTreeNode(range(train_data.shape[1]))
    node_list = []
    node_list.append(root)
    if regressors is None:
        regressors = range(row_tree.size)

    while node_list:
        process_node(train_data,row_tree,node_list,regressors)
    
    root.make_index()
    return root