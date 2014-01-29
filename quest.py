import affinity 
import bintree_construct
import dual_affinity
import markov
import tree_building
import numpy as np
import scipy.spatial as spsp

def pyquest_bintree(data,row_alpha=0.5,col_alpha=0.5,beta=1.0,bal_constant=1.0,n_iters=3):
    """
    runs what is momentarily the standard questionnaire algorithm:
    initial affinity = mutual cosine similarity
    initial tree based on median of successive eigenvectors
    dual affinities based on earth mover distance.
    dual trees based on eigen_cut method
    """
    #Generate initial affinity
    init_row_aff = affinity.mutual_cosine_similarity(data.T,False,0,threshold=0.1)
    
    #Compute diffusion embedding of initial affinities
    init_row_vecs,init_row_vals = markov.markov_eigs(init_row_aff, 12)
    #Generate median trees
    init_row_tree = bintree_construct.median_tree(init_row_vecs,init_row_vals,max_levels=12)
    
    dual_col_trees = []
    dual_row_trees = [init_row_tree]
    
    for _ in xrange(n_iters):
        dual_col_trees.append(bintree_construct.old_eigen_tree(data,dual_row_trees[-1],alpha=col_alpha,beta=beta,noise=0.0))
        dual_row_trees.append(bintree_construct.old_eigen_tree(data.T,dual_col_trees[-1],alpha=row_alpha,beta=beta,noise=0.0))
#        dual_col_trees.append(bintree_construct.eigen_tree(data,dual_row_trees[-1],alpha=col_alpha,beta=beta,bal_constant=bal_constant))
#        dual_row_trees.append(bintree_construct.eigen_tree(data.T,dual_col_trees[-1],alpha=row_alpha,beta=beta,bal_constant=bal_constant))
        
    col_tree = dual_col_trees[-1]
    row_tree = dual_row_trees[-1]
    
    col_emd = dual_affinity.calc_emd(data,row_tree,alpha=0.5,beta=1.0)
    row_emd = dual_affinity.calc_emd(data.T,col_tree,alpha=0.5,beta=1.0)
    
    row_aff = dual_affinity.emd_dual_aff(row_emd)
    col_aff = dual_affinity.emd_dual_aff(col_emd)
    
    row_vecs,row_vals = markov.markov_eigs(row_aff, 12)
    col_vecs,col_vals = markov.markov_eigs(col_aff, 12)
    
    return row_tree,col_tree,row_vecs,col_vecs,row_vals,col_vals

def pyquest_newtree(data,tree_constant=0.25,row_alpha=0.5,col_alpha=0.5,beta=1.0,n_iters=3):

    init_row_aff = affinity.mutual_cosine_similarity(data.T,False,0,threshold=0.1)
    
    #Compute diffusion embedding of initial affinities
    init_row_vecs,init_row_vals = markov.markov_eigs(init_row_aff, 12)
    init_row_vals[np.isnan(init_row_vals)] = 0.0
    row_embedding = init_row_vecs.dot(np.diag(init_row_vals))
    row_distances = spsp.distance.squareform(spsp.distance.pdist(row_embedding))
    row_affinity = np.max(row_distances) - row_distances
    
    #Generate initial tree
    #print "call1 tree_constant:{}".format(tree_constant)
    init_row_tree = tree_building.make_tree_embedding(row_affinity,tree_constant)
    
    dual_col_trees = []
    dual_row_trees = [init_row_tree]
    
    for _ in xrange(n_iters):
        #print "Beginning iteration {}".format(i)
        col_emd = dual_affinity.calc_emd(data,dual_row_trees[-1],alpha=col_alpha,beta=beta)
        col_aff = dual_affinity.emd_dual_aff(col_emd)
        #print "call2 tree_constant:{}".format(tree_constant)
        dual_col_trees.append(tree_building.make_tree_embedding(col_aff,tree_constant))
    
        row_emd = dual_affinity.calc_emd(data.T,dual_col_trees[-1],alpha=row_alpha,beta=beta)
        row_aff = dual_affinity.emd_dual_aff(row_emd)
        #print "call3 tree_constant:{}".format(tree_constant)
        dual_row_trees.append(tree_building.make_tree_embedding(row_aff,tree_constant))
        
    col_tree = dual_col_trees[-1]
    row_tree = dual_row_trees[-1]
    
    col_emd = dual_affinity.calc_emd(data,row_tree,alpha=col_alpha,beta=beta)
    row_emd = dual_affinity.calc_emd(data.T,col_tree,alpha=row_alpha,beta=beta)
    
    row_aff = dual_affinity.emd_dual_aff(row_emd)
    col_aff = dual_affinity.emd_dual_aff(col_emd)
    
    row_vecs,row_vals = markov.markov_eigs(row_aff, 12)
    col_vecs,col_vals = markov.markov_eigs(col_aff, 12)   

    return row_tree,col_tree,row_vecs,col_vecs,row_vals,col_vals

def pyquest_spin_bintrees(data,row_alpha=0.5,col_alpha=0.5,beta=1.0,bal_constant=1.0,n_iters=3,n_spin=10):
    """
    runs what is momentarily the standard questionnaire algorithm:
    initial affinity = mutual cosine similarity
    initial tree based on median of successive eigenvectors
    dual affinities based on earth mover distance.
    dual trees based on eigen_cut method
    """
    #Generate initial affinity
    init_row_aff = affinity.mutual_cosine_similarity(data.T,False,0,threshold=0.1)
    
    #Compute diffusion embedding of initial affinities
    init_row_vecs,init_row_vals = markov.markov_eigs(init_row_aff, 12)
    #Generate median trees
    init_row_tree = bintree_construct.median_tree(init_row_vecs,init_row_vals,max_levels=12)

    row_trees, col_trees = [],[]

    for _ in xrange(n_spin):
        dual_col_trees = []
        dual_row_trees = [init_row_tree]
        
        for _ in xrange(n_iters):
            dual_col_trees.append(bintree_construct.eigen_tree(data,dual_row_trees[-1],alpha=col_alpha,beta=beta,bal_constant=bal_constant))
            dual_row_trees.append(bintree_construct.eigen_tree(data.T,dual_col_trees[-1],alpha=row_alpha,beta=beta,bal_constant=bal_constant))

        row_trees.append(dual_row_trees[-1])
        col_trees.append(dual_col_trees[-1])
            
    return row_trees,col_trees

