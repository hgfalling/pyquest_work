import numpy as np
import cluster_diffusion as cdiff
import qmain
import random
import matplotlib.pyplot as plt
import scipy.stats

def pk_random_true(n_kickers,n_goalies):
    true_pct = np.zeros([n_kickers,n_goalies])
    kickers_base = np.minimum(scipy.stats.norm.rvs(0.75,0.05,size=n_kickers),0.925)
    goalies_true = np.minimum(scipy.stats.norm.rvs(0.75,0.05,size=n_goalies),1.0)
    #+15% differential with good foot
    #if kickers_foot is 1, then bonus going left.
    #if kickers_foot is 0, then bonus going right.
    kickers_foot = scipy.stats.bernoulli.rvs(0.7,size=n_kickers)
    kickers_left_true = kickers_base + 0.15*kickers_foot - 0.075 
    kickers_right_true = kickers_base - 0.15*kickers_foot + 0.075 
    #goalies_foot = scipy.stats.bernoulli.rvs(0.7,size=n_goalies)
    #pct they go left
    kickers_strat = scipy.stats.uniform.rvs(loc=0.2,scale=0.6,size=n_kickers)
    goalies_strat = scipy.stats.uniform.rvs(loc=0.2,scale=0.6,size=n_goalies)
    
    for kicker in xrange(n_kickers):
        for goalie in xrange(n_goalies):
            freq_LL = kickers_strat[kicker] * goalies_strat[goalie]
            pct_LL = (kickers_left_true[kicker] + goalies_true[goalie])/2.0
            freq_LR = kickers_strat[kicker] * (1-goalies_strat[goalie])
            pct_LR = 0.9 + 0.05*kickers_foot[kicker]
            freq_RL = (1-kickers_strat[kicker]) * goalies_strat[goalie]
            pct_RL = 0.95 - 0.05*kickers_foot[kicker]
            freq_RR = (1-kickers_strat[kicker]) * (1-goalies_strat[goalie])
            pct_RR = (kickers_right_true[kicker] + goalies_true[goalie])/2.0
            true_pct[kicker,goalie] = freq_LL*pct_LL + freq_LR*pct_LR + \
                                        freq_RL*pct_RL + freq_RR*pct_RR 
    
    return kickers_left_true, kickers_right_true, goalies_true, \
        kickers_strat, goalies_strat, true_pct


def pk_questionnaire(matrix,iters):
    col_data = matrix
    col_aff = qmain.local_geometry_norm_ip(col_data)
    p,eigvals,eigvecs = cdiff.diffusion_embed(col_aff,normalized=True)
    n_eigs = np.sum(np.abs(eigvals[~np.isnan(eigvals)]) > 1e-14)
    tree_cols = cdiff.cluster(eigvecs[:,1:n_eigs],eigvals[1:n_eigs])
    #tree_cols.disp_tree()
    
    row_data = qmain.extend_coords_means(matrix.T,tree_cols,False)
    row_aff = qmain.local_geometry_norm_ip(row_data)
    p,eigvals,eigvecs = cdiff.diffusion_embed(row_aff,normalized=True)
    n_eigs = np.sum(np.abs(eigvals[~np.isnan(eigvals)]) > 1e-14)
    tree_rows = cdiff.cluster(eigvecs[:,1:n_eigs],eigvals[1:n_eigs])
    #tree_rows.disp_tree()
    
    for i in xrange(iters):
        col_data = qmain.extend_coords_means(matrix,tree_rows,False,False)
        col_aff = qmain.local_geometry_gaussian(col_data,knn=100)
        p,col_eigvals,col_eigvecs = cdiff.diffusion_embed(col_aff,normalized=True)
        n_eigs = np.sum(np.abs(col_eigvals[~np.isnan(col_eigvals)]) > 1e-14)
        tree_cols = cdiff.cluster(col_eigvecs[:,1:n_eigs],col_eigvals[1:n_eigs])
        #tree_cols.disp_tree()
        
        row_data = qmain.extend_coords_means(matrix.T,tree_cols,False,False)
        row_aff = qmain.local_geometry_gaussian(row_data,knn=75)
        p,row_eigvals,row_eigvecs = cdiff.diffusion_embed(row_aff,normalized=True)
        n_eigs = np.sum(np.abs(row_eigvals[~np.isnan(row_eigvals)]) > 1e-14)
        tree_rows = cdiff.cluster(row_eigvecs[:,1:n_eigs],row_eigvals[1:n_eigs])
        #tree_rows.disp_tree()

    return tree_rows,tree_cols,row_eigvecs,col_eigvecs