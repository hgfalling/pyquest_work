import scipy.stats as spst
import numpy as np

import tree_util

_hg_cache = {}

def _hg_p_value(n_parent,k_parent,n_child,k_child):
    """
    one-tailed hypothesis test.
    H0: The partition is random.
    """
    hg = spst.hypergeom(n_parent,k_parent,n_child)
    parent_mean = n_child*(k_parent*1.0/n_parent)
    
    if k_child <= parent_mean:
        #then we want to know what the probability is
        #that we would observe a result as extreme as this one.
        return max(0.0,hg.cdf(k_child))
    else:
        return max(1-hg.cdf(k_child-1),0.0)    

def hg_p_value(n_parent,k_parent,n_child,k_child):
    """
    Retrieves cached p_value for hypothesis test, or if not cached, 
    computes, adds to cache, and returns p_value.
    """
    if (n_parent,k_parent,n_child,k_child) in _hg_cache:
        return _hg_cache[(n_parent,k_parent,n_child,k_child)]
    else:
        p_val = _hg_p_value(n_parent,k_parent,n_child,k_child)
        _hg_cache[(n_parent,k_parent,n_child,k_child)] = p_val
        return p_val
    
def bitree_p_values(data,row_tree,col_tree):
    """
    Performs associated hypothesis test for each bitree coefficient.
    Returns matrix of p_values corresponding to those coefficients.
    """
    p_values = np.zeros([row_tree.tree_size,col_tree.tree_size])
    k_values = tree_util.bitree_sums(data*(data > 0),row_tree,col_tree).astype(np.int)
    for i in xrange(row_tree.tree_size):
        for j in xrange(col_tree.tree_size):
            row_node = row_tree[i]
            col_node = col_tree[j]
            if i == 0 and j == 0:
                #it's the entire matrix, so let p_value = 0.
                continue
            elif i==0:
                #then it's the entire matrix in the row direction
                col_parent = col_node.parent
                n_parent = col_parent.size*row_node.size
                k_parent = k_values[row_node.idx,col_parent.idx]
                n_child = col_node.size*row_node.size
                k_child = k_values[row_node.idx,col_node.idx]
            elif j==0:
                #then it's the entire matrix in the column direction
                row_parent = row_node.parent
                n_parent = row_parent.size*col_node.size
                k_parent = k_values[row_parent.idx,col_node.idx]
                n_child = col_node.size*row_node.size
                k_child = k_values[row_node.idx,col_node.idx]
            else:
                #it's a node with two parents.
                #the test here is between the subnode and the total of the
                #two parent nodes (which includes the subnode twice). 
                row_parent = row_node.parent
                col_parent = col_node.parent
                n_parent1 = col_parent.size*row_node.size
                n_parent2 = col_node.size*row_parent.size
                n_child = col_node.size*row_node.size
                n_parent = n_parent1 + n_parent2 - n_child
                k_parent1 = k_values[row_node.idx,col_parent.idx]
                k_parent2 = k_values[row_parent.idx,col_node.idx]
                k_child = k_values[row_node.idx,col_node.idx]
                k_parent = k_parent1 + k_parent2 - k_child
            p_values[i,j] = hg_p_value(n_parent,k_parent,n_child,k_child)
    return p_values

def bitree_null_coeffs(data,row_tree,col_tree):
    null_coeffs = np.zeros([row_tree.tree_size,col_tree.tree_size],np.float)

    data_avgs = tree_util.bitree_averages(data,row_tree,col_tree)
    for i in xrange(row_tree.tree_size):
        for j in xrange(col_tree.tree_size):
            row_node = row_tree[i]
            col_node = col_tree[j]
            if i == 0 and j == 0:
                #it's the entire matrix, so the null coeff is the average.
                null_coeffs[0,0] = data_avgs[0,0]
            elif i==0 or j==0:
                #if we're on the outside of the matrix, then the null 
                #coefficients are just zero.
                null_coeffs[i,j] = 0.0
            else:
                #it's a node with two parents.
                #now the null coefficient is more complicated.
                row_parent = row_node.parent
                col_parent = col_node.parent
                
                #W = B_A + B_WX + B_WY + B_W
                #we want W = avg on the union of the parents.
                total_avg = data_avgs[row_parent.idx,col_parent.idx]
                parent_avg1 = data_avgs[row_node.idx,col_parent.idx]
                parent_avg2 = data_avgs[row_parent.idx,col_node.idx]
                sub_avg = data_avgs[row_node.idx,col_node.idx]
                parent_size1 = row_node.size*col_parent.size
                parent_size2 = row_parent.size*col_node.size
                sub_size = row_node.size*col_node.size

                union_sum = parent_avg1*parent_size1 + parent_avg2*parent_size2 - sub_avg*sub_size
                union_denom = parent_size1 + parent_size2 - sub_size
                union_avg = union_sum/(1.0*union_denom)

                null_coeffs[i,j] = union_avg - (parent_avg1 + parent_avg2 - total_avg) 
    return null_coeffs
                
