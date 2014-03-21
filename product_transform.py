import numpy as np
import tree_util

def tree_product_transform(data,row_tree):
    avs = tree_util.tree_averages(data,row_tree)
    coefs = np.zeros(np.shape(avs))
    if avs.ndim == 1:
        for node in row_tree:
            if node.parent is None:
                coefs[node.idx] = avs[node.idx]
            else:
                coefs[node.idx] = avs[node.idx]/avs[node.parent.idx]
    else:
        for node in row_tree:
            if node.parent is None:
                coefs[node.idx,:] = avs[node.idx,:]
            else:
                coefs[node.idx,:] = avs[node.idx,:]/avs[node.parent.idx,:]
    coefs[np.isnan(coefs)] = 1.0
    return coefs

def bitree_product_transform(data,row_tree,col_tree):
    avs = tree_util.bitree_averages(data,row_tree,col_tree)
    coefs = np.zeros(np.shape(avs))
    
    #requires that node 0 is the root of the tree
    coefs[0,0] = avs[0,0]
    for node in col_tree[1:]:
        coefs[0,node.idx] = avs[0,node.idx]/avs[0,node.parent.idx]
    for node in row_tree[1:]:
        coefs[node.idx,0] = avs[node.idx,0]/avs[node.parent.idx,0]
    
    for row_node in row_tree[1:]:
        for col_node in col_tree[1:]:
            dparent = avs[row_node.parent.idx,col_node.parent.idx]*avs[row_node.idx,col_node.idx]
            parent_product = avs[row_node.parent.idx,col_node.idx]*avs[row_node.idx,col_node.parent.idx]
            coefs[row_node.idx,col_node.idx] = dparent/parent_product
    
    coefs[np.isnan(coefs)] = 1.0
    return coefs

def inverse_bitree_product_transform(coefs,row_tree,col_tree,threshold=0.0):
    return inverse_tree_product_transform(inverse_tree_product_transform(coefs, row_tree, threshold).T,col_tree,threshold).T
            
def inverse_tree_product_transform(coefs,row_tree,threshold=0.0):
    n = row_tree.size
    if coefs.ndim == 1:
        mat = np.ones([row_tree.size],np.float)
        for node in row_tree:
            if node.size*1.0/n >= threshold:
                mat[node.elements] *= coefs[node.idx]
    else:
        mat = np.ones([row_tree.size,np.shape(coefs)[1]])
        for node in row_tree:
            if node.size*1.0/n >= threshold:
                mat[node.elements,:] *= coefs[node.idx,:]
    
    return mat
    
