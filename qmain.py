import scipy as sp
import scipy.spatial as spatial
import scipy.sparse as sparse
import numpy as np
import datetime

fout = None

def nn_search(reference_set,query_set,knn=None):
    if knn is None:
        knn = reference_set.shape[1]
    dist_mat = spatial.distance.cdist(reference_set.T,query_set.T).T
    distances = np.zeros([query_set.shape[1],knn])
    nn_idx = np.argsort(dist_mat,axis=1,kind="mergesort")
    
    indices = nn_idx[:,0:knn]
    
    for i in xrange(query_set.shape[1]):
        distances[i] = dist_mat[i,nn_idx[i,0:knn]]
        
    return distances, indices

def local_geometry_gaussian(query_set,reference_set=None,knn=None,
                            init_eps_pt=True,eps_tune=1.0,symm_inner=True):
    
    if reference_set is None:
        reference_set = query_set
    elif reference_set.size == 0:
        reference_set = query_set

        

    ref_features, ref_points = np.shape(reference_set)
    q_features, q_points = np.shape(query_set)
    
    assert ref_features == q_features,  "Error: Dimension mismatch between " + \
                                        "query set and reference set."

    if knn is None:
        knn = ref_points

    knn = min(ref_points,knn)

    dists,idxs = nn_search(reference_set,query_set,knn)
    
    if init_eps_pt:
        eps_pt = eps_tune*(np.median(dists,1)**2)[idxs]
    else:
        medians = np.median(dists,1)
        eps_pt = eps_tune*np.reshape(medians,[q_points,1])*medians[idxs]

    assert len(eps_pt) == q_points,   "Error: Dimension mismatch between " + \
                                        "epsilon by point and query set."

    affinity = np.exp(-(dists**2)/(1e-200 + eps_pt))
    
    aff_lil = sparse.lil_matrix((affinity.shape[0],affinity.shape[0]))
    
    for i in xrange(affinity.shape[0]):
        for j in xrange(affinity.shape[1]):
            aff_lil[i,idxs[i,j]] = affinity[i,j]
    
    aff_csr = aff_lil.tocsr()
    
    if symm_inner:
        return np.dot(aff_csr,aff_csr.transpose())
    else:
        return aff_csr + aff_csr.transpose()

def mean_on_subset(query_set,elements):
    return np.mean(query_set[elements,:],0)

def local_geometry_abs_norm_ip(query_set,reference_set=None):
    #floored at 0
    
    if reference_set is None:
        reference_set = np.zeros(np.shape(query_set))
        reference_set[:] = query_set

    q_set = np.zeros(np.shape(query_set))
    q_set[:] = query_set
    
    query_rows,query_columns = np.shape(query_set)
    ref_rows,ref_columns = np.shape(reference_set)
    
    for i in xrange(query_columns):
        query_norm = np.linalg.norm(q_set[:,i])
        if query_norm > 1e-10:
            q_set[:,i] /= query_norm
    
    for i in xrange(ref_columns):
        ref_norm = np.linalg.norm(reference_set[:,i])
        if ref_norm > 1e-10:
            reference_set[:,i] /= ref_norm

    aff = q_set.T.dot(reference_set)
    return np.abs(aff)

def local_geometry_norm_ip(query_set,reference_set=None):
    #floored at 0
    
    if reference_set is None:
        reference_set = np.zeros(np.shape(query_set))
        reference_set[:] = query_set

    q_set = np.zeros(np.shape(query_set))
    q_set[:] = query_set
    
    query_rows,query_columns = np.shape(query_set)
    ref_rows,ref_columns = np.shape(reference_set)
    
    for i in xrange(query_columns):
        query_norm = np.linalg.norm(q_set[:,i])
        if query_norm > 1e-10:
            q_set[:,i] /= query_norm
    
    for i in xrange(ref_columns):
        ref_norm = np.linalg.norm(reference_set[:,i])
        if ref_norm > 1e-10:
            reference_set[:,i] /= ref_norm

    aff = q_set.T.dot(reference_set)
    return np.maximum(aff,0)

def dual_geometry_means_old(query_set,row_tree,knn,eps_tune=1.0,singletons=False):
    m,n = np.shape(query_set)
    if singletons:
        points = np.zeros([row_tree.tree_size,n])
        points[0:m,:] = query_set
        i = m
    else:
        points = np.zeros([row_tree.tree_size-1-np.shape(query_set)[0],n])
        i = 0
    for node in row_tree.traverse():
        if not node.parent is None and len(node.children) > 0:
            points[i,:] = mean_on_subset(query_set,node.elements)
            i+=1
    return local_geometry_gaussian(points,knn,init_eps_pt=False,symm_inner=False)
    
def extend_coords_means(query_set,row_tree,weighted=False,singletons=False):
    m,n = np.shape(query_set)
    if singletons:
        points = np.zeros([row_tree.tree_size,n])
        points[0:m,:] = query_set[:,:]
        i = m
    else:
        points = np.zeros([row_tree.tree_size-1-np.shape(query_set)[0],n])
        i = 0
    for node in row_tree.traverse():
        if not node.parent is None and len(node.children) > 0:
            points[i,:] = mean_on_subset(query_set,node.elements)
            if weighted:
                points[i,:] *= np.sqrt(len(node.elements))
            i+=1
    return points

def dual_geometry_norm_ip_max(query_set,row_tree,singletons=False):
    m,n = np.shape(query_set)
    points = np.zeros([n,n])
    for node in row_tree.traverse():
        if not singletons and len(node.children) == 0:
            pass
        else:
            node_matrix = query_set[node.elements,:]
            for i in xrange(len(node.elements)):
                row_norm = np.linalg.norm(node_matrix[i,:])
                if row_norm > 1e-10:
                    node_matrix[i,:] /= row_norm
            ips = node_matrix.T.dot(node_matrix)
            points = np.maximum(points,ips)
    return points


        