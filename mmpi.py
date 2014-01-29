import qmain
import cluster_diffusion as cdiff
import numpy as np

SINGLETONS = False
WEIGHTED = False

ROWS_KNN = 10
COLS_KNN = 10
CLUSTER_KNN = 10
N_ROW_EIGS = 8
N_COL_EIGS = 8

def mmpi_questionnaire(matrix,iters):

    col_data = matrix
    col_aff = qmain.local_geometry_gaussian(col_data,knn=COLS_KNN,
                                            init_eps_pt=False,symm_inner=True)
    p,eigvals,eigvecs = cdiff.diffusion_embed(col_aff,n_eigs=N_COL_EIGS,
                                              normalized=True)
    n_eigs = np.sum(np.abs(eigvals[~np.isnan(eigvals)]) > 1e-14)
    tree_cols = cdiff.cluster(eigvecs[:,1:n_eigs],eigvals[1:n_eigs],
                              knn=CLUSTER_KNN)
    #tree_cols.disp_tree()
    
    row_data = qmain.extend_coords_means(matrix.T,tree_cols,
                                         weighted=WEIGHTED,
                                         singletons=SINGLETONS)
    #row_aff = qmain.local_geometry_norm_ip(row_data)
    row_aff = qmain.local_geometry_gaussian(row_data,knn=ROWS_KNN,
                                            init_eps_pt=False,symm_inner=True)
    p,eigvals,eigvecs = cdiff.diffusion_embed(row_aff,n_eigs=N_ROW_EIGS,
                                              normalized=True)
    n_eigs = np.sum(np.abs(eigvals[~np.isnan(eigvals)]) > 1e-14)
    tree_rows = cdiff.cluster(eigvecs[:,1:n_eigs],eigvals[1:n_eigs],
                              knn=CLUSTER_KNN)
    #tree_rows.disp_tree()
    
    for i in xrange(iters):
        col_data = qmain.extend_coords_means(matrix,tree_rows,
                                             weighted=WEIGHTED,
                                             singletons=SINGLETONS)
        col_aff = qmain.local_geometry_gaussian(col_data,knn=COLS_KNN,
                                            init_eps_pt=False,symm_inner=False)
        p,col_eigvals,col_eigvecs = cdiff.diffusion_embed(col_aff,
                                                          n_eigs=N_COL_EIGS,
                                                          normalized=True)
        #print col_eigvals
        n_eigs = np.sum(np.abs(col_eigvals[~np.isnan(col_eigvals)]) > 1e-14)
        tree_cols = cdiff.cluster(col_eigvecs[:,1:n_eigs],col_eigvals[1:n_eigs],
                                  knn=CLUSTER_KNN)
        #tree_cols.disp_tree()
        
        row_data = qmain.extend_coords_means(matrix.T,tree_cols,WEIGHTED,SINGLETONS)
        row_aff = qmain.local_geometry_gaussian(row_data,knn=ROWS_KNN,
                                                init_eps_pt=False,symm_inner=False)

        p,row_eigvals,row_eigvecs = cdiff.diffusion_embed(row_aff,
                                                          n_eigs=N_ROW_EIGS,
                                                          normalized=True)
        #print row_eigvals
        n_eigs = np.sum(np.abs(row_eigvals[~np.isnan(row_eigvals)]) > 1e-14)
        tree_rows = cdiff.cluster(row_eigvecs[:,1:n_eigs],row_eigvals[1:n_eigs],
                                  knn=CLUSTER_KNN)
        #tree_rows.disp_tree()

    return tree_rows,tree_cols,row_eigvecs,col_eigvecs,row_eigvals,col_eigvals

    