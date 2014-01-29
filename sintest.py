import qmain
import cluster_diffusion as cdiff
import numpy as np
import tree_utils
reload(qmain)
reload(cdiff)
reload(tree_utils)

matrix = np.zeros([128,256])
for i in xrange(128):
    for j in xrange(256):
        matrix[i,j] = np.sin(i*j*np.pi/256.0)

affinity = qmain.local_geometry_gaussian(matrix, 10)
p,eigvals,eigvecs = cdiff.diffusion_embed(affinity,normalized=True)
tree_cols = cdiff.cluster(eigvecs[:,1:],eigvals[1:])

#dual_affinity = qmain.dual_geometry_means(matrix.T, tree_cols, 4)
#p,eigvals,eigvecs = cdiff.diffusion_embed(dual_affinity,normalized=True)
#tree_rows = cdiff.cluster(eigvecs[:,1:],eigvals[1:])

new_aff = qmain.dual_geometry_norm_ip_max(matrix.T, tree_cols)
p,eigvals,eigvecs = cdiff.diffusion_embed(new_aff,normalized=True)
tree_rows = cdiff.cluster(eigvecs[:,1:],eigvals[1:])

new_col_aff = qmain.dual_geometry_norm_ip_max(matrix, tree_rows)
p,eigvals,eigvecs = cdiff.diffusion_embed(new_col_aff,normalized=True)
tree_cols2 = cdiff.cluster(eigvecs[:,1:],eigvals[1:])

new_row_aff = qmain.dual_geometry_norm_ip_max(matrix.T, tree_cols2)
p,eigvals,eigvecs = cdiff.diffusion_embed(new_row_aff,normalized=True)
tree_rows2 = cdiff.cluster(eigvecs[:,1:],eigvals[1:])

new_col_aff = qmain.dual_geometry_norm_ip_max(matrix, tree_rows2)
p,eigvals,eigvecs = cdiff.diffusion_embed(new_col_aff,normalized=True)
tree_cols2 = cdiff.cluster(eigvecs[:,1:],eigvals[1:])

new_row_aff = qmain.dual_geometry_norm_ip_max(matrix.T, tree_cols2)
p,eigvals,eigvecs = cdiff.diffusion_embed(new_row_aff,normalized=True)
tree_rows2 = cdiff.cluster(eigvecs[:,1:],eigvals[1:])

ta = tree_utils.bitree_averages(matrix, tree_rows2, tree_cols2)