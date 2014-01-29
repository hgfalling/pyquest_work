import scipy.io
import qmain
import cluster_diffusion as cdiff
import datetime
import tree_utils
import numpy as np
reload(qmain)
reload(cdiff)

def matlab_to_pyquest(tree):
    """
    in: dictionary entry containing struct from matlab
    out: cluster
    """
    matlab_tree = tree[0]
    sub_folders = []
    super_folders = []
    for level in xrange(np.shape(matlab_tree)[0]):
        #tree structures from matlab questionnaire are funny-shaped in numpy.
        folder_count = matlab_tree[level][0][0][0][0][0]
        partition = matlab_tree[level][0][0][3][0] - 1
        new_folders = []
        for i in xrange(folder_count):
            elements = (np.nonzero(partition == i)[0]).tolist() 
            new_folders.append(cdiff.ClusterTreeNode(elements))
        for idx,sub_folder in enumerate(sub_folders):
            sub_folder.assign_to_parent(new_folders[super_folders[idx]])
        sub_folders = new_folders[:]
        super_folders = matlab_tree[level][0][0][2][0] - 1
    
    new_folders[0].makeindex()
    return new_folders[0]

def sensors_data(matlab_data):
    str_vals = []
    for x in matlab_data["sensors_dat"][0,0][0]:
        str_vals.append(x[0][0])
    return str_vals

if __name__ == "__main__":
    
    start = datetime.datetime.now()
    
    data = scipy.io.loadmat("c:/users/jerrod/google drive/yale_research/Questionnaire2D_20121016/Examples/MMPI2_AntiQuestions.mat")
    matrix = data["matrix"]
    
#    fpdata = scipy.io.loadmat("c:/users/jerrod/google drive/yale_research/fold_points.mat")
#    fp = matlab_to_pyquest(fpdata["fold_points"])
        
    affinity = qmain.local_geometry_gaussian(matrix, 10)
    p,eigvals,eigvecs = cdiff.diffusion_embed(affinity,normalized=True)
    tree_cols = cdiff.cluster(eigvecs[:,1:],eigvals[1:])
    
#    tree_cols = fp
    
    dual_affinity = qmain.dual_geometry_means(matrix.T, tree_cols, 10)
    p,eigvals,eigvecs = cdiff.diffusion_embed(dual_affinity,normalized=True)
    tree_rows = cdiff.cluster(eigvecs[:,1:],eigvals[1:])

    dual_affinity = qmain.dual_geometry_means(matrix, tree_rows, 10)
    p,eigvals,eigvecs = cdiff.diffusion_embed(dual_affinity,normalized=True)
    tree_cols = cdiff.cluster(eigvecs[:,1:],eigvals[1:])
    
    dual_affinity = qmain.dual_geometry_means(matrix.T, tree_cols, 10)
    p,eigvals,eigvecs = cdiff.diffusion_embed(dual_affinity,normalized=True)
    tree_rows = cdiff.cluster(eigvecs[:,1:],eigvals[1:])
    
    print datetime.datetime.now() - start
    

