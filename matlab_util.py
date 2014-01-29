import numpy as np
import cluster_diffusion as cdiff
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        folder_count = matlab_tree[level]['folder_count'][0,0][0,0]
        partition = matlab_tree[level]['partition'][0,0][0] - 1
        new_folders = []
        for i in xrange(folder_count):
            elements = (np.nonzero(partition == i)[0]).tolist() 
            new_folders.append(cdiff.ClusterTreeNode(elements))
        for idx,sub_folder in enumerate(sub_folders):
            sub_folder.assign_to_parent(new_folders[super_folders[idx]])
        sub_folders = new_folders[:]
        super_folders = matlab_tree[level]['super_folders'][0,0][0] - 1
    
    new_folders[0].make_index()
    return new_folders[0]

def plot_embdedding(col_eigvecs):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    x = col_eigvecs[:,1:2]
    y = col_eigvecs[:,2:3]
    z = col_eigvecs[:,3:4]
    ax.scatter(x,y,z)
    plt.show()