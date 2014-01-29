import numpy as np
import scipy.sparse as sparse
import scipy.spatial as spatial
import scipy.sparse.linalg as splinalg
import qmain

def default_chooser(data):
    return data[np.random.randint(0,len(data))] 

def diffusion_embed(aff_matrix,n_eigs=8,threshold=1e-6,normalized=False):
    
    if not sparse.issparse(aff_matrix):
        aff_matrix = sparse.csr_matrix(aff_matrix)
    
    aff_coo = aff_matrix.tocoo()
    
    aff_shape = np.shape(aff_matrix)
    assert aff_shape[0] == aff_shape[1], "Affinity matrix must be square."
    
    aff2 = aff_matrix - aff_matrix.transpose()
    assert aff2.sum() < 1e-12, "Affinity matrix must be symmetric."
    
    n_eigs = min(n_eigs,aff_shape[0])
    
    #Laplace-Beltrami
    row_sums = np.array(np.abs(aff_matrix).sum(1))
    d = row_sums.flatten()
    ridx,cidx = aff_coo.row, aff_coo.col
    new_data = aff_coo.data/(d[ridx]*d[cidx])

    p_mat = sparse.coo_matrix((new_data,[ridx,cidx]),shape=aff_shape)
    d2 = np.sqrt(np.array(p_mat.sum(1)))
    d3 = d*d2.flatten()
    
    new_data2 = aff_coo.data/(d3[ridx]*d3[cidx])
    p_mat = sparse.coo_matrix((new_data2,[ridx,cidx]),shape=aff_shape)
    
    u,s,v = splinalg.svds(p_mat,k=n_eigs)
    
    sidx = np.argsort(-s)
    eigvals = s[sidx]
    eigvecs = u[:,sidx]
    
    if normalized:
        d = abs(eigvecs[:,0])
        for i in xrange(n_eigs):
            eigvecs[:,i] = eigvecs[:,i] / d
            
            if eigvecs[0,i] != 0.0:
                eigvecs[:,i] = eigvecs[:,i] * np.sign(eigvecs[0,i])
        
    
    return p_mat,eigvals,eigvecs
    
def cluster(eigvecs,eigvals,base_radius=None,knn=10):
    n_points = np.shape(eigvecs)[0]
    orphans = []
    for i in xrange(n_points):
        orphans.append(ClusterTreeNode([i]))
        
    d = np.diag(eigvals)
    points = d.dot(eigvecs.T)
    
    new_parents = []
    if base_radius is None:
        knn = min(np.shape(points)[1],knn)
        dists = qmain.nn_search(points,points,knn)[0]
        base_radius = np.median(dists)
        print "The estimated radius for building folders is {}.".format(base_radius)

    partition_centers = np.array([[1,2],[1,2]]) #run loop at least once

    while np.shape(partition_centers)[1] > 1:
        partition, partition_centers = euclidean_cluster(points,base_radius)
        for center in np.unique(partition):
            new_parents.append(ClusterTreeNode([]))
        for idx,orphan in enumerate(orphans):
            orphan.assign_to_parent(new_parents[partition[idx]])
        orphans = new_parents[:]
        new_parents = []
        d = d.dot(d)
        points = d.dot(partition_centers)
    assert len(orphans) == 1
    orphans[0].make_index()  
    return orphans[0]

def partition_centers(points,partition):
    pts_centers = np.zeros([np.shape(points)[0],len(np.unique(partition))])
    j=0
    for center in np.unique(partition):
        pts_centers[:,j] = np.mean(points[:,partition==center],1)
        j+=1
    return pts_centers

def euclidean_cluster(points,base_radius=None,n_clusters=None,rgen=default_chooser):
    
    msg = "Must specify either a radius or a number of clusters."
    assert not(base_radius is None and n_clusters is None), msg
    
    iters=10
    unassigned = range(np.shape(points)[1])
    assigned = []
    
    centers= []
    
    while len(unassigned) > 0:
        new_center = rgen(unassigned)
        centers.append(new_center)
        assigned.append(new_center)
        unassigned.remove(new_center)
        
        center_2d = np.reshape(points[:,new_center],[1,-1])
        ua_dists = spatial.distance.cdist(points[:,unassigned].T,
                                          center_2d)
        ball = np.array(unassigned)[ua_dists.flatten() < base_radius]
        for pt in ball:
            assigned.append(pt)
            unassigned.remove(pt)

    pts_centers = points[:,centers]
    for i in xrange(iters):
        idxs = qmain.nn_search(pts_centers, points, 1)[1]
        partition = idxs.flatten()
        pts_centers = partition_centers(points,partition)

    idxs = qmain.nn_search(pts_centers, points, 1)[1]
    partition = idxs.flatten()
    
    return partition, pts_centers
    
class ClusterTreeNode(object):
    def __init__(self,elements,parent=None):
        self.parent = parent
        self.elements = sorted(set(elements))
        self.children = []
    
    def create_subclusters(self,partition):
        assert len(partition) == len(self.elements)
        p_elements = set(partition)
        for subcluster in p_elements:
            sc_elements = [x for (x,y) in zip(self.elements,partition) 
                           if y == subcluster]
            self.children.append(ClusterTreeNode(sc_elements,self))

    def assign_to_parent(self,parent):
        self.parent = parent
        parent.children.append(self)
        parent.elements.extend(self.elements)
        parent.elements = sorted(set(parent.elements))

    def traverse(self,floor_level=None):
        #BFS
        queue = []
        traversal = []
        queue.append(self)
        while len(queue) > 0:
            node = queue.pop(0)
            traversal.append(node)
            if floor_level is None:
                queue.extend(node.children)
            elif node.level <= floor_level - 1:
                queue.extend(node.children)
        traversal.sort(key=lambda x:x.level*1e10+min(x.elements))    
        return traversal
    
    def dfs_leaves(self):
        traversal = []
        if len(self.elements) == 1:
            traversal.append(self)
        else:
            for child in self.children:
                traversal.extend(child.dfs_leaves())
        return traversal
     
    def dfs_level(self,level=None):
        if level is None:
            level = self.tree_depth
        if level < 0:
            level = self.tree_depth + level
        traversal = []
        if self.level == level:
            traversal.append(self)
        else:
            for child in self.children:
                traversal.extend(child.dfs_level(level))
        return traversal
    
    def leaves(self):
        leaves_list = []
        for node in self.traverse():
            if len(node.children) == 0:
                leaves_list.append(node)
        return leaves_list
    
    @property
    def tree_size(self):
        return len([x for x in self.traverse()])

    @property
    def level(self):
        if self.parent is None:
            return 1
        else:
            return 1+self.parent.level
    
    @property
    def tree_depth(self):
        if self.children == []:
            return 1
        else:
            return 1 + self.children[0].tree_depth
                        
    @property
    def size(self):
        return len(self.elements)

    @property
    def child_sizes(self):
        return [x.size for x in self.children]
    
    def sublevel_elements(self,level):
        elist = []
        for x in self.traverse():
            if x.level + 1 - self.level == level:
                elist.append(x.elements)
        return elist
    
    def level_nodes(self,level):
        elist = []
        for x in self.traverse():
            if x.level + 1 - self.level == level:
                elist.append(x)
        return elist
    
    def make_index(self):
        idx = 0
        for node in self.traverse():
            node.idx = idx
            idx += 1
            
    def disp_tree(self):
        for i in xrange(self.tree_depth):
            print i,self.sublevel_elements(i+1)
            
    def disp_tree_folder_sizes(self):
        for i in xrange(self.tree_depth):
            print i,sorted([len(x) for x in self.sublevel_elements(i+1)])
            
    def calc_delta_library(self):
        tree_size = self.size
        
        for node in self.traverse():
            node.calc_delta(tree_size)
            
    def delta_library(self,weights=None):
        indices = []
        dlib = np.zeros([self.size,self.tree_size])
        cweights = np.zeros([self.tree_size])
        for (idx,node) in enumerate(self.traverse()):
            if np.sum(np.abs(node.d_vector)) > 0.0:
                indices.append(idx)
            dlib[:,idx] = node.d_vector
            cweights[idx] = 1.0*node.size/self.size

        if weights is None:
            weights = np.eye(len(indices))
        elif weights == "foldersize":
            weights = np.diag(cweights)
            print weights
            
        return dlib[:,indices]
    
    def calc_delta(self, tree_size=None):
        if tree_size is None:
            tree_size = self.size
        
        support = []
        if len(self.children) == 0:
            support = self.elements
        else:
            for child in self.children:
                support.extend(child.elements) 
        
        self.norm_c_vector = np.zeros([tree_size])
        self.c_vector = np.zeros([tree_size])
        #print len(support), tree_size
        self.norm_c_vector[support] = 1.0/len(support)
        self.c_vector[support] = 1.0
        
        if self.parent is None:
            self.d_vector = self.norm_c_vector
        else:
            self.d_vector = self.parent.norm_c_vector - self.c_vector
            
    def char_library(self,indices=None,alpha=1.0):
        
        dlib = np.zeros([self.size,self.tree_size])
        ct = 0
        
        for node in self.traverse():
            dlib[:,ct] = node.c_vector
            ct += 1
            
        penalties = (np.sum(dlib,axis=0)/self.size)**alpha
        return dlib.dot(np.diag(penalties))
    
    def filtered_char_library(self,indices,alpha=1.0):
        col_indices = []
        
        if indices is None:
            indices = range(self.size)
        
        dlib = np.zeros([self.size,self.tree_size])
        ct = 0
        idx = 0
        
        for node in self.traverse():
            if (node.parent is None):
                dlib[:,ct] = node.c_vector
                ct += 1
                col_indices.append(idx)
            elif (node.c_vector[indices] == node.parent.c_vector[indices]).all():
                #print "vectors match", node.c_vector[indices], node.parent.c_vector[indices]
                pass
            elif np.sum(node.c_vector[indices]) > 0.0:
                dlib[:,ct] = node.c_vector
                ct += 1
                col_indices.append(idx)
                
            idx += 1
            
        penalties = (np.sum(dlib,axis=0)/self.size)**alpha
        return dlib.dot(np.diag(penalties))[:,0:ct], col_indices
    
        
            
def dyadic_tree(n):
    elements = range(2**n)
    tree_list = [ClusterTreeNode([element]) for element in elements]
    tree_list2 = []

    for i in xrange(n):
        while len(tree_list) > 0:
            tree_list2.append(ClusterTreeNode([]))
            tree_list[0].assign_to_parent(tree_list2[-1])
            tree_list[1].assign_to_parent(tree_list2[-1])
            tree_list = tree_list[2:]
        tree_list = tree_list2
        tree_list2 = []

    tree_list[0].make_index()

    return tree_list[0]
        
def filter_tree(tree,elements):
    """Returns a different tree which contains only folders with non-empty 
    intersection with the list elements."""
    print tree.elements, elements
    elements = set(elements)
    
    new_tree = ClusterTreeNode(tree.elements)
    
    ct = len([x for x in tree.children if elements.intersection(x.elements)])
    print ct
    if ct > 1:
        for child in tree.children:
            if elements.intersection(child.elements):
                nt2 = filter_tree(child,elements)
                nt2.assign_to_parent(new_tree)
                print "nt2:",nt2.elements,nt2.parent.elements
                

    return new_tree
            
def multi_for(iterables):
    if not iterables:
        yield ()
    else:
        for item in iterables[0]:
            for rest_tuple in multi_for(iterables[1:]):
                yield (item,) + rest_tuple        

              
    
        
    
    
    
    
    
           
            