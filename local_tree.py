import numpy as np
import tree
import sklearn.neighbors as sknn

def make_tree(vecs,vals,diffusion_time,approx_cluster_size):
    diff_vals = vals**diffusion_time
    diff_vecs = vecs.dot(np.diag(diff_vals))

    cluster_list = []
    centroids = diff_vecs    
    iters = 0
    while len(centroids) != 1 and iters < 10:
        #print len(cluster_list), len(centroids)
        approx_size = min(approx_cluster_size,len(centroids)/2+1)
        clusters, partition = cluster(centroids,approx_size)
        cluster_list.append(clusters)
        diff_vecs = centroids.dot(np.diag(diff_vals))
        centroids = calc_centroids(diff_vecs,clusters)
        iters += 1
    return cluster_list

def construct_tree(cluster_list,n):
    child_level = []
    parent_level = []
    for i in xrange(n):
        child_level.append(tree.ClusterTreeNode([i]))
    
    for clusters in cluster_list:
        for cluster in clusters:
            new_node = tree.ClusterTreeNode([])
            for node in cluster:
                child_level[node].assign_to_parent(new_node)
            parent_level.append(new_node)

        child_level = list(parent_level)
        parent_level = []
            
    return new_node

def estimate_radius(data,approx_cluster_size):
    
    kp,dists = random_cover(data,n_neighbors=approx_cluster_size*2)
    return np.median(dists) + 1e-10
    
def cluster(data,approx_cluster_size):
    est_dist = estimate_radius(data,approx_cluster_size)
    #print est_dist
    key_points,distances = random_cover(data,radius=est_dist)

    knn = sknn.NearestNeighbors(n_neighbors=2)
    knn.fit(data[key_points,:])
    
    t_assigns = knn.kneighbors(data)[1][:,0]
    assigns = [key_points[x] for x in t_assigns]
    
    clusters = []
    for idx in np.unique(assigns):
        clusters.append(np.where(np.array(assigns)==idx)[0])
    return clusters, assigns

def calc_centroids(data,clusters):
    return np.vstack([np.mean(data[x,:],axis=0) for x in clusters])

def diffusion_random_cover(A):
    """
    A should be a row-stochastic nxn matrix.
    Returns a set of centers such that diffusing these centers one step will 
    cover the entire dataset.
    """
    n_points = np.shape(A)[0]
    points_list = range(n_points)
    centers = []
    
    while points_list:
        r_pt = np.random.choice(points_list)
        centers.append(r_pt)
        onestep = np.where(A[r_pt,:] > 0.0)[0]
        for pt in onestep:
            if pt in points_list:
                points_list.remove(pt)
    return np.array(centers)

def random_cover(data,**kwargs):
    n_points = np.shape(data)[0]

    if "radius" in kwargs:
        base_radius = kwargs["radius"]
        knn = sknn.NearestNeighbors(radius=base_radius)
        knn.fit(data)
    elif "distances" in kwargs:
        if "n_neighbors" in kwargs:
            n_neighbors = kwargs["n_neighbors"]
        elif "radius" in kwargs:
            base_radius = kwargs["radius"]
    elif "n_neighbors" in kwargs:
        n_neighbors = kwargs["n_neighbors"]
        knn = sknn.NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(data)
    else:
        print "need to specify a type of random cover."
        return None

    points_list = range(n_points)
    key_points = []
    distances = []
    
    while points_list:
        r_pt = np.random.choice(points_list)
        key_points.append(r_pt)
        if "radius" in kwargs:
            nn = knn.radius_neighbors(data[r_pt,:],radius=base_radius)
            distances.append(nn[0][0][-1])
            for pt in nn[1][0]:
                if pt in points_list:
                    points_list.remove(pt)
        elif "distances" in kwargs:
            if "radius" in kwargs:
                nn = np.where(data[r_pt,:] < base_radius)[0]
            else:
                nn = np.argsort(data[r_pt,:])
            added = 0
            for pt in nn:
                if pt in points_list:
                    points_list.remove(pt)
                    added += 1
                if added == n_neighbors:
                    break
            #print len(points_list)
        elif "n_neighbors" in kwargs:
            nn = knn.kneighbors(data[r_pt,:],n_neighbors=n_neighbors)
            distances.append(nn[0][0][-1])
            for pt in nn[1][0]:
                if pt in points_list:
                    points_list.remove(pt)
    return key_points,distances
