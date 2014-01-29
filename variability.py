import sklearn.neighbors as sknn
import numpy as np

def variability_by_knn(knn,data,vecs,vals):
    knn_model = sknn.NearestNeighbors(n_neighbors=knn+1)
    knn_model.fit(vecs)
    nnd,nnidx = knn_model.kneighbors(vecs)
    
    var = np.zeros([567,2428])
    
    for Q_NO in xrange(567):
        q_std = np.std(data[Q_NO,:])
        for i in xrange(2428):
            var[Q_NO,i] = np.std(data[Q_NO,nnidx[i]])/q_std
    return var