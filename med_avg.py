import numpy as np

def med_avg(data,n_clusters):
    n = np.shape(data)[0]
    buckets = np.array_split(np.random.permutation(np.arange(n)),n_clusters)
    bucket_size = [n_clusters] + list(np.shape(data)[1:])
    bucket_means = np.zeros(bucket_size)
    for i in xrange(n_clusters):
        bucket_means[i,:,:] = np.mean(data[buckets[i],:,:],axis=0)
    
    return np.median(bucket_means,axis=0)
    
    
    
    
    