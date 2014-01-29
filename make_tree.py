import affinity
import numpy as np
import local_tree2 as lt2
import local_tree as lt

row_aff = affinity.mutual_cosine_similarity(data.T)

for m in np.arange(0.1,0.25,0.01):
    components = lt2.find_components(row_aff,m)
    counts = np.bincount(components)
    print m, np.shape(row_aff)[0] - np.max(counts)