from imports import *

MEAN_POSSIBILITIES = [0.2,0.8]
PEOPLE_COUNT = 1024
QUESTIONS = 4
SAMPLES_PER_QUESTION = 8
UNIFORM, DIAGONAL, COMPLEMENT, DIAGONAL_NOISE, GMM = 0,1,2,3,4
MEANS_TYPE = GMM
CENTER_COMBOS = [x for x in itertools.combinations_with_replacement(MEAN_POSSIBILITIES,QUESTIONS)]
CENTER_LISTS = [list(set([z for z in itertools.permutations(y,QUESTIONS)])) for y in CENTER_COMBOS]
CENTERS = []
for center in CENTER_LISTS:
    CENTERS.extend(center)
COVARIANCE = 0.01*np.eye(QUESTIONS)
TRUTH,SAMPLE,SHUFFLE = 0,1,2
SORT_TYPE = TRUTH

means = np.zeros([QUESTIONS*SAMPLES_PER_QUESTION,PEOPLE_COUNT])

if MEANS_TYPE == UNIFORM:
    q_means = np.random.rand(QUESTIONS,PEOPLE_COUNT)
if MEANS_TYPE == DIAGONAL:
    q_means1 = np.random.rand(PEOPLE_COUNT)
    q_means = np.vstack([q_means1,q_means1])
if MEANS_TYPE == COMPLEMENT:
    q_means1 = np.random.rand(PEOPLE_COUNT)
    q_means = np.vstack([q_means1,1.0-q_means1])
if MEANS_TYPE == DIAGONAL_NOISE:
    q_means1 = np.random.rand(PEOPLE_COUNT)
    q_means2 = scipy.stats.norm.rvs(0.0,0.1,size=PEOPLE_COUNT)
    q_means = np.vstack([q_means1,q_means1+q_means2])
if MEANS_TYPE == GMM:
    picker = np.random.randint(0,len(CENTERS),PEOPLE_COUNT) 
    q_means = np.zeros([QUESTIONS,PEOPLE_COUNT])
    for i in range(len(CENTERS)):
        q_means[:,picker == i] = np.random.multivariate_normal(np.array(CENTERS[i]),COVARIANCE,size=(np.sum(picker==i),)).T
    
for row in xrange(QUESTIONS):
    means[row*SAMPLES_PER_QUESTION:(row+1)*SAMPLES_PER_QUESTION,:] = q_means[row,:]

means[means < 0] = 0.0
means[means > 1.0] = 1.0

rdata = np.random.rand(QUESTIONS*SAMPLES_PER_QUESTION,PEOPLE_COUNT)
data = np.zeros([QUESTIONS*SAMPLES_PER_QUESTION,PEOPLE_COUNT])
data[rdata < means] = 1
data[rdata >= means] = -1
if SORT_TYPE == TRUTH:
    row_indices = range(QUESTIONS*SAMPLES_PER_QUESTION)
    if MEANS_TYPE == GMM:
        col_indices = picker.argsort()
    else:
        col_indices = means[:,0].argsort()
elif SORT_TYPE == SAMPLE:
    row_indices = np.mean(data,axis=0).argsort()
    if MEANS_TYPE == GMM:
        col_indices = picker.argsort()
    else:
        col_indices = np.mean(data,axis=1).argsort()
elif SORT_TYPE == SHUFFLE:
    row_indices = np.array(range(QUESTIONS*SAMPLES_PER_QUESTION))
    np.random.shuffle(row_indices)
    col_indices = np.array(range(PEOPLE_COUNT))
    np.random.shuffle(col_indices)

data = data[:,col_indices][row_indices,:]
means = means[:,col_indices][row_indices,:]


#initial affinity and stuff is the same for both methods

#Generate initial affinity
init_row_aff = affinity.mutual_cosine_similarity(data.T,False,0,threshold=0.0)

#Compute diffusion embedding of initial affinities
init_row_vecs,init_row_vals = markov.markov_eigs(init_row_aff, 12)

q = np.eye(init_row_aff.shape[0])
cluster_list = []
i=0
while 1:
    #print "clustering at level {}".format(i)
    i+=1 
    new_affinity = q.dot(init_row_aff).dot(q.T)
    cluster_list.append(tree_building.cluster_from_affinity(new_affinity,0.25))
    if len(cluster_list[-1]) == 1:
        break
    temp_tree = tree_building.clusterlist_to_tree(cluster_list)
    cpart = tree_building.ClusteringPartition([x.elements for x in temp_tree.dfs_level(2)])
    q,_ = tree_building.cluster_transform_matrices(cpart)