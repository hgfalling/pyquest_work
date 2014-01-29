import numpy as np
import tree
import bintree_cut
import collections

def make_tree(affinity):

    #start with initial affinity
    
    #threshold initial affinity
    
    #normalize to MC
    
    #apply MC to each point.
    
    #construct new affinity on 
    
    
    
    
    
    
    
    return bintree_cut.make_markov(affinity)

def threshold(affinity,threshold):
    aff = affinity.copy()
    aff[affinity<threshold] = 0.0
    return aff

def bfs(affinity, start):
    queue, enqueued = collections.deque([(None, start)]), set([start])
    while queue:
        parent, n = queue.popleft()
        yield parent, n
        new = set(np.where(affinity[n,:]>0)[0].tolist()) - enqueued
        enqueued |= new
        queue.extend([(n, child) for child in new])
        
def find_components(affinity,thres):
    t_affinity = threshold(affinity,thres)
    t_affinity[t_affinity > 0] = 1.0
    
    assign_components = np.zeros(np.shape(t_affinity)[0], np.int)
    component_no = 1
    while len(np.where(assign_components == 0)[0]) > 0:
        init_pos = np.where(assign_components == 0)[0][0]
        component = [x[1] for x in bfs(t_affinity,init_pos)]
        #_proc_step(t_affinity,init_pos,component)
        assign_components[component] = component_no
        component_no += 1
    #print thres, -np.sort(-np.bincount(assign_components))[0:3]
    return assign_components