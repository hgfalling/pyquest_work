import numpy as np
import matplotlib.pyplot as plt
import matplotlib

cmap = plt.get_cmap("RdBu_r")
cnorm = matplotlib.colors.Normalize(vmin=-1,vmax=1,clip=False)
cmap.set_under('blue')
cmap.set_over('red') 

def plot_tree(t,nodecolors=None,leafcolors=None):
    node_locs = np.zeros([t.tree_size,2])
    node_order = []
    for level in xrange(1,t.tree_depth+1):
        nodes = t.dfs_level(level)
        node_order.extend([x.idx for x in nodes])
        node_idxs = np.array([node.idx for node in nodes])
        x_intervals = np.cumsum(np.array([0]+[node.size for node in nodes])*1.0/t.size)
        node_xs = x_intervals[:-1] + np.diff(x_intervals)/2.0
        node_ys = (t.tree_depth - level)*np.ones(np.shape(node_xs))
        node_locs[node_idxs,:] = np.hstack([node_xs[:,np.newaxis],node_ys[:,np.newaxis]])
    if nodecolors is not None:
        nc = nodecolors
        plt.scatter(node_locs[:,0],node_locs[:,1],marker='.',edgecolors='none',c=nc,norm=cnorm,cmap=cmap,s=80)
    elif leafcolors is not None:
        lc = leafcolors
        nonleaves = (t.tree_size - t.size)
        nc = ['k']*nonleaves
        plt.scatter(node_locs[0:nonleaves,0],node_locs[0:nonleaves,1],
                    edgecolors='none',marker='.',c=nc,s=80)
        plt.scatter(node_locs[nonleaves:,0],node_locs[nonleaves:,1],
                    edgecolors='none',marker='.',c=lc,s=80)
            
    else:
        nc = 'k'
        plt.scatter(node_locs[:,0],node_locs[:,1],marker='.',c=nc,s=80)


    for node in t:
        if node.parent is not None:
            x1,y1 = node_locs[node.idx,:]
            x2,y2 = node_locs[node.parent.idx,:]
            plt.plot((x1,x2),(y1,y2),'r')
    plt.yticks(np.arange(0,t.tree_depth,1))
    plt.xlim([0.0,1.0])
    plt.ylim([-0.2,(t.tree_depth - 1) + 0.2])
    
def plot_embedding(vecs,vals,diff_time=None,nodecolors=None,partition=None,
                   ax=None):
    
    if diff_time is None:
        diff_time = 1.0/(1.0 - vals[1])
    
    x=vecs[:,1] * (vals[1] ** diff_time)
    y=vecs[:,2] * (vals[2] ** diff_time)
    z=vecs[:,3] * (vals[3] ** diff_time)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")

    if partition is not None:
        COLORS = "krcmybg"
        c = [COLORS[w % len(COLORS)] for w in partition]
    elif nodecolors is not None:
        c = nodecolors
    else:
        c = 'b'
    
    ax.scatter3D(x,y,z,c=c,norm=cnorm,cmap=cmap)
    plt.title("Diffusion Embedding: $t={0:1.3}$".format(diff_time))
