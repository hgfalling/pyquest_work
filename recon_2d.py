"""
This file is for 2d (but not tensor product) reconstructions of functions
on the unit square. Some of the stuff will be reused; some refactoring will
be necessary as all this comes together.
"""

import cluster_diffusion as cdiff
import numpy as np
#import itertools

def k_tree(k,n_levels):
    elements = range(k**n_levels)
    tree_list = [cdiff.ClusterTreeNode([element]) for element in elements]
    tree_list2 = []

    for _ in xrange(n_levels):
        while len(tree_list) > 0:
            tree_list2.append(cdiff.ClusterTreeNode([]))
            for j in xrange(k):
                tree_list[j].assign_to_parent(tree_list2[-1])
            tree_list = tree_list[k:]
        tree_list = tree_list2
        tree_list2 = []

    tree_list[0].make_index()

    return tree_list[0]

def qt_map_unit_square(tree):
    tree.bbox = (0.0,1.0,1.0,0.0)
    for node in tree.traverse():
        if node.children == []:
            continue
        else:
            midpt_x = (node.bbox[2] + node.bbox[0])*0.5
            midpt_y = (node.bbox[1] + node.bbox[3])*0.5
            node.children[0].bbox = (node.bbox[0],node.bbox[1],midpt_x,midpt_y)
            node.children[1].bbox = (midpt_x,node.bbox[1],node.bbox[2],midpt_y)
            node.children[2].bbox = (node.bbox[0],midpt_y,midpt_x,node.bbox[3])
            node.children[3].bbox = (midpt_x,midpt_y,node.bbox[2],node.bbox[3])
    return tree

def qt_map_unit_sq(tree,minx=0.5,miny=0.5,maxx=0.5,maxy=0.5):
    tree.bbox = (0.0,1.0,1.0,0.0)
    for node in tree.traverse():
        if node.children == []:
            continue
        else:
            xpull = np.random.uniform(minx,maxx)
            ypull = np.random.uniform(miny,maxy)
            midpt_x = node.bbox[2]*xpull + node.bbox[0]*(1.0-xpull)
            midpt_y = node.bbox[3]*ypull + node.bbox[1]*(1.0-ypull)
            node.children[0].bbox = (node.bbox[0],node.bbox[1],midpt_x,midpt_y)
            node.children[1].bbox = (midpt_x,node.bbox[1],node.bbox[2],midpt_y)
            node.children[2].bbox = (node.bbox[0],midpt_y,midpt_x,node.bbox[3])
            node.children[3].bbox = (midpt_x,midpt_y,node.bbox[2],node.bbox[3])
    return tree

def plotbox(node,fig,color):
    fig.hlines([node.bbox[1]],node.bbox[0],node.bbox[2],color)
    fig.hlines([node.bbox[3]],node.bbox[0],node.bbox[2],color)
    fig.vlines([node.bbox[0]],node.bbox[1],node.bbox[3],color)
    fig.vlines([node.bbox[2]],node.bbox[1],node.bbox[3],color)
    
def plotcross(node,fig,color):
    if node.children == []:
        pass
    else:
        bbox = node.children[0].bbox
        fig.hlines([bbox[3]],node.bbox[0],node.bbox[2],color)
        fig.vlines([bbox[2]],node.bbox[1],node.bbox[3],color)

def plotbox3d(node,fig,color):
    fig.plot([node.bbox[0],node.bbox[0]],[node.bbox[1],node.bbox[3]],0,color)
    fig.plot([node.bbox[2],node.bbox[2]],[node.bbox[1],node.bbox[3]],0,color)
    fig.plot([node.bbox[0],node.bbox[2]],[node.bbox[1],node.bbox[1]],0,color)
    fig.plot([node.bbox[0],node.bbox[2]],[node.bbox[3],node.bbox[3]],0,color)
    
def plotcross3d(node,fig,color):
    if node.children == []:
        pass
    else:
        bbox = node.children[0].bbox
        fig.plot(xs=[node.bbox[0],node.bbox[2]],ys=[bbox[3]]*2,zs=0,color=color)
        fig.plot(xs=[bbox[2]]*2,ys=[node.bbox[1],node.bbox[3]],zs=0,color=color)

def point_in_node(point,tree):
    x,y = point
    return (x >= tree.bbox[0] and x <= tree.bbox[2] and y<=tree.bbox[1] and y>=tree.bbox[3])

def search_point(point,tree):
    if tree.children == []:
        return tree
    else:
        for child in tree.children:
            if point_in_node(point,child):
                return search_point(point,child)

#def fval(x,y,tree,f_imputed):
#    
#    m = np.shape(x)[0]
#    n = np.shape(y)[0]
#    
#    
#    f_on_grid = np.zeros([m,n])
#    for (i,j) in itertools.product(xrange(m),xrange(n)):
#        idx = search_point((x[i,j],y[i,j]),tree).elements[0]
#        f_on_grid[i,j] = f_imputed[idx]
#    
#    return f_on_grid
    
def f_to_tree(f,tree):
    n,_ = np.shape(f)
    cts = np.zeros([tree.size],dtype=np.int)
    y = np.zeros([tree.size],dtype=np.float)
    for i in xrange(n):
        idx = search_point(f[i,0:2],tree).elements[0]
        cts[idx] += 1
        y[idx] += f[i,2]
    y[cts>0] /= cts[cts>0]
    y[cts==0] = -1.0
    return y, np.where(cts>0)

def midpt_box(bbox):
    return ( (bbox[0]+bbox[2])*0.5, (bbox[1]+bbox[3])*0.5 )

def plot3ddata(zvals,tree,fig):
    midpts = [midpt_box(x.bbox) for x in tree.leaves()]
    xs = np.array([x[0] for x in midpts])
    ys = np.array([x[1] for x in midpts])
    #fig.plot_surface(xs,ys,zvals)
    return xs,ys

def bbox_p(bbox):
    return np.array([[bbox[0],bbox[0]],[bbox[2],bbox[2]]]),np.array([[bbox[1],bbox[3]],[bbox[1],bbox[3]]])

def gridify(f,tree,res):
    """
    Suppose we have a vector y that is the value on all the leaf folders.
    Now we want to write that vector down at arbitrary resolution 
    in order to combine different folder schemes on the square.
    """
    x_res = np.arange(0,1+res,res)
    f_on_grid = np.zeros([len(x_res),len(x_res)])
    
    for (idx,node) in enumerate(tree.leaves()):
        if node.bbox[0] == 0.0:
            yl = 0
        else:
            yl = int((node.bbox[0]-1e-5) / res) + 1
        ye = int((node.bbox[2]) / res) + 1 
        if node.bbox[3] == 0.0:
            xl = 0
        else:
            xl = int((node.bbox[3]-1e-5) / res) + 1
        xe = int((node.bbox[1]) / res) + 1
        f_on_grid[xl:xe,yl:ye] = f[idx]
    
    return f_on_grid
