import numpy as np
import l1_bregman as lb
import sklearn.linear_model as sklm
import warnings
import tree

#six levels worth of how to render folder boundaries
leveldata = [('k','solid'),('m','solid'),('c','solid'),('g','solid'),('y','dashed'),('b','dotted')]

def random_function_data(seed,shape):
    np.random.seed(seed)
    return np.random.rand(*shape)

def fsort(f):
    """expects a 2d array, x values in the first row, y values in the second."""
    return f[:,np.argsort(f[0,:])]

def randomize_folder_size(tree,minfrac,maxfrac):
    for node in tree.traverse():
        if node.parent is None:
            node.lbound = 0.0
            node.rbound = 1.0
            node.frac = np.random.uniform(minfrac,maxfrac)
        else:
            if min(node.elements) == min(node.parent.elements):
                #left of the tree
                node.lbound = node.parent.lbound
                node.rbound = node.parent.lbound + node.parent.frac*(node.parent.rbound - node.parent.lbound)
                node.frac = np.random.uniform(minfrac,maxfrac)
            else:
                node.lbound = node.parent.lbound + node.parent.frac*(node.parent.rbound - node.parent.lbound)
                node.rbound = node.parent.rbound
                node.frac = np.random.uniform(minfrac,maxfrac)
    return tree     

def plot_folder_boundaries(tree,tau,fig):
    lines = []
    for i in xrange(1,tree.tree_depth+1):
        folderlocs = [(node.lbound + tau) % 1.0 for node in tree.level_nodes(i)]
        linelocs = [x for x in folderlocs if x not in lines]
        lines.extend(linelocs)
        fig.vlines(linelocs,0,1,linestyle=leveldata[i-1][1],color=leveldata[i-1][0],label=str(i))
        fig.legend(loc='lower center', bbox_to_anchor=(0.5,1.08),ncol=3,fancybox=True)
    return lines
    
def scatter(f,fig):
    fig.scatter(f[0,:],f[1,:],marker='o')

def reconstruct(f,tree,tau,minfrac,maxfrac):
    randomize_folder_size(tree,minfrac,maxfrac)
    fshifted = (f[0,:] - tau) % 1.0
    tree_intervals = np.array([node.lbound for node in tree.leaves()])
    indices = np.digitize(fshifted,tree_intervals) - 1
    
    fy = np.zeros([tree.size,1])
    fcts = np.zeros(np.shape(fy))
    
    for (i,idx) in enumerate(indices):
        fcts[idx,0] += 1.0
        n = fcts[idx,0]
        fy[idx,0] = ((n-1)/n)*fy[idx,0]+(1/n)*f[1,i]
    
    cl = tree.char_library()
    coeffs,iters = lb.l1_bregman(cl[indices,:],fy[indices,:],1,threshold=1e-6,verbose=False)
    
    shiftedy = cl.dot(coeffs).ravel()
    shiftedx = tree_intervals
    
    nx = (shiftedx + tau) % 1.0
    sorted_ind = nx[0:tree.size].argsort()
    y = shiftedy[sorted_ind]
    y = np.hstack([y[-1],y[-1],y])
    x = nx.copy()[0:tree.size]
    x.sort()
    x = np.hstack([[0.0],x,[1.0]])
    return x,y,coeffs

def sample_recon(f,tree,iters,minfrac,maxfrac,tau=None):
    
    if tau is None:
        taus = np.random.rand(iters)
    elif np.isscalar(tau):
        taus = np.array([tau]*iters)
    else:
        taus = np.array(tau)
    
    xhist = np.zeros([iters,tree.size+2])
    yhist = np.zeros([iters,tree.size+2])
    
    for i in xrange(iters):
        x,y,coeffs = reconstruct(f,tree,taus[i],minfrac,maxfrac)
        xhist[i,:] = x
        yhist[i,:] = y
        
    return xhist,yhist
        
def threshold_mean(fdetail,threshold):
    if fdetail.ndim == 1:
        fdetail = fdetail.reshape([-1,1])
        cols = 1
    else:
        cols = np.shape(fdetail)[1]
    values = np.zeros(cols)
    
    for col in xrange(cols):
        g = fdetail[:,col]
        r = (g.max() - g.min())*1e-7
        h,bins = np.histogram(g,range=(g.min()-r,g.max()+r))
        bins = np.digitize(g,bins) - 1
        indices = (-h).argsort()
        last_bin = np.argmax(np.cumsum(h[indices]) > np.sum(h)*threshold)
        values[col] = np.mean(g[np.array([x in indices[0:last_bin+1] for x in bins])])
    return values

def combine(xhist,yhist,xres=0.01):

    n = int(1.0/xres)
    iters = np.shape(xhist)[0]
    xgrid = np.linspace(0,1,n+1)
    ydetail = np.zeros([iters,n+1])
    
    for i in xrange(iters):
        indices = np.digitize(xgrid,xhist[i,:],True)
        indices[0] = 1
        ydetail[i,:] = yhist[i,indices]
            
    return xgrid,ydetail    
    
def histogram_x(x,y,x_value,fig):
    idx = (np.abs(x-x_value)).argmin()
    h = fig.hist(y[:,idx])
    return h,idx

def gaussian_kernel_smooth(f,eps,xres):
    n = int(1.0/xres)
    xgrid = np.linspace(0,1,n+1)
    y = np.zeros(n+1)
    for (idx,xval) in enumerate(xgrid):
        f_kernel = np.exp(-np.abs(f[0,:] - xval)**2/eps)
        y[idx] = np.sum(f_kernel*f[1,:]/np.sum(f_kernel))
        
    return xgrid,y

"""things below this are all totally dev-ish"""

def reconstruct_l2(f,tree,minfrac,maxfrac,alpha=1.0,suppress_warnings=True):
    randomize_folder_size(tree,minfrac,maxfrac)
    tree_intervals = np.array([node.lbound for node in tree.leaves()])
    indices = np.digitize(f[0,:],tree_intervals) - 1
    
    fy = np.zeros([tree.size])
    fcts = np.zeros(np.shape(fy))
    
    for (i,idx) in enumerate(indices):
        fcts[idx] += 1.0
        n = fcts[idx]
        fy[idx] = ((n-1)/n)*fy[idx]+(1/n)*f[1,i]
    
    active_indices = np.where(fcts>0)[0]
    #print active_indices
    
    cl = tree.char_library(alpha)
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            alphas,active_vars,coef_path = sklm.lars_path(cl[active_indices,:],
                                                   fy[active_indices,:],
                                                   method='lasso',max_iter=2000)
    else:
        alphas,active_vars,coef_path = sklm.lars_path(cl[active_indices,:],
                                               fy[active_indices,:],method='lasso',
                                               max_iter=2000)

    return tree_intervals, fy, alphas, active_vars, coef_path, active_indices
    #coeffs,iters = lb.l1_bregman(cl[indices,:],fy[indices,:],1,threshold=1e-6,verbose=False)
    
    #shiftedy = cl.dot(coeffs).ravel()
    #shiftedx = tree_intervals
    
    #nx = (shiftedx + tau) % 1.0
    #sorted_ind = nx[0:tree.size].argsort()
    #y = shiftedy[sorted_ind]
    #y = np.hstack([y[-1],y[-1],y])
#    x = nx.copy()[0:tree.size]
#    x.sort()
#    x = np.hstack([[0.0],x,[1.0]])
#    return x,y,coeffs


class ReconTree(tree.ClusterTreeNode):
    def calc_delta_library(self):
        tree_size = self.size
        
        for node in self.nodes_list:
            node.calc_delta(tree_size)
            
    def delta_library(self,weights=None):
        indices = []
        dlib = np.zeros([self.size,self.tree_size])
        cweights = np.zeros([self.tree_size])
        for (idx,node) in enumerate(self.nodes_list):
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
        
        for node in self.nodes_list:
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
        
        for node in self.nodes_list:
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
    
