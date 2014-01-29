import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

reds = plt.get_cmap('Reds')
cnorm = matplotlib.colors.Normalize(vmin=0.15,vmax=0.35)
reds.set_under('white')
reds.set_over('red')

MARKERS = "o^h+p8Ds"
COLORS = "bgrcmykw"

def plot_embedding(vecs,vals,tree,diff_time=None,level=1,ax=None,highlights = []):
    
    if diff_time is None:
        diff_time = 1.0/(1-vals[1])
    
    print highlights
    points = [x for x in xrange(np.shape(vecs)[0]) if x not in highlights]
    x=vecs[points,1] * (vals[1] ** diff_time)
    y=vecs[points,2] * (vals[2] ** diff_time)
    z=vecs[points,3] * (vals[3] ** diff_time)

    folders = tree.level_nodes(level)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")

    ax.clear()
    ax.scatter3D(x,y,z,c='r')
    
    x=vecs[highlights,1] * (vals[1] ** diff_time)
    y=vecs[highlights,2] * (vals[2] ** diff_time)
    z=vecs[highlights,3] * (vals[3] ** diff_time)
    ax.scatter3D(x,y,z,c='b')
    return ax

def plot_heatmap_embedding(vecs,vals,tree,heatmap,diff_time=None,level=1,ax=None):
    
    if diff_time is None:
        diff_time = 1.0/(1-vals[1])
    
    x=vecs[:,1] * (vals[1] ** diff_time)
    y=vecs[:,2] * (vals[2] ** diff_time)
    z=vecs[:,3] * (vals[3] ** diff_time)
    c=heatmap

    folders = tree.level_nodes(level)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")

    ax.clear()
    for (idx,folder) in enumerate(folders):
        ax.scatter3D(x[folder.elements],y[folder.elements],z[folder.elements],
                     marker=MARKERS[idx],c=c[folder.elements],cmap=reds,norm=cnorm)
    
    return ax

class EmbeddingPlotter:
    def __init__(self,vecs,vals,tree,heatmap,diff_time=None,level=1):
        self.vecs = vecs
        self.vals = vals
        self.tree = tree
        self.heatmap = heatmap
        if diff_time is None:
            self.diff_time = 1.0/(1-vals[1])
        else:
            self.diff_time = diff_time

        self.folders = tree.level_nodes(level)
    
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,projection="3d")
        
        self.draw()
        
        self.fig.canvas.mpl_connect('key_press_event',self.key)
        
    def draw(self):
        x=self.vecs[:,1] * (self.vals[1] ** self.diff_time)
        y=self.vecs[:,2] * (self.vals[2] ** self.diff_time)
        z=self.vecs[:,3] * (self.vals[3] ** self.diff_time)
        c=self.heatmap

        for (idx,folder) in enumerate(self.folders):
            self.ax.scatter3D(x[folder.elements],y[folder.elements],z[folder.elements],
                         marker=MARKERS[idx],c=c[folder.elements],cmap=reds,norm=cnorm)
        
    def key(self,event):
        if event.key == 'alt+right':
            print 'r'
        elif event.key == 'alt+left':
            print 'l'
        
        self.draw()
        self.fig.canvas.draw()
    
class TestingPlotter:
    def __init__(self,data):
        self.data = data
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,projection="3d")
        self.draw()
        
        self.fig.canvas.mpl_connect('key_press_event',self.key)
        
    def draw(self):
        self.ax.scatter3D(self.data[:,0],self.data[:,1],self.data[:,2],c=self.data[:,3],
                          cmap=reds,norm=cnorm)
        
    def key(self,event):
        if event.key == 'alt+right':
            print 'r'
        elif event.key == 'alt+left':
            print 'l'
            
        print event.key
        
        self.draw()
        self.fig.canvas.draw()

if __name__=="__main__":
    data = np.random.rand(10,4)
    tp = TestingPlotter(data)
    plt.show()    