import matplotlib

#matplotlib.interactive(True)
#matplotlib.use('WXAgg')

import affinity
import dual_affinity
import bintree_construct
import bin_tree_build
import barcode
import markov
#import scoring
import tree_util
import tree
import tree_building
import tree_recon
import viewer_files
#import quest
import artificial_data
import haar

import run_quest
import contextlib

import cPickle
import itertools
import numpy as np
import scipy.io
import scipy.spatial as spsp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plot_utils

cmap = plt.get_cmap("RdBu_r")
cnorm = matplotlib.colors.Normalize(vmin=-1,vmax=1,clip=False)
cmap.set_under('blue')
cmap.set_over('red') 

bwmap = plt.get_cmap("binary_r")
bwmap.set_under('black')
bwmap.set_over('white') 
bwnorm = matplotlib.colors.Normalize(vmin=0,vmax=1,clip=False)

def bwplot(data):
    plt.imshow(data,interpolation='nearest',aspect='auto',cmap=bwmap,norm=bwnorm)

def bwplot2(data):
    plt.imshow(data,interpolation='nearest',aspect='auto',cmap=bwmap,norm=cnorm)
    
def cplot(data):
    plt.imshow(data,interpolation='nearest',aspect='auto',cmap=cmap,norm=cnorm)

def write_data_file(filename,*args,**kwargs):
    fout = open(filename,"wb")
    np.savez_compressed(fout,*args,**kwargs)
    fout.close()
    
@contextlib.contextmanager
def printoptions(*args,**kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args,**kwargs)
    yield
    np.set_printoptions(**original)
    
    