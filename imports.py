import matplotlib

#matplotlib.interactive(True)
#matplotlib.use('WXAgg')

import affinity
import dual_affinity
#import bintree_construct
import bin_tree_build
import flex_tree_build
import barcode
import markov
#import scoring
import tree_util
import tree
#import tree_building
import tree_recon
import viewer_files
#import quest
import artificial_data
import haar
import question_tree

#import run_quest
import questionnaire
import contextlib

import cPickle
import itertools
import numpy as np
import scipy.io
import scipy.spatial as spsp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plot_utils import *

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
    
    