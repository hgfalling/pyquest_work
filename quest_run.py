"""
This file is intended to be used as a script for running the questionnaire
with arbitrary parameters, and to keep the results in the local namespace.
"""

import numpy as np
import scipy.io

import affinity
reload(affinity)
import dual_affinity
reload(dual_affinity)
import bintree_cut
reload(bintree_cut)
import matlab_util
reload(matlab_util)
import tree_util
reload(tree_util)
import scoring
reload(scoring)
import barcode
reload(barcode)
import embedding
reload(embedding)
import tree
reload(tree)

#load data
DATA_PATH = ("/users/jerrod/Google Drive/Yale_Research/Questionnaire_2D_20130614/Examples/")
DATA_FILE = "MMPI2_AntiQuestions.mat"

mdict = scipy.io.loadmat(DATA_PATH+DATA_FILE)
data = mdict["matrix"]
q_descs = [x[0][0] for x in mdict["sensors_dat"][0,0][0]]
p_score_descs = [x[0] for x in mdict["points_dat"][0,0][1][0]]
p_scores = mdict["points_dat"][0,0][0]

#Generate initial affinity
init_row_aff = affinity.mutual_cosine_similarity(data.T,False,0,None)
init_col_aff = affinity.mutual_cosine_similarity(data,False,0,None)

#Compute diffusion embedding of initial affinities
init_row_vecs,init_row_vals = bintree_cut.markov_eigs(init_row_aff, 12)
init_col_vecs,init_col_vals = bintree_cut.markov_eigs(init_col_aff, 12)

#Generate median trees
init_row_tree = bintree_cut.median_tree(init_row_vecs,init_row_vals,max_levels=12)
init_col_tree = bintree_cut.median_tree(init_col_vecs,init_col_vals,max_levels=12)