import scipy.io
import affinity
reload(affinity)
import bintree_cut
reload(bintree_cut)
reload(bintree_cut.tree)

mdict = scipy.io.loadmat('/users/jerrod/Google Drive/Yale_Research/Questionnaire_2D_20130614/Examples/MMPI2.mat')
data = mdict['matrix']

sim_mat = affinity.norm_ip_abs_raw(data)

vecs,vals = bintree_cut.markov_eigs(sim_mat,8)

