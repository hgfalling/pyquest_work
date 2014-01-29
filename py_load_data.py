"""
Small script for loading MMPI data to run questionnaire from .npz files
    (nothing) = load MMPI2.npz  (the original file)
    aq        = load MMPI2_AntiQuestions.npz (the doubled questions file)
    de        = load MMPI2.Depolarized.npz (the depolarized file)
Fix the DEFAULT_DATA_PATH before using.
"""

import numpy as np
import sys

def load_data(file_path):
    with np.load(file_path) as fdict:
        data = fdict["data"]
        q_descs = fdict["q_descs"]
        p_score_descs = fdict["p_score_descs"]
        p_scores = fdict["p_scores"]
    
    return data,q_descs,p_score_descs,p_scores 

if __name__ == "__main__":
    DEFAULT_DATA_PATH = "./"

    if len(sys.argv) == 1:
        file_path = DEFAULT_DATA_PATH + "MMPI2.npz"
    elif sys.argv[1]=="aq":
        file_path = DEFAULT_DATA_PATH + "MMPI2_AntiQuestions.npz"
    elif sys.argv[1]=="de":
        file_path = DEFAULT_DATA_PATH + "MMPI2_Depolarized.npz"

    data,q_descs,p_score_descs,p_scores = load_data(file_path)
