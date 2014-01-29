import matlab_util as mu
reload(mu)
import scipy.io
import cluster_diffusion as cdiff
reload(cdiff)
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm

def folder_path(element_id,tree):
    path = []
    path.append(tree)
    cur_node = tree
    
    while cur_node.elements != [element_id]:
        for child in cur_node.children:
            if element_id in child.elements:
                path.append(child)
                cur_node = child
    
    return path
    
def bi_folder_predict(row_folder,col_folder,target,data):
    regressors = [z for z in row_folder.elements if z < 500]
    training_col_elements = [z for z in col_folder.elements if z >=2000]
    test_col_elements = [z for z in col_folder.elements if z<2000]
    regr_x = data[regressors,:][:,training_col_elements]
    regr_y = data[target,training_col_elements]
    test_x = data[regressors,:][:,test_col_elements]
    test_y = data[target,test_col_elements]
    
    if len(training_col_elements) < 5:
        print "Less than 5 training columns, using next bigger column folder."
        
    
    if np.sum(regr_y) == len(regr_y) or np.sum(regr_y) == 0:
        prediction = np.repeat(regr_y[0],len(test_col_elements))
        l2_score = np.sum(prediction == test_y)*1.0/len(test_y)
        #print "all classes equal"
        #l2_score = l1_score
    else:
        test_x = data[regressors,:][:,test_col_elements]
        test_y = data[target,test_col_elements]
        l2_regr = sklm.LogisticRegression()
        try:
            l2_regr.fit(regr_x.T,regr_y)
        except ValueError:
            print np.sum(regr_y)
        l2_score = l2_regr.score(test_x.T,test_y)
        #prediction = l2_regr.predict_proba(test_x.T)[:,1]
        prediction = l2_regr.predict(test_x.T)
        #print row_folder.size,col_folder.size,len(training_col_elements)
    #print row_folder.level, col_folder.level, row_folder.size, col_folder.size, np.sum(l2_regr.predict(test_x.T)), np.shape(test_y)[0], "Score: {}".format(l2_score)
    #print np.shape(prediction)
    return prediction, l2_score

def rscore(predicted,true):
    return np.sum(predicted==true)*1.0/np.product(np.shape(predicted))

def filterfolders(folderlist,filter_list,threshold=5):
    ret_list = []
    for folder in folderlist:
        if len([x for x in folder.elements if x in filter_list]) > threshold:
            ret_list.append(folder)
    return ret_list

class BiFolderPrediction(object):
    def __init__(self,row_folder,col_folder,data):
        self.row_folder = row_folder
        self.col_folder = col_folder
        
        self.prediction = {}
        self.prob = {}
        self.regr_score = {}
        self.match_score = {}
        self.regr = {}
        
        self.regressors = [z for z in row_folder.elements if z < 500]
        self.target_rows = [z for z in row_folder.elements if z >= 500]
        self.training_col_elements = [z for z in col_folder.elements if z >=2000]
        self.test_col_elements = [z for z in col_folder.elements if z < 2000]
        
        train_x = data[self.regressors,:][:,self.training_col_elements]
        test_x = data[self.regressors,:][:,self.test_col_elements]
        
        for target_row in self.target_rows:
            train_y = data[target_row,self.training_col_elements]
            test_y = data[target_row,self.test_col_elements]

            if np.sum(train_y) == len(train_y) or np.sum(train_y) == 0:
                self.prediction[target_row] = np.repeat(train_y[0],len(self.test_col_elements))
                self.prob[target_row] = np.repeat(train_y[0],len(self.test_col_elements))
                self.regr_score[target_row] = 1.0
                self.match_score[target_row] = np.sum(np.abs(self.prediction[target_row] - test_y))/len(train_y) 
            else:
                regr = sklm.LogisticRegression(penalty='l1')
                regr.fit(train_x.T,train_y)
                self.prediction[target_row] = regr.predict(test_x.T)
                self.prob[target_row] = regr.predict_proba(test_x.T)
                self.regr_score[target_row] = regr.score(train_x.T,train_y)
                self.match_score[target_row] = regr.score(test_x.T,test_y) 
                self.regr[target_row] = regr
            


