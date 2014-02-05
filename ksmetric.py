"""
A few functions for calculating the discrete Kolmogorov-Smirnov distance 
between observed_data and data from a probability model. typically 
observed_data would be ones and zeros while model_data would be a probability 
field.
"""
import numpy as np
import matplotlib.pyplot as plt

def ks_metric(observed_data,model_data):
    strout = "samples must have the same shape"
    assert observed_data.shape == model_data.shape, strout
    
    od = observed_data.flatten()
    md = model_data.flatten()
    
    sorder = md.argsort()
    residual = od-md
    
    ksm = np.max(np.abs(np.cumsum(residual[sorder])))
    
    return ksm
    
def ks_metric_L1(observed_data,model_data):
    strout = "samples must have the same shape"
    assert observed_data.shape == model_data.shape, strout
    
    od = observed_data.flatten()
    md = model_data.flatten()
    
    sorder = md.argsort()
    residual = od-md
    
    ksm = np.mean(np.abs(np.cumsum(residual[sorder])))
    
    return ksm
    
    
def realize(model):
    return np.ones(model.shape) * (np.random.rand(*model.shape) < model)

def ksm_plot(X,Xhat):
    sorder = Xhat.flatten().argsort()
    print "L1: {} KS: {} L1KS: {}".format(np.sum(np.abs(X-Xhat)),ks_metric(X,Xhat),ks_metric_L1(X,Xhat))
    plt.plot(np.cumsum((Xhat-X).flatten()[sorder]))
    plt.show()