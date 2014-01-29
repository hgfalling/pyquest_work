import numpy as np
import barcode
import matplotlib.pyplot as plt

class Score(object):
    def __init__(self,data,recon_data):
        denom = 1.0*np.product(np.shape(data))
        self.recon_data = recon_data
        self.classifier = np.sign(recon_data)
        self.classifier[np.abs(self.classifier)<1e-7] = 0.0
        self.prob_L1 = np.sum(np.abs(data-recon_data))/denom
        self.prob_L2 = np.sum((data-recon_data)**2)/denom
        self.class_pct = np.sum(data == self.classifier)/denom       
        self.class_misses = self.classifier != data

    def error_rate_hist(self,axis=0):
        y = 1.0*np.sum(self.class_misses,axis=axis)/np.shape(self.class_misses)[axis]
        plt.hist(y,bins=20)
        plt.xlabel("Error rate")
        plt.show()
        
    def error_rate_bar(self,axis=0,organized=False,row_tree=None,col_tree=None):
        if not organized:
            y = 1.0*np.sum(self.class_misses,axis=axis)/np.shape(self.class_misses)[axis]
        else:
            ydata = barcode.organize_folders(row_tree,col_tree,self.class_misses)
            y = 1.0*np.sum(ydata,axis=axis)/np.shape(self.class_misses)[axis]
            
        x = np.arange(0,np.shape(self.class_misses)[1-axis])
        plt.bar(x,y)
        title_str = 'Error rates'
        if organized:
            title_str += " (organized)"
        plt.title(title_str)
        plt.show()