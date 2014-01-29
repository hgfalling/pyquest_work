import numpy as np
#import scipy as sp
#import scipy.spatial as ssp

def old_gaussian_kernel_smooth(f,eps,xres):
    
    n = int(1.0/xres)
    xgrid = np.linspace(0,1,n+1)
    y = np.zeros(n+1)
    for (idx,xval) in enumerate(xgrid):
        f_kernel = np.exp(-np.abs(f[0,:] - xval)**2/eps)
#        if xval == 0.4:
#            print "distances: ", np.abs(f[0,:] - xval)
#            print "eps: ",eps
#            print "kernel vals: ", f_kernel
#            print -np.abs(f[0,:] - xval)**2/eps
        y[idx] = np.sum(f_kernel*f[1,:]/np.sum(f_kernel))
    m = np.shape(f)[1]
    fprime = np.zeros(m)
    for i in xrange(m):
        f_kernel = np.exp(-np.abs(f[0,:] - f[0,i])**2/eps)
        fprime[i] = np.sum(f_kernel*f[1,:]/np.sum(f_kernel))
        
    return xgrid,y,fprime


def gaussian_kernel_smooth(f,eps,xres):
    if np.isscalar(eps):
        eps = np.repeat(eps,np.shape(f)[1])
    #print eps
    n = int(1.0/xres)
    xgrid = np.linspace(0,1,n+1)
    y = np.zeros(n+1)
    for (idx,xval) in enumerate(xgrid):
        f_kernel = np.exp(-np.abs(f[0,:] - xval)**2/eps)
        if np.sum(f_kernel) < 1e-32:
            y[idx] = 0.0
        else:
            y[idx] = np.sum(f_kernel*f[1,:]/(np.sum(f_kernel)))
    m = np.shape(f)[1]
    fprime = np.zeros(m)

    for i in xrange(m):
        f_kernel = np.exp(-np.abs(f[0,:] - f[0,i])**2/eps)
        fprime[i] = np.sum(f_kernel*f[1,:]/np.sum(f_kernel))
        
    return xgrid,y,fprime

def adaptive_fit_gks(f,xres):
    s_matrix = variation(f[1,:])/(variation(f[0,:])+1e-10)
    eps = 1.0/(np.max(s_matrix,axis=0)**2)
    
    return eps,gaussian_kernel_smooth(f,eps,xres)

def calc_eps(f):
    s_matrix = variation(f[1,:])/(variation(f[0,:])+1e-10)
    return 1.0/(np.max(s_matrix,axis=0)**2)

def ms_gaussian(f,init_eps,xres,scales=10):
    
    if init_eps is None:
        eps = calc_eps(f)
    elif np.isscalar(init_eps):
        eps = np.repeat(init_eps,np.shape(f)[1])
    else:
        eps = init_eps
    
    print "init_eps:{}".format(eps)
    f_recon = np.zeros([scales,int(1.0/xres)+1])
    f_prime_recon = np.zeros([scales,np.shape(f)[1]])
    residuals = np.zeros([scales,np.shape(f)[1]])
    
    for scale_idx in xrange(scales):
        residual = f[1,:] - np.sum(f_prime_recon,axis=0)
        residuals[scale_idx] = residual
        res_f = np.vstack([f[0,:],residual])
        xgrid,f_recon[scale_idx],f_prime_recon[scale_idx] = gaussian_kernel_smooth(res_f,eps,xres)
        eps /= 2.0

    return f_recon, residuals

def new_ms_gaussian(f,xres,max_iters=10,scales=10):
    
    f_recon = np.zeros([max_iters*scales,int(1.0/xres)+1])
    f_prime_recon = np.zeros([max_iters*scales,np.shape(f)[1]])
    residuals = np.zeros([max_iters*scales,np.shape(f)[1]])
    
    iter = 0
    
    while iter < max_iters and np.sum(np.abs(residuals[iter*scales-1])) < 1e-16:
        residual = f[1,:] - np.sum(f_prime_recon,axis=0)
        residuals[iter*scales] = residual
        res_f = np.vstack([f[0,:],residual])
        eps,(xgrid,f_recon[iter*scales],f_prime_recon[iter*scales]) = adaptive_fit_gks(res_f,xres)
        for j in xrange(1,scales):
            eps /= 2.0
            residual = f[1,:] - np.sum(f_prime_recon,axis=0)
            residuals[iter*scales+j] = residual
            res_f = np.vstack([f[0,:],residual])
            xgrid,f_recon[iter*scales+j],f_prime_recon[iter*scales+j] = gaussian_kernel_smooth(res_f,eps,xres)
            
        iter += 1

    print iter, np.sum(np.abs(residuals[iter*scales-1]))
    return f_recon, residuals

def ms_poisson(f,xres,init_eps,exponent=2,scales=10):

    f_recon = np.zeros([scales,int(1.0/xres)+1])
    f_prime_recon = np.zeros([scales,np.shape(f)[1]])
    residuals = np.zeros([scales,np.shape(f)[1]])
    
    eps = init_eps
    
    for scale_idx in xrange(scales):
        residual = f[1,:] - np.sum(f_prime_recon,axis=0)
        residuals[scale_idx] = residual
        res_f = np.vstack([f[0,:],residual])
        xgrid,f_recon[scale_idx],f_prime_recon[scale_idx] = poisson_kernel_smooth(res_f,eps,xres,exponent)
        eps /= 2.0

    return f_recon, residuals

def poisson_kernel_smooth(f,eps,xres,exponent):
    if np.isscalar(eps):
        eps = np.repeat(eps,np.shape(f)[1])
    #print eps
    n = int(1.0/xres)
    xgrid = np.linspace(0,1,n+1)
    y = np.zeros(n+1)
    for (idx,xval) in enumerate(xgrid):
        f_kernel = 1.0/(1.0+(np.abs(f[0,:] - xval)/eps)**exponent)
        if np.sum(f_kernel) < 1e-32:
            y[idx] = 0.0
        else:
            y[idx] = np.sum(f_kernel*f[1,:]/(np.sum(f_kernel)))
    m = np.shape(f)[1]
    fprime = np.zeros(m)

    for i in xrange(m):
        f_kernel = 1.0/(1.0+(np.abs(f[0,:] - f[0,i])/eps)**exponent)
        fprime[i] = np.sum(f_kernel*f[1,:]/np.sum(f_kernel))
        
    return xgrid,y,fprime

    
    
    
    
def variation(f):
    f_rep = np.tile(f,[np.shape(f)[0],1])
    return np.abs(f_rep - f_rep.T)    
    
    