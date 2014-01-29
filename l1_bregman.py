import numpy as np

def l1_bregman(A,y,mu,positive=False,n_inner=50,delta=1e-3,threshold=1e-4,verbose=True):
    
    n = np.shape(A)[1]
    Q = np.linalg.inv(mu*(A.T.dot(A))+np.eye(n))
    #print "Q:", np.shape(Q)
    u = min_split(Q,A,y,mu,positive,n_inner,delta,threshold,verbose)
    
    return u

def l1_l2_bregman(A,y,mu,positive=False,n_inner=50,delta=1e-3,threshold=1e-4,verbose=True,guess=None):
    n = np.shape(A)[1]
    Q = np.linalg.inv(mu*(A.T.dot(A))+np.eye(n))
    u = min_split_l2(Q,A,y,mu,positive)

    return u

def min_split(Q,A,y0,mu,positive,n_inner=50,delta=1e-3,threshold=1e-4,verbose=True):
    
    n = np.shape(A)[1]
    if np.ndim(y0) == 1:
        b = np.zeros([n])
        d = np.zeros([n])
    else:
        n2 = np.shape(y0)[-1]
        b = np.zeros([n,n2])
        d = np.zeros([n,n2])
    
    y = y0
    iters = []
    muAt = mu*A.T
    
    norm1 = 1000.0
    new_norm = 0.0
    x=0
    #while abs(new_norm - norm1) > 1e-8 and x < 25000:
    while abs(new_norm - norm1)/(norm1 + 1e-10) > threshold and x < 25000:
        norm1 = new_norm
#        if x % 100 == 0:
#            print x
        x+=1
        muAty = muAt.dot(y)
        
        for j in xrange(n_inner):
            u = Q.dot(muAty - (b-d))
            
            if positive:
                d = np.maximum(0,u+b-delta)
            else:
                d = np.sign(u+b)*np.maximum(np.abs(u+b)-delta,0)
            
            b = b + u - d
        y = y - (A.dot(u)-y0)
        new_norm = np.sum(np.abs(d))
        iters.append(d)
#        if threshold == 0.0:
#            threshold = new_norm*1e-4
#            print "iteration convergence threshold: {}".format(threshold)
        if verbose:
            print new_norm, norm1, new_norm - norm1
    if verbose:
        print "L1 norm: {}".format(new_norm)
        print "Total iterations: {}".format(x)
    
    return d,iters

def min_split_l2(Q,A,y0,mu,positive,iters=5000,delta=1e-3,threshold=1e-4,verbose=True,guess=None):
    
    n = np.shape(A)[1]
    if np.ndim(y0) == 1:
        if guess is None:
            b = np.zeros([n])
            d = np.zeros([n])
        else:
            b = guess.copy()
            d = guess.copy()
    else:
        n2 = np.shape(y0)[-1]
        if guess is None:
            b = np.zeros([n,n2])
            d = np.zeros([n,n2])
        else:
            b = guess.copy()
            d = guess.copy()
    
    y = y0
    iters = []
    muAt = mu*A.T
    muAty = muAt.dot(y)
    
    norm1 = 1000.0
    new_norm = 0.0
    x=0
    #while abs(new_norm - norm1) > 1e-8 and x < 25000:
    while abs(new_norm - norm1)/(norm1 + 1e-10) > threshold and x < iters:
        norm1 = new_norm
        x+=1
        
        u = Q.dot(muAty - (b-d))
        
        if positive:
            d = np.maximum(0,u+b-delta)
        else:
            d = np.sign(u+b)*np.maximum(np.abs(u+b)-delta,0)
        
        b = b + u - d
        if np.ndim(d) == 1:
            errnorm = np.linalg.norm(A.dot(d)-y0,2)
        else:
            errnorm = np.linalg.norm(A.dot(d)-y0,'fro')
        new_norm = np.sum(np.abs(d)) + mu*errnorm
        iters.append(d)
#        if threshold == 0.0:
#            threshold = new_norm*1e-4
#            print "iteration convergence threshold: {}".format(threshold)
        if verbose:
            #pass
            print new_norm, norm1, new_norm - norm1
    if verbose:
        l1_norm = np.sum(abs(d))
        print "L1 norm: {}".format(l1_norm)
        if np.ndim(d) == 1:
            errnorm = np.linalg.norm(A.dot(d)-y0,2)
        else:
            errnorm = np.linalg.norm(A.dot(d)-y0,'fro')
        print "L2 norm: {}".format(errnorm)
        print "Total norm: {}".format(new_norm)
        print "Total iterations: {}".format(x)
    
    return d,iters
