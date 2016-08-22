import numpy as np
def unfold(X,n):
    ''' mode-n unfolding of tensor X
    Args: 
        X: input tensor
        n: targeted mode
    Returns:
        matricized version of X
    '''
    shape = np.shape(X)
    ndim = np.ndim(X)
    perm_order = np.roll(np.arange(ndim),n-1)
    X_n= np.reshape(np.transpose(X, perm_order), [shape[n-1],-1])
    return X_n