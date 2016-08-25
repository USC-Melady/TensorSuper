import numpy as np

def from_to_without(frm, to, without, step=1, skip=1, reverse=False, separate=False):
    """
    Helper function to create ranges with missing entries
    """
    if reverse:
        frm, to = (to - 1), (frm - 1)
        step *= -1
        skip *= -1
    a = list(range(frm, without, step))
    b = list(range(without + skip, to, step))
    if separate:
        return a, b
    else:
        return a + b
    
    
def unfold(X,mode):
    ''' mode-n unfolding of tensor X
    Args: 
        X: input tensor
        mode: targeted mode
    Returns:
        matricized version of X
    '''
    # shape = np.shape(X)
    # ndim = np.ndim(X)
    # perm_order = np.roll(np.arange(ndim),mode-1)
    # X_n= np.reshape(np.transpose(X, perm_order), [shape[mode-1],-1])
    sz = np.array(X.shape)
    N = len(sz)
    order = ([mode], from_to_without(N - 1, -1, mode, step=-1, skip=-1))
    newsz = (sz[order[0]][0], np.prod(sz[order[1]]))
    arr = np.transpose(X, axes=(order[0] + order[1]))
    arr = arr.reshape(newsz)
    return arr

def fold(X, mode, shape):
    """ Fold a matrix back into a tensor
    Args:
        X: input matrix
        n: unfolding mode
        shape: tensor shape
    Returns:
        refolded tensor
    """
    X_out = np.ndarray(shape)
    shape_array = np.array(shape)
    N = len(shape)
    order = ([mode], from_to_without(0, N, mode, reverse=True))
    arr = np.reshape(X, tuple(shape_array[order[0]],) + tuple(shape_array[order[1]]))
    arr = np.transpose(arr, np.argsort(order[0] + order[1]))
    return arr

def shrink(X, thres):
    """Soft thresholding the singular values of a matrix"""
    [U, S, V] = np.linalg.svd(X,full_matrices=False)
    S = np.maximum(S - thres,0)
    return np.dot(np.dot(U,  np.diag(S)), V)


def tensor_norm(X, p):
    """ norm (measure) for tensor array """
    X_v = np.ndarray.flatten(X)

    if p=='fro':
        norm_val = np.sqrt(np.dot(X_v.T, X_v))
    return norm_val
    
    