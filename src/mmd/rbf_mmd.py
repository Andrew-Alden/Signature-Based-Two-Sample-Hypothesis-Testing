# Code adapted from https://github.com/tgcsaba/KSig

import torch
import numpy as np
import torch.nn.functional as F


def validate_data(X):

    """
    Check if X has correct shape.
    :param X: Numpy array.
    :return: The array in the correct shape.
    """

    if X.ndim > 2:
      X = X.reshape(X.shape[:-2] + (-1,))
    return X


def rbf_kernel_diag(X):

    """
    Diagonal elements of RBF gram matrix.
    :param X: Input array.
    :return: 1-dimensional Tensor of 1 with size equal to the rows of X.
    """

    return torch.full((X.shape[0],), 1).to(device=X.get_device())


def matrix_mult(X, Y=None, transpose_Y=False):

    """
    Perform matrix multiplication.
    :param X: Input matrix.
    :param Y: Input matrix. If Y is None then Y=X. Default is None.
    :param transpose_Y: Boolean indicating whether to transpose Y. Default is False.
    :return: X*Y.
    """

    subscript_X = '...ij'
    subscript_Y = '...kj' if transpose_Y else '...jk'
    return torch.einsum(f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)

def squared_norm(X, axis=-1):

    """
    Compute squared norm.
    :param X: Input array.
    :param axis: Axis along which to perform the sum. Default is -1.
    :return: Squared norm.
    """

    return torch.sum(torch.pow(X, 2), axis=axis)

def squared_euclid_distance(X, Y=None):

    """
    Compute squared Euclidean distance.
    :param X: Input array.
    :param Y: Input array. If Y is None, then Y=X. Default is None.
    :return: squared Euclidean distance.
    """

    X_n2 = squared_norm(X)
    if Y is None:
        return X_n2[..., :, None] + X_n2[..., None, :] - 2 * matrix_mult(X, X, transpose_Y=True)
    else:
        Y_n2 = squared_norm(Y)
        return X_n2[..., :, None] + Y_n2[..., None, :] - 2 * matrix_mult(X, Y, transpose_Y=True)

        
def rbf_kernel_compute(X, Y=None, sigma=1.0):

    """
    Compute RBF Gram matrix.
    :param X: Input array.
    :param Y: Input array. If Y is None, then Y=X. Default is None.
    :param sigma: RBF parameter. Default is 1.0.
    :return: RBF Gram matrix.
    """

    D2_scaled = squared_euclid_distance(X, Y) / (2*(sigma**2))
    return torch.exp(-D2_scaled)
    

def rbf_kernel(X, Y=None, diag=False, sigma=1.0):

    """
    Compute RBF kernel.
    :param X: Input array.
    :param Y: Input array. If Y is None, then Y=X. Default is None.
    :param sigma: RBF parameter. Default is 1.0.
    :return: Either RBF Gram matrix or 1-dimensional Tensor of 1 with size equal to the rows of X.
    """

    X = validate_data(X)
    if diag:
        return rbf_kernel_diag(X)
    else:
        Y = validate_data(Y) if Y is not None else None
        return rbf_kernel_compute(X, Y, sigma=sigma)

def rbf_embedding(X, Y=None, diag=False, sigma=1.0):

    """
    Compute RBF kernel embedding.
    :param X: Input array.
    :param Y: Input array. If Y is None, then Y=X. Default is None.
    :param sigma: RBF parameter. Default is 1.0.
    :return: Either RBF Gram matrix or 1-dimensional Tensor of 1 with size equal to the rows of X.
    """

    if diag:
        return rbf_kernel(X[..., None, :], sigma=sigma)
    else:
        if Y is None:
            return rbf_kernel(X[:, None, :, None, :], X[None, :, :, None, :], sigma=sigma)
        else:
            return rbf_kernel(X[:, None, :, None, :], Y[None, :, :, None, :], sigma=sigma)


def multi_cumsum(M, exclusive=False, axis=-1):

    """
    Multi-dimensional cumulative sum.
    :param M: Input tensor.
    :param exclusive: Perform additional operations to last element.
    :param axis: Axis over which to perform operation.
    :return: Tensor.
    """

    ndim = M.ndim
    axis = [axis] if np.isscalar(axis) else axis
    axis = [ndim+ax if ax < 0 else ax for ax in axis]
    
    if exclusive:
        # Slice off last element.
        slices = tuple(
          slice(-1) if ax in axis else slice(None) for ax in range(ndim))
        M = M[slices]
    
    for ax in axis:
        M = torch.cumsum(M, axis=ax)
    
    if exclusive:
        # Pre-pad with zeros.
        pads = tuple((1, 0) if ax in axis else (0, 0) for ax in range(ndim))
        d = M.get_device()
        M = torch.Tensor(np.pad(M.cpu(), pads)).to(device=d)
    
    return M


def signature_kern(M, _k):

    """
    Compute signature kernel.
    :param M: Gram matrix.
    :param _k: Signature truncation level.
    :return: Signature kernel.
    """

    M = torch.diff(torch.diff(M, axis=-2), axis=-1)
    d = M.get_device()

    if M.ndim == 4:
        n_X, n_Y  = M.shape[:2]
        K = torch.ones((n_X, n_Y), dtype=M.dtype).to(device=d)
    else:
        n_X = M.shape[0]
        K = torch.ones((n_X,), dtype=M.dtype).to(device=d)

    K = [K, torch.sum(M, axis=(-2, -1))]

    R = M.clone()
    for i in range(1, _k):
        R = M * multi_cumsum(R, exclusive=True, axis=(-2, -1))
        K.append(torch.sum(R, axis=(-2, -1)))


    return torch.stack(K, axis=0)


def compute_rbf_kernel(X, _k, Y=None, diag=False, sigma=1.0):

    """
    Compute signature kernel with RBF kernel as statis kernel.
    :param X: Input array.
    :param _k:Signature truncation level.
    :param Y: Input array. If Y is None, then Y=X. Default is None.
    :param sigma: RBF parameter. Default is 1.0.
    :return: Signature kernel computation.
    """

    M = rbf_embedding(X, Y=Y, diag=diag, sigma=sigma)
    return signature_kern(M, _k)


def construct_block_diagonals(block_matrix):

    """
    Construct block diagonal matrix.
    :param block_matrix: Input block matrix.
    :return: Diagonal matrices.
    """

    diagonals_tensor = block_matrix.diagonal(dim1=1, dim2=2)
    index = torch.arange(block_matrix.size(1), device=diagonals_tensor.device)
    diagonal_matrices = torch.zeros_like(block_matrix)
    diagonal_matrices[torch.arange(block_matrix.size(0)).unsqueeze(1), index, index] = diagonals_tensor
    return diagonal_matrices

def rbfsig_level_mmd(x1, x2, level=6, sigma=1.0, phik=lambda x: 1, return_levels=False, unbiased=True):

    """
    Truncated sig-MMD with static kernel the RBF kernel.
    :param x1: Collection of paths.
    :param x2: Collection of paths.
    :param level: Signature truncation level. Default is 6.
    :param sigma: RBF parameter. Default is 1.0.
    :param phik: Weight function. Default is contstant weighting of 1.0.
    :param return_levels: Flag indicating whether to return individual levels. Default is False.
    :param unbiased: Flag indicating whether to use unbiased estimator. Default is True.
    :return: Sig-MMD.
    """

    scaling_factors = torch.Tensor([0]+[phik(i) for i in range(1, level+1)])
    K_XX_block = torch.einsum("i,ijk->ijk", scaling_factors, compute_rbf_kernel(x1, level, sigma=sigma).cpu())
    K_YY_block = torch.einsum("i,ijk->ijk", scaling_factors, compute_rbf_kernel(x2, level, sigma=sigma).cpu())
    K_XY_block = torch.einsum("i,ijk->ijk", scaling_factors, compute_rbf_kernel(x1, level, x2, sigma=sigma).cpu())

    if not return_levels:
        
        K_XX = torch.sum(K_XX_block, axis=0)
        K_YY = torch.sum(K_YY_block, axis=0)
        K_XY = torch.sum(K_XY_block, axis=0)

        if unbiased:
            K_XX -= torch.diag(torch.diag(K_XX))
            K_YY -= torch.diag(torch.diag(K_YY))
            N = x1.size(0)
            M = x2.size(0)
            return 1/(N*(N-1)) * torch.sum(K_XX) + 1/(M*(M-1)) * torch.sum(K_YY) - 2 * torch.mean(K_XY)
        else:
            return torch.mean(K_XX) + torch.mean(K_YY) - 2 * torch.mean(K_XY)
        
    else:

        if unbiased:

            K_XX_block_diagonals = construct_block_diagonals(K_XX_block) 
            K_YY_block_diagonals = construct_block_diagonals(K_YY_block) 
    
            K_XX_block -= K_XX_block_diagonals
            K_YY_block -= K_YY_block_diagonals
    
            N = x1.size(0)
            M = x2.size(0)
            return 1/(N*(N-1)) * torch.sum(K_XX_block, dim=(1, 2)) + 1/(M*(M-1)) * torch.sum(K_YY_block, dim=(1, 2)) - 2 * torch.mean(K_XY_block, dim=(1, 2))
        else:
            return torch.mean(K_XX_block, dim=(1, 2)) + torch.mean(K_YY_block, dim=(1, 2)) - 2 * torch.mean(K_XY_block, dim=(1, 2))

    