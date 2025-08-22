# Code to compute the MMD was adapted from Github repo: https://github.com/maudl3116/higherOrderKME.
# The code contains Cython and GPU functionality to speed up computations.

import numpy as np
import torch
import torch.cuda
from numba import cuda
import math
import torch.nn.functional as F

from .cython_backend import sig_kernel_batch_varpar, sig_kernel_Gram_varpar
from .cuda_backend import compute_sig_kernel_batch_varpar_from_increments_cuda, \
    compute_sig_kernel_Gram_mat_varpar_from_increments_cuda


# ===========================================================================================================
# Static kernels
#
# We start by defining the kernels we want to sequentialize
# ===========================================================================================================
class LinearKernel():
    """Linear kernel k: R^d x R^d -> R"""

    def __init__(self, add_time=0, scaling=1.0):
        self.add_time = add_time
        self.scaling = scaling

    def batch_kernel(self, X, Y):
        """Input:
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """

        k = torch.bmm(X, Y.permute(0, 2, 1))

        if self.add_time != 0:
            fact = 1. / self.add_time
            time_cov = fact * torch.arange(X.shape[1], device=X.device, dtype=X.dtype)[:, None] * fact * torch.arange(
                Y.shape[1], device=Y.device, dtype=Y.dtype)[None, :]
            k += time_cov[None, :, :]

        return self.scaling * k

    def Gram_matrix(self, X, Y):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """

        K = torch.einsum('ipk,jqk->ijpq', X, Y)

        if self.add_time != 0:
            fact = 1. / self.add_time
            time_cov = fact * torch.arange(X.shape[1], device=X.device, dtype=X.dtype)[:, None] * fact * torch.arange(
                Y.shape[1], device=Y.device, dtype=Y.dtype)[None, :]
            K += time_cov[None, None, :, :]
        return self.scaling * K


class RBFKernel():
    """RBF kernel k: R^d x R^d -> R"""

    def __init__(self, sigma, add_time=0, scaling=1):
        self.sigma = sigma
        self.add_time = add_time
        self.scaling = scaling

    def batch_kernel(self, X, Y):
        """Input:
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        A, M, N = X.shape[0], X.shape[1], Y.shape[1]

        Xs = torch.sum(X ** 2, dim=2)
        Ys = torch.sum(Y ** 2, dim=2)
        dist = -2. * torch.bmm(X, Y.permute(0, 2, 1))
        dist += torch.reshape(Xs, (A, M, 1)) + torch.reshape(Ys, (A, 1, N))

        if self.add_time != 0:
            fact = 1. / self.add_time
            time_component = (fact * torch.arange(X.shape[1], device=X.device, dtype=X.dtype)[:,
                                     None] - fact * torch.arange(Y.shape[1], device=Y.device, dtype=Y.dtype)[None,
                                                    :]) ** 2
            dist += time_component[None, :, :]

        return self.scaling * torch.exp(-dist / self.sigma)

    def Gram_matrix(self, X, Y):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        A, B, M, N = X.shape[0], Y.shape[0], X.shape[1], Y.shape[1]

        Xs = torch.sum(X ** 2, dim=2)
        Ys = torch.sum(Y ** 2, dim=2)
        dist = -2. * torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs, (A, 1, M, 1)) + torch.reshape(Ys, (1, B, 1, N))

        if self.add_time:
            fact = 1. / self.add_time
            time_component = (fact * torch.arange(X.shape[1], device=X.device, dtype=X.dtype)[:,
                                     None] - fact * torch.arange(Y.shape[1], device=Y.device, dtype=Y.dtype)[None,
                                                    :]) ** 2
            dist += time_component[None, None, :, :]

        return self.scaling * torch.exp(-dist / self.sigma)


# ===========================================================================================================

# ===========================================================================================================
# Main Signature Kernel class
#
# Now we can sequentialize the static kernels, and provide various
# functionalities including:
#
# * Batch kernel evaluation
# * Gram matrix computation
# * MMD computation
# ===========================================================================================================


class SigKernel():
    """Wrapper of the signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""

    def __init__(self, static_kernel, dyadic_order, _naive_solver=False):
        if isinstance(static_kernel, list):
            self.static_kernel = static_kernel[0]
            self.static_kernel_higher_order = static_kernel[1]
        else:
            self.static_kernel = static_kernel
            self.static_kernel_higher_order = static_kernel

        if isinstance(dyadic_order, list):
            self.dyadic_order = dyadic_order[0]
            self.dyadic_order_higher_order = dyadic_order[1]
        else:
            self.dyadic_order = dyadic_order
            self.dyadic_order_higher_order = dyadic_order

        self._naive_solver = _naive_solver

    def compute_kernel(self, X, Y, pad=True):
        """Input:
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output:
                  - vector k(X^i_T,Y^i_T) of shape (batch,)
        """
        return _SigKernel.apply(X, Y, self.static_kernel, self.dyadic_order, self._naive_solver, pad)

    def compute_Gram(self, X, Y, sym=False, return_sol_grid=False, pad=True):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
                  - sym: (bool) whether X=Y
                  - return_sol_grid: (bool) whether to return the full PDE solution,
                    or the solution at final times
           Output:
                  - matrix k(X^i_T,Y^j_T) of shape (batch_X, batch_Y)
        """
        return _SigKernelGram.apply(X, Y, self.static_kernel, self.dyadic_order, sym, self._naive_solver,
                                    return_sol_grid, pad)

    def compute_mmd(self, X, Y, estimator='ub', u_stat=False, pad=True):
        """
            Corresponds to Algorithm 3 or 5 in "Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes"
            Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim),
                  - estimator: (string) whether to compute a biased or unbiased estimator
                  - order: (int) the order of the MMD
                  - lambda_: (float) hyperparameter for the conditional KME estimator (to be specified if order=2)
           Output:
                  - scalar: MMD signature distance between samples X and samples Y
        """

        assert not Y.requires_grad, "the second input should not require grad"
        assert estimator == 'b' or estimator == 'ub', "the estimator should be 'b' or 'ub' "

        K_XX = self.compute_Gram(X, X, sym=True, pad=pad)
        K_YY = self.compute_Gram(Y, Y, sym=True, pad=pad)
        K_XY = self.compute_Gram(X, Y, sym=False, pad=pad)

        if estimator == 'b':
            return torch.mean(K_XX) + torch.mean(K_YY) - 2 * torch.mean(K_XY)
        else:
            K_XX_m = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))
            K_YY_m = (torch.sum(K_YY) - torch.sum(torch.diag(K_YY))) / (K_YY.shape[0] * (K_YY.shape[0] - 1.))

            if u_stat:
                K_XY_m = (torch.sum(K_XY) - torch.sum(torch.diag(K_XY))) / (K_XY.shape[0] * (K_XY.shape[0] - 1.))
            else:
                K_XY_m = torch.mean(K_XY)

            return K_XX_m + K_YY_m - 2. * K_XY_m


# Now let's actually implement the method which computes the signature kernel
# for a batch of n pairs paths {(x^i,y^i)}_i=1^{n}
#
# Here we also implement the backward pass for faster backpropagation
# ===========================================================================================================


class _SigKernel(torch.autograd.Function):
    """Signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, _naive_solver=False, pad=False):
        '''Corresponds to Algorithm 1 in "Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes" '''
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # computing dsdt k(X^i_s,Y^i_t)
        G_static = static_kernel.batch_kernel(X, Y)
        G_static_ = G_static[:, 1:, 1:] + G_static[:, :-1, :-1] - G_static[:, 1:, :-1] - G_static[:, :-1, 1:]
        G_static_ = tile(tile(G_static_, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        if pad:
            if math.log2(A).is_integer():
                G_static_ = F.pad(input=G_static_, pad=(0, 0, 0, 0, 0, 1), mode='constant', value=0)

        # if on GPU
        if X.device.type == 'cuda':

            assert max(MM + 1,
                       NN + 1) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM + 1, NN + 1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            K = torch.zeros((A, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
            K[:, 0, :] = 1.
            K[:, :, 0] = 1.

            # Compute the forward signature kernel
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](
                cuda.as_cuda_array(G_static_.detach()),
                MM + 1, NN + 1, n_anti_diagonals,
                cuda.as_cuda_array(K), _naive_solver)
            K = K[:, :-1, :-1]

        # if on CPU
        else:
            K = torch.tensor(sig_kernel_batch_varpar(G_static_.detach().numpy(), _naive_solver, pad), dtype=G_static.dtype,
                             device=G_static.device)

        ctx.save_for_backward(X, Y, G_static, K)
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver
        ctx.pad = pad

        return K[:, -1, -1]

    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G_static, K = ctx.saved_tensors
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver
        pad = ctx.pad

        G_static_ = G_static[:, 1:, 1:] + G_static[:, :-1, :-1] - G_static[:, 1:, :-1] - G_static[:, :-1, 1:]
        G_static_ = tile(tile(G_static_, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # computing dsdt k(X_rev^i_s,Y_rev^i_t) for variation of parameters
        G_static_rev = flip(flip(G_static_, dim=1), dim=2)

        if pad:
            if math.log2(A).is_integer():
                G_static_rev = F.pad(input=G_static_rev, pad=(0, 0, 0, 0, 0, 1), mode='constant', value=0)

        # if on GPU
        if X.device.type == 'cuda':

            # Prepare the tensor of output solutions to the PDE (backward)
            K_rev = torch.zeros((A, MM + 2, NN + 2), device=G_static_rev.device, dtype=G_static_rev.dtype)
            K_rev[:, 0, :] = 1.
            K_rev[:, :, 0] = 1.

            # cuda parameters
            threads_per_block = max(MM, NN)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](
                cuda.as_cuda_array(G_static_rev.detach()),
                MM + 1, NN + 1, n_anti_diagonals,
                cuda.as_cuda_array(K_rev), _naive_solver)

            K_rev = K_rev[:, :-1, :-1]

        # if on CPU
        else:
            K_rev = torch.tensor(sig_kernel_batch_varpar(G_static_rev.detach().numpy(), _naive_solver, pad),
                                 dtype=G_static.dtype, device=G_static.device)

        K_rev = flip(flip(K_rev, dim=1), dim=2)
        KK = K[:, :-1, :-1] * K_rev[:, 1:, 1:]

        # finite difference step
        h = 1e-9

        Xh = X[:, :, :, None] + h * torch.eye(D, dtype=X.dtype, device=X.device)[None, None, :]
        Xh = Xh.permute(0, 1, 3, 2)
        Xh = Xh.reshape(A, M * D, D)

        G_h = static_kernel.batch_kernel(Xh, Y)
        G_h = G_h.reshape(A, M, D, N)
        G_h = G_h.permute(0, 1, 3, 2)

        Diff_1 = G_h[:, 1:, 1:, :] - G_h[:, 1:, :-1, :] - (G_static[:, 1:, 1:])[:, :, :, None] + (G_static[:, 1:, :-1])[
                                                                                                 :, :, :, None]
        Diff_1 = tile(tile(Diff_1, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2, 2 ** dyadic_order) / float(
            2 ** dyadic_order)
        Diff_2 = G_h[:, 1:, 1:, :] - G_h[:, 1:, :-1, :] - (G_static[:, 1:, 1:])[:, :, :, None] + (G_static[:, 1:, :-1])[
                                                                                                 :, :, :, None]
        Diff_2 += - G_h[:, :-1, 1:, :] + G_h[:, :-1, :-1, :] + (G_static[:, :-1, 1:])[:, :, :, None] - (G_static[:, :-1,
                                                                                                        :-1])[:, :, :,
                                                                                                       None]
        Diff_2 = tile(tile(Diff_2, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2, 2 ** dyadic_order) / float(
            2 ** dyadic_order)

        grad_1 = (KK[:, :, :, None] * Diff_1) / h
        grad_2 = (KK[:, :, :, None] * Diff_2) / h

        grad_1 = torch.sum(grad_1, axis=2)
        grad_1 = torch.sum(grad_1.reshape(A, M - 1, 2 ** dyadic_order, D), axis=2)
        grad_2 = torch.sum(grad_2, axis=2)
        grad_2 = torch.sum(grad_2.reshape(A, M - 1, 2 ** dyadic_order, D), axis=2)

        grad_prev = grad_1[:, :-1, :] + grad_2[:, 1:, :]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, 1, D), dtype=X.dtype, device=X.device), grad_1[:, 1:, :]], dim=1)  # /
        grad_incr = grad_prev - grad_1[:, 1:, :]
        grad_points = torch.cat(
            [(grad_2[:, 0, :] - grad_1[:, 0, :])[:, None, :], grad_incr, grad_1[:, -1, :][:, None, :]], dim=1)

        if Y.requires_grad:
            grad_points *= 2

        return grad_output[:, None, None] * grad_points, None, None, None, None, None


# ===========================================================================================================
# Now let's actually implement the method which computes the classical (order 1) signature kernel Gram matrix
# for n x m pairs paths {(x^i,y^j)}_{i,j=1}^{n,m}
#
# Here we also implement the backward pass for faster backpropagation
# ===========================================================================================================

class _SigKernelGram(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, sym=False, _naive_solver=False, return_sol_grid=False, pad=False):
        '''Corresponds to Algorithm 2 in "Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes" '''

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        pad_dim1 = False
        pad_dim2 = False

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # computing dsdt k(X^i_s,Y^j_t)
        G_static = static_kernel.Gram_matrix(X, Y)
        G_static_ = G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1] - G_static[:, :, 1:, :-1] - G_static[:, :, :-1,
                                                                                                  1:]
        G_static_ = tile(tile(G_static_, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        if pad:
            if math.log2(B).is_integer():
                G_static_ = F.pad(input=G_static_, pad=(0, 0, 0, 0, 0, 1, 0, 0), mode='constant', value=0)
                pad_dim2 = True
            if math.log2(A).is_integer():
                G_static_ = F.pad(input=G_static_, pad=(0, 0, 0, 0, 0, 0, 0, 1), mode='constant', value=0)
                pad_dim1 = True


        # if on GPU
        if X.device.type == 'cuda':
            assert max(MM,
                       NN) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM + 1, NN + 1)
            n_anti_diagonals = 2 * threads_per_block - 1


            # Prepare the tensor of output solutions to the PDE (forward)
            G = torch.zeros((A, B, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
            G[:, :, 0, :] = 1.
            G[:, :, :, 0] = 1.


            # Run the CUDA kernel.
            blockspergrid = (A, B)

            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](
                cuda.as_cuda_array(G_static_.detach()),
                MM + 1, NN + 1, n_anti_diagonals,
                cuda.as_cuda_array(G), _naive_solver)
            G = G[:, :, :-1, :-1]


        else:
            G = torch.tensor(sig_kernel_Gram_varpar(G_static_.detach().numpy(), sym, _naive_solver, pad_dim1, pad_dim2),
                             dtype=G_static.dtype, device=G_static.device)

        ctx.save_for_backward(X, Y, G, G_static)
        ctx.sym = sym
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver
        ctx.pad = pad


        if not return_sol_grid:
            return G[:, :, -1, -1]
        else:
            return G[:, :, ::2 ** dyadic_order, ::2 ** dyadic_order]

    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G, G_static = ctx.saved_tensors
        sym = ctx.sym
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver
        pad = ctx.pad

        G_static_ = G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1] - G_static[:, :, 1:, :-1] - G_static[:, :, :-1,
                                                                                                  1:]
        G_static_ = tile(tile(G_static_, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # computing dsdt k(X_rev^i_s,Y_rev^j_t) for variation of parameters
        G_static_rev = flip(flip(G_static_, dim=2), dim=3)

        pad_dim1 = False
        pad_dim2 = False

        if pad:
            if math.log2(B).is_integer():
                G_static_rev = F.pad(input=G_static_rev, pad=(0, 0, 0, 0, 0, 1, 0, 0), mode='constant', value=0)
                pad_dim2 = True
            if math.log2(A).is_integer():
                G_static_rev = F.pad(input=G_static_rev, pad=(0, 0, 0, 0, 0, 0, 0, 1), mode='constant', value=0)
                pad_dim1 = True

        # if on GPU
        if X.device.type == 'cuda':

            # Prepare the tensor of output solutions to the PDE (backward)
            G_rev = torch.zeros((A, B, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
            G_rev[:, :, 0, :] = 1.
            G_rev[:, :, :, 0] = 1.

            # cuda parameters
            threads_per_block = max(MM + 1, NN + 1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            blockspergrid = (A, B)

            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](
                cuda.as_cuda_array(G_static_rev.detach()),
                MM + 1, NN + 1, n_anti_diagonals,
                cuda.as_cuda_array(G_rev), _naive_solver)

            G_rev = G_rev[:, :, :-1, :-1]

        # if on CPU
        else:
            G_rev = torch.tensor(sig_kernel_Gram_varpar(G_static_rev.detach().numpy(), sym, _naive_solver, pad_dim1, pad_dim2),
                                 dtype=G_static.dtype, device=G_static.device)


        G_rev = flip(flip(G_rev, dim=2), dim=3)
        GG = G[:, :, :-1, :-1] * G_rev[:, :, 1:, 1:]

        # finite difference step
        h = 1e-9

        Xh = X[:, :, :, None] + h * torch.eye(D, dtype=X.dtype, device=X.device)[None, None, :]
        Xh = Xh.permute(0, 1, 3, 2)
        Xh = Xh.reshape(A, M * D, D)

        G_h = static_kernel.Gram_matrix(Xh, Y)
        G_h = G_h.reshape(A, B, M, D, N)
        G_h = G_h.permute(0, 1, 2, 4, 3)

        Diff_1 = G_h[:, :, 1:, 1:, :] - G_h[:, :, 1:, :-1, :] - (G_static[:, :, 1:, 1:])[:, :, :, :, None] + (
                                                                                                             G_static[:,
                                                                                                             :, 1:,
                                                                                                             :-1])[:, :,
                                                                                                             :, :, None]
        Diff_1 = tile(tile(Diff_1, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3, 2 ** dyadic_order) / float(
            2 ** dyadic_order)
        Diff_2 = G_h[:, :, 1:, 1:, :] - G_h[:, :, 1:, :-1, :] - (G_static[:, :, 1:, 1:])[:, :, :, :, None] + (
                                                                                                             G_static[:,
                                                                                                             :, 1:,
                                                                                                             :-1])[:, :,
                                                                                                             :, :, None]
        Diff_2 += - G_h[:, :, :-1, 1:, :] + G_h[:, :, :-1, :-1, :] + (G_static[:, :, :-1, 1:])[:, :, :, :, None] - (
                                                                                                                   G_static[
                                                                                                                   :, :,
                                                                                                                   :-1,
                                                                                                                   :-1])[
                                                                                                                   :, :,
                                                                                                                   :, :,
                                                                                                                   None]
        Diff_2 = tile(tile(Diff_2, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3, 2 ** dyadic_order) / float(
            2 ** dyadic_order)

        grad_1 = (GG[:, :, :, :, None] * Diff_1) / h
        grad_2 = (GG[:, :, :, :, None] * Diff_2) / h

        grad_1 = torch.sum(grad_1, axis=3)
        grad_1 = torch.sum(grad_1.reshape(A, B, M - 1, 2 ** dyadic_order, D), axis=3)
        grad_2 = torch.sum(grad_2, axis=3)
        grad_2 = torch.sum(grad_2.reshape(A, B, M - 1, 2 ** dyadic_order, D), axis=3)

        grad_prev = grad_1[:, :, :-1, :] + grad_2[:, :, 1:, :]
        grad_incr = grad_prev - grad_1[:, :, 1:, :]
        grad_points = torch.cat(
            [(grad_2[:, :, 0, :] - grad_1[:, :, 0, :])[:, :, None, :], grad_incr, grad_1[:, :, -1, :][:, :, None, :]],
            dim=2)

        if sym:
            grad = (grad_output[:, :, None, None] * grad_points + grad_output.t()[:, :, None, None] * grad_points).sum(
                dim=1)
            return grad, None, None, None, None, None, None, None
        grad = (grad_output[:, :, None, None] * grad_points).sum(dim=1)
        return grad, None, None, None, None, None, None, None

# ===========================================================================================================


# ===========================================================================================================
# Various utility functions
# ===========================================================================================================
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


# ===========================================================================================================
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        a.device)
    return torch.index_select(a, dim, order_index)
# ===========================================================================================================