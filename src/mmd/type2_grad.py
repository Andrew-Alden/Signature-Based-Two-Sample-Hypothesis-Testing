from src.mmd.level_functions import level_k_contribution
from src.mmd.rbf_mmd import *
import numpy as np
import torch
from tqdm import tqdm


def sigmoid_grad(crit_val, d, xi):

    """
    Gradient of sigmoid function.
    :param crit_val: Critical value.
    :param d: Test statistic.
    :param xi: Smoothing parameter.
    :return: Gradient of sigmoid function with parameter xi evaluated at (crit_val - d).
    """

    val = xi * (crit_val - d)
    if np.abs(val) > 700:
        return 0
    else:
        numerator = xi * np.exp(-val)
        denominator = (1 + np.exp(-val))**2
        return numerator/denominator


def distance_grad(x, y, _k, scaling, scaling_mult=True, estimator='ub', RBF=False, sigma=None):

    """
    Gradient of test statistic.
    :param x: Collection of paths.
    :param y: Collection of paths.
    :param _k: Truncation level.
    :param scaling: Scaling value.
    :param scaling_mult: Boolean indicating whether the scaling value is actually scaling**2. Default is True.
    :param estimator: String indicating whether unbiased or biased estimator. Default is 'ub'.
    :param RBF: Boolean indicating whether to use RBF kernel as static kernel. Default is False.
    :param sigma: Sigma value of RBF kernel. Default is None.
    :return: Gradient of test statistic.
    """
    
    unbiased = False

    if estimator == 'ub':
        unbiased = True

    s = 0
    if not RBF:
        phi_l = lambda l: np.power(scaling, 2*(l-1)) * l
        for lvl in range(_k):
            s += level_k_contribution(x, y, lvl+1, phik=phi_l, unbiased=unbiased)
    else:
        phi_l = lambda l: np.power(scaling, 2*(l-1)) * l
        s = rbfsig_level_mmd(x, y, _k, sigma, phik=phi_l, unbiased=unbiased).cpu()
        
    factor = 1
    if scaling_mult:
        factor = 2*scaling
    return factor*s
        

def empirical_cdf_grad(h0_paths, h1_paths, crit_val, distances, xi, random_indices, scaling, _k, const_val,
                       scaling_mult=False, scale=False, print_res=False, device='cpu', estimator='ub', RBF=False,
                       sigma=None, verbose=True):

    """
    Approximate gradient of empirical CDF.
    :param h0_paths: H0 paths.
    :param h1_paths: H1 paths.
    :param crit_val: Critical value.
    :param distances: Collection of test statistic values over which to define the CDF.
    :param xi: Sigmoid smoothing parameter.
    :param random_indices: Collection of B index sets.
    :param scaling: Scaling value.
    :param _k: Truncation level.
    :param const_val: Value which is kept constant in gradient computation.
    :param scaling_mult: Boolean indicating whether the scaling value is actually scaling**2 when computing gradient of
                         test statistic. Default is True.
    :param scale: Boolean indicating whether the scaling value is actually scaling**2. Default is True.
    :param device: Device on which to perform calculations. Default is 'cpu'.
    :param estimator: String indicating whether unbiased or biased estimator. Default is 'ub'.
    :param RBF: Boolean indicating whether to use RBF kernel as static kernel. Default is False.
    :param sigma: Sigma value of RBF kernel. Default is None.
    :param verbose: Boolean indicating whether to display intermediate results. Default is True.
    :return: Gradient of empirical CDF evaluated at const_val.
    """

    if type(distances) == type([]):
        distances = np.asarray(distances)
    B = distances.shape[0]

    vals = []

    if verbose:
        itr = tqdm(range(B))
    else:
        itr = range(B)
    
    for i in itr:
        
        dist_grad = distance_grad(h0_paths[random_indices[i][0]], 
                                  h1_paths[random_indices[i][1]],
                                  _k, 
                                  scaling, 
                                  scaling_mult=scaling_mult, 
                                  estimator=estimator, 
                                  RBF=RBF,
                                  sigma=sigma)
        
        if print_res:
            print(sigmoid_grad(crit_val, distances[i], xi))
            print(const_val)
            print(dist_grad)
            print(f'{"*"*50}')
        vals.append(sigmoid_grad(crit_val, distances[i], xi) * (const_val - dist_grad))

    factor = 1
    if scale:
        factor = 2 * scaling
    return factor*np.mean(vals)


def empirical_critval_grad(h0_paths, crit_val, h0_dists, xi_2, random_indices, scaling, _k, scaling_mult=False, 
                           scale=False, device='cpu', estimator='ub', RBF=False, sigma=None, verbose=True):

    """
    Gradient of critical value.
    :param h0_paths: H0 paths.
    :param crit_val: Critical value.
    :param h0_dists: Collection of test statistic values under H0.
    :param xi_2: Sigmoid smoothing parameter.
    :param random_indices: Collection of B index sets.
    :param scaling: Scaling value.
    :param _k: Truncation level.
    :param scaling_mult: Boolean indicating whether the scaling value is actually scaling**2 when computing gradient of
                         test statistic. Default is True.
    :param scale: Boolean indicating whether the scaling value is actually scaling**2. Default is True.
    :param device: Device on which to perform calculations. Default is 'cpu'.
    :param estimator: String indicating whether unbiased or biased estimator. Default is 'ub'.
    :param RBF: Boolean indicating whether to use RBF kernel as static kernel. Default is False.
    :param sigma: Sigma value of RBF kernel. Default is None.
    :param verbose: Boolean indicating whether to display intermediate results. Default is True.
    :return: Gradient of critical value.
    """

    if type(h0_dists) == type([]):
        h0_dists = np.asarray(h0_dists)
    B = h0_dists.shape[0]

    val_1 = empirical_cdf_grad(h0_paths, h0_paths, crit_val, h0_dists, xi_2, random_indices[:, :2, :], scaling, _k, 0, 
                               scaling_mult=scaling_mult, scale=False, device=device, estimator=estimator, RBF=RBF,
                               sigma=sigma, verbose=verbose)
    vals_2 = []
    for i in range(B):
        vals_2.append(sigmoid_grad(crit_val, h0_dists[i], xi_2))
    val_2 = np.mean(vals_2)
    

    factor = -1
    if scale:
        factor = -2*scaling
        
    return factor * val_1 * 1/val_2
    

def scaling_grad(h0_paths, h1_paths, crit_val, h1_dists, h0_dists, xi_1, random_indices, scaling, _k, scaling_mult=False, 
                 print_res=False, crit_val_grad_input=None, xi_2=None, device='cpu', estimator='ub', RBF=False, sigma=None,
                 verbose=True):

    """
    Approximate gradient of probability of a Type 2 error occurring.
    :param h0_paths: H0 paths.
    :param h1_paths: H1 paths.
    :param crit_val: Critical value.
    :param h1_dists: Collection of test statistic values under H1.
    :param h0_dists: Collection of test statistic values under H0.
    :param xi_1: Sigmoid smoothing parameter for computation of gradient of P[Type 2 Error].
    :param random_indices: Collection of B index sets.
    :param scaling: Scaling value.
    :param _k: Truncation level.
    :param scaling_mult: Boolean indicating whether the scaling value is actually scaling**2 when computing gradient of
                         test statistic. Default is True.
    :param print_res: Boolean indicating whether to print result. Default is False.
    :param crit_val_grad_input: Gradient of critical value. If this has already been calculated, by passing it as input
                                it saves time in the computation since it is not re-computed. Default is None.
    :param xi_2: Sigmoid smoothing parameter for computation of the gradient of critical value. If the value is None
                 then use the smoothing parameter xi_1. Defualt is None.
    :param device: Device on which to perform calculations. Default is 'cpu'.
    :param estimator: String indicating whether unbiased or biased estimator. Default is 'ub'.
    :param RBF: Boolean indicating whether to use RBF kernel as static kernel. Default is False.
    :param sigma: Sigma value of RBF kernel. Default is None.
    :param verbose: Boolean indicating whether to display intermediate results. Default is True.
    :return: Gradient of probability of a Type 2 error occurring.
    """

    if xi_2 is None:
        xi_2 = xi_1

    if RBF:
        assert sigma is not None

    if crit_val_grad_input is None:
        crit_val_grad_input = empirical_critval_grad(h0_paths, crit_val, h0_dists, xi_2, random_indices[:, :2, :], scaling, _k, 
                                                     scaling_mult=scaling_mult, scale=False, device=device, estimator=estimator, 
                                                     RBF=RBF, sigma=sigma, verbose=verbose)
    type2_grad = empirical_cdf_grad(h0_paths, h1_paths, crit_val, h1_dists, xi_1, random_indices[:, 0::2, :], scaling, 
                                    _k, crit_val_grad_input, scaling_mult=scaling_mult, scale=True, print_res=print_res, 
                                    device=device, estimator=estimator, RBF=RBF, sigma=sigma, verbose=verbose)


    if print_res:
        print(f'Critical Value Grad: {np.round(crit_val_grad, 5)}')
        print(f'Type 2 Grad: {np.round(type2_grad, 5)}')
    
    return type2_grad, crit_val_grad_input


def gradient_descent(h0_paths, h1_paths, crit_val, h1_dists, h0_dists, xi_1, random_indices, scaling, _k, lr, thresh, current_iter, 
                     maxiter=1000, scaling_mult=False, crit_val_grad_input=None, xi_2=None, device='cpu', estimator='ub', RBF=False,
                     sigma=None, scaling_fn=lambda s: s, verbose=True, print_res=False):

    """
    Gradient descent algorithm.
    :param h0_paths: H0 paths.
    :param h1_paths: H1 paths.
    :param crit_val: Critical value.
    :param h1_dists: Collection of test statistic values under H1.
    :param h0_dists: Collection of test statistic values under H0.
    :param xi_1: Sigmoid smoothing parameter for computation of gradient of P[Type 2 Error].
    :param random_indices: Collection of B index sets.
    :param scaling: Scaling value.
    :param _k: Truncation level.
    :param lr: Learning rate function. Input to this function is iteration number.
    :param thresh: Threshold value which determines stopping criteria of algorithm.
    :param current_iter: Current iteration.
    :param maxiter: Maximum number of iterations. Default is 1000.
    :param scaling_mult: Boolean indicating whether the scaling value is actually scaling**2 when computing gradient of
                         test statistic. Default is True.
    :param crit_val_grad_input: Gradient of critical value. If this has already been calculated, by passing it as input
                                it saves time in the computation since it is not re-computed. Default is None.
    :param xi_2: Sigmoid smoothing parameter for computation of the gradient of critical value. If the value is None
                 then use the smoothing parameter xi_1. Defualt is None.
    :param device: Device on which to perform calculations. Default is 'cpu'.
    :param estimator: String indicating whether unbiased or biased estimator. Default is 'ub'.
    :param RBF: Boolean indicating whether to use RBF kernel as static kernel. Default is False.
    :param sigma: Sigma value of RBF kernel. Default is None.
    :param scaling_fn: Scaling function. Default is identity function.
    :param verbose: Boolean indicating whether to display intermediate results. Default is True.
    :param print_res: Boolean indicating whether to print result. Default is False.
    :return: New scaling value.
    """

    if print_res:
        print(f'{"*"*50}')
        print(f'Old Scaling: {np.round(scaling, 4)}')
    grad, _ = scaling_grad(h0_paths, h1_paths, crit_val, h1_dists, h0_dists, xi_1, random_indices, scaling_fn(scaling), _k, 
                           scaling_mult=scaling_mult, crit_val_grad_input=crit_val_grad_input, xi_2=xi_2, device=device,
                           estimator=estimator, RBF=RBF, sigma=sigma, verbose=verbose, print_res=print_res)
        
    new_scaling = scaling - lr(current_iter) * np.clip(grad, a_min=-5, a_max=5)

    done = (np.abs(new_scaling - scaling) < thresh)

    if print_res:
        print(f'Grad: {np.round(grad, 4)}')
        print(f'New Scaling: {np.round(new_scaling, 4)}')

    if current_iter == maxiter:
        done = True
    
    return new_scaling, done, grad