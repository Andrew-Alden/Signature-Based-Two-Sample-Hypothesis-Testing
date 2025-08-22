import math
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import norm

from src.mmd.distribution_functions import return_mmd_distributions, expected_type2_error, get_level_values, generate_error_probs_linear_kernel
from src.mmd.two_sample_stats import ppf, scale, rate
from src.StochasticProcesses.ScaledBM import generate_scaled_brownian_motion_paths



# =======================================Unbiased==================================================================
def Exp_Gamma_1_ub(sigma, beta, T=1):

    """
    Expected value of level 1 contribution to unbiased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param T. Terminal time. Default is 1.
    :return: Expected value of level 1 contribution to unbiased estimator.
    """
    return 0

def Exp_Gamma_2_ub(sigma, beta, T=1):

    """
    Expected value of level 2 contribution to unbiased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param T. Terminal time. Default is 1.
    :return: Expected value of level 2 contribution to unbiased estimator.
    """

    return (sigma**2 - beta**2)**2 * 1/4 * T**2

def Exp_Gamma_3_ub(sig, beta, T=1):

    """
    Expected value of level 3 contribution to unbiased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param T. Terminal time. Default is 1.
    :return: Expected value of level 3 contribution to unbiased estimator.
    """

    return T**4 * 1/8 * (sig**2 - beta**2)**2

def Var_Gamma_1_ub(sigma, beta, N, T=1):

    """
    Variance of level 1 contribution to unbiased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of level 1 contribution to unbiased estimator.
    """

    return 2 * sigma**4 * T**2 * 1/(N*(N-1)) + 2 * beta**4 * T**2 * 1/(N*(N-1)) + 4 * sigma**2 * beta**2 * T**2 * 1/N * 1/N


def Var_Gamma_2_X_X_ub(sigma, N, T=1):

    """
    Variance of Lambda(X, X) term of the level 2 contribution to unbiased estimator.
    :param sigma: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of Lambda(X, X) term of the level 2 contribution to unbiased estimator.
    """

    return sigma**4 * 2 * T**4 * 1/(N*(N-1)) * (5/18 * T**2 + sigma**4 * 1/2) + (N-2) * 1/2 * 1/(N*(N-1)) * T**4 * sigma**8

def Var_Gamma_2_X_Y_ub(sigma, beta, N, T=1):

    """
    Variance of Lambda(X, Y) term of the level 2 contribution to unbiased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of Lambda(X, Y) term of the level 2 contribution to unbiased estimator.
    """

    return sigma**2 * beta**2 * T**4 * 1/N * 1/N * (5/18 * T**2 + sigma**2 * beta**2 * 1/2) + (N-1) * 1/N * 1/N * sigma**4 * beta**4 * T**4 * 1/4

def CoVar_Gamma_2_ub(sigma, beta, N, T=1):

    """
    Covariance of Lambda(X, Y) term of the level 2 contribution to unbiased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Covariance of Lambda(X, Y) term of the level 2 contribution to unbiased estimator.
    """

    return 1/N * T**4 * sigma**6 * beta**2

def Var_Gamma_2_ub(sigma, beta, N, T=1):

    """
    Variance of level 2 contribution to unbiased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of level 2 contribution to unbiased estimator.
    """

    return Var_Gamma_2_X_X_ub(sigma, N, T) + Var_Gamma_2_X_X_ub(beta, N, T) + 4*Var_Gamma_2_X_Y_ub(sigma, beta, N, T) - CoVar_Gamma_2_ub(sigma, beta, N, T) - CoVar_Gamma_2_ub(beta, sigma, N, T)

def Exp_MMD_3_ub(sigma, beta, T=1):

    """
    Expected value of squared sig-MMD unbiased estimator with truncation level of 3.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param T. Terminal time. Default is 1.
    :return: Expected value of squared sig-MMD unbiased estimator with truncation level of 3.
    """

    return Exp_Gamma_2_ub(sigma, beta, T) + Exp_Gamma_3_ub(sigma, beta, T)
    
# =======================================Biased==================================================================

def Exp_Gamma_1_b(sigma, beta, N, T=1):

    """
    Expected value of level 1 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Expected value of level 1 contribution to biased estimator.
    """

    return 1/N * T * (sigma**2 + beta**2)


def Exp_Gamma_2_b(sigma, beta, N, T=1):

    """
    Expected value of Lambda(X, Y) term of level 3 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Expected value of Lambda(X, Y) term of level 3 contribution to biased estimator.
    """

    return (N-1) * 1/N * T**2 * 1/4 * (sigma**2 - beta**2)**2 + 1/N * T**2 * (2 * 1/3 * sigma**2 * T + 3 * 1/4 * sigma**4 + 2 * 1/3 * beta**2 * T + 3 * 1/4 * beta**4 - 1/2 * sigma**2 * beta**2)


def Exp_Gamma_3_Cross(param, T=1):

    """
    Expected value of level 3 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Expected value of level 1 contribution to biased estimator.
    """

    return 2/15 * param**2 * T**5 + 3/8 * param**4 * T**4 + 15/36 * T**3 * param**6

def Exp_Gamma_3_b(sigma, beta, N, T=1):

    """
    Expected value of level 3 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Expected value of level 3 contribution to biased estimator.
    """

    return (N-1) * 1/N * Exp_Gamma_3_ub(sigma, beta, T=T) + 1/N * (Exp_Gamma_3_Cross(sigma, T=T) + Exp_Gamma_3_Cross(beta, T=T) - 1/4 * sigma**2 * beta**2 * T**4)


def Var_Gamma_1_b(sigma, beta, N, T=1):

    """
    Variance of level 1 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of level 1 contribution to biased estimator.
    """

    return 1/(N**2) * 2 * T**2 * (sigma**4 + beta**4 + 2 * sigma**2 * beta**2)

def Var_off_diag_terms(sigma, N, T=1):

    """
    Variance of level 2 off-diagonal terms to the biased estimator.
    :param sigma: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of level 2 off-diagonal terms to the biased estimator.
    """

    return sigma**4 * 2 * T**4 * (N-1) * 1/(N**3) * (2/9 * T**2 + sigma**4 * 1/2 + T**2 * 1/18) + (N-1) * (N-2) * 1/2 * 1/(N**3) * T**4 * sigma**8

def Var_diag_terms(sigma, N, T=1):

    """
    Variance of level 2 diagonal terms to the biased estimator.
    :param sigma: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of level 2 diagonal terms to the biased estimator.
    """

    return 1/(N**3) * sigma**4 * T**4 * (5/9 * T**2 + 6 * sigma**4 + 3 * sigma**2 * T)

def Cov_diag_off_diag_terms(sigma, N, T=1):

    """
    Covariance of level 2 off-diagonal terms to the biased estimator.
    :param sigma: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Covariance of level 2 off-diagonal terms to the biased estimator.
    """

    return (N-1) * 1/(N**3) * Cov_Gamma_2_b(sigma, sigma, N, T)

def Cov_Gamma_2_b(sigma, beta, N, T=1):

    """
    Covariance of level 2 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Covariance of level 1 contribution to biased estimator.
    """

    return 3/4 * sigma**6 * beta**2 * T**4 + 1/4 * sigma**4 * beta**2 * T**5

def Var_Gamma_2_X_X_b(sigma, N, T=1):

    """
    Variance of Lambda(X, X) term of the level 2 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of Lambda(X, X) term of the level 2 contribution to biased estimator.
    """

    return Var_off_diag_terms(sigma, N, T) + Var_diag_terms(sigma, N, T) + 4*Cov_diag_off_diag_terms(sigma, N, T)

def Var_Gamma_2_X_Y(sigma, beta, N, T=1):

    """
    Variance of Lambda(X, Y) term of the level 2 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of Lambda(X, Y) term of the level 2 contribution to biased estimator.
    """

    return sigma**2 * beta**2 * T**4 * 1/N * 1/N * (2/9 * T**2 + sigma**2 * beta**2 * 1/2 + T**2 * 1/18) + (N-1) * 1/N * 1/N * sigma**4 * beta**4 * T**4 * 1/4

def CoVar_Gamma_2_b(sigma, beta, N, T=1):

    """
    Full covariance of level 2 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Full covariance of level 1 contribution to biased estimator.
    """

    return (N-1) * 1/(N**2) * 1/4 * sigma**6 * T**4 * beta**2 + 1/(N**2) * Cov_Gamma_2_b(sigma, beta, N, T)

def Var_Gamma_2_b(sigma, beta, N, T=1):

    """
    Variance of level 2 contribution to biased estimator.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of level 2 contribution to biased estimator.
    """

    return Var_Gamma_2_X_X_b(sigma, N, T) + Var_Gamma_2_X_X_b(beta, N, T) + 4*Var_Gamma_2_X_Y(sigma, beta, N, T) - 4*CoVar_Gamma_2_b(sigma, beta, N, T) - 4*CoVar_Gamma_2_b(beta, sigma, N, T)


def Exp_MMD_2_b(sigma, beta, N, T=1):

    """
    Expected value of squared sig-MMD biased estimator with truncation level of 2.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Expected value of squared sig-MMD biased estimator with truncation level of 2.
    """

    return Exp_Gamma_1_b(sigma, beta, N, T) + Exp_Gamma_2_b(sigma, beta, N, T) + Exp_Gamma_3_b(sigma, beta, N, T)

def CoVar_MM_2_X_X_b(sigma, N, T=1):

    """
    Covariance of (X, X) terms in the computation of the squared sig-MMD biased estimator with truncation level of 2.
    :param sigma: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Covariance of (X, X) terms in the computation of the squared sig-MMD biased estimator with truncation level
             of 2.
    """

    return sigma**4 * T**3 * 2 * 1/(N**2) * (T + (N+2) * 1/N * sigma**2)

def CoVar_MM_2_X_Y_b(sigma, beta, N, T=1):

    """
    Covariance of (X, Y) terms in the computation of the squared sig-MMD biased estimator with truncation level of 2.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Covariance of (X, Y) terms in the computation of the squared sig-MMD biased estimator with truncation level
             of 2.
    """

    return 2 * T * sigma**2 * beta**2 * 1/(N**2)

def Var_MMD_2_b(sigma, beta, N, T=1):

    """
    Variance of squared sig-MMD biased estimator with truncation level of 2.
    :param sigma: Volatility parameter.
    :param beta: Volatility parameter.
    :param N: Number of samples.
    :param T. Terminal time. Default is 1.
    :return: Variance of squared sig-MMD biased estimator with truncation level of 2.
    """

    return Var_Gamma_1_b(sigma, beta, N, T) + Var_Gamma_2_b(sigma, beta, N, T) + CoVar_MM_2_X_X_b(sigma, N, T) + CoVar_MM_2_X_X_b(beta, N, T) - CoVar_MM_2_X_Y_b(sigma**2, beta, N, T**3) - CoVar_MM_2_X_Y_b(sigma, beta**2, N, T**3) + 2 * CoVar_MM_2_X_Y_b(sigma, beta, N, T**4)




# =======================================Type 2 Error==================================================================

trunc_vol_H_A_dist = lambda sigma, beta, T: T**4 * (sigma**2 - beta**2)**2 * (sigma**4 + beta**4)

def type2errors_sim(param_list_dict, h0_param_key, iterator_keys, n_atoms, signature_kernel, alpha=0.05, n_paths=128,
                    estimator='ub', num_sim=100, verbose1=False, verbose2=False, device='cpu', T=1, 
                    path_bank_size=10000, grid_points=32):

    """
    Simulate P[Type 2 Error] using empirical simulation and closed-form formulae.
    :param param_list_dict: Dictionary of scaled Brownian motion parameters.
    :param h0_param_key: Parameter key corresponding to volatility under H0.
    :param iterator_keys: List of iterator keys for nested iterations. First iterate through key at iterator_keys[0] and
                          then through key at iterator_keys[1].
    :param n_atoms: Number of simulations.
    :param signature_kernel: Signature kernel object to compute sig-MMD.
    :param alpha: Level of the test. Default is 0.05.
    :param n_paths: Collection size. Default is 128.
    :param estimator: Type of estimator. Default is 'ub'.
    :param num_sim: Number of simulations. Default is 100.
    :param verbose1: Flag indicating whether to display progress in terms of iterator 1. Default is False.
    :param verbose2: Flag indicating whether to display progress in terms of iterator 2. Default is False.
    :param device: Device on which to perform simulations. Default is 'cpu'.
    :param T: Terminal time. Default is 1.0.
    :param path_bank_size: Number of available paths. Default is 10000.
    :param grid_points: Number of discrete data points excluding initial value. Default is 32.
    :return: Dictionary containing P[Type 2 Error] with key given by parameter values corresponding to iterator_keys[0].
             Values of dictionary are tuples containing P[Type 2 Error] corresponding to empirical simulations,
             asymptotic calculations, and Gaussian distribution. The errors corresponding to the empirical simulations
             are pairs, with first element being the mean and second element being the standard deviation.
    """
    
    error_dict = {}

    if verbose1:
        iterator1 = tqdm(param_list_dict[iterator_keys[0]])
    else:
        iterator1 = param_list_dict[iterator_keys[0]]

    if verbose2:
        iterator2 = tqdm(param_list_dict[iterator_keys[1]])
    else:
        iterator2 = param_list_dict[iterator_keys[1]]

    with torch.no_grad():

        for param1 in iterator1:

            type2_errors_sim = []
            crit_vals_sim = []
            type2_errors_analytical = []
            type2_errors_gaussian = []

            for param2 in iterator2:

                if h0_param_key == iterator_keys[0]:
                    h0_param = param1
                    h1_param = param2
                else:
                    h0_param = param2
                    h1_param = param1

                mean = Exp_MMD_2_b(h0_param, h0_param, n_paths, T=T)
                var  = Var_MMD_2_b(h0_param, h0_param, n_paths, T=T)
                mu    = scale(mean, var, n_paths)
                theta = rate(mean, var, n_paths)
                threshold_est = ppf(1-alpha, mu, theta, n_paths)

                errors_list = []

                for _ in range(num_sim):

                    h0_paths = torch.transpose(torch.from_numpy(
                        generate_scaled_brownian_motion_paths(0.0, 
                                                              h0_param,
                                                              path_bank_size,
                                                              grid_points,
                                                              T)), 0, 1).to(device=device)

                    h1_paths = torch.transpose(torch.from_numpy(
                        generate_scaled_brownian_motion_paths(0.0, 
                                                              h1_param,
                                                              path_bank_size,
                                                              grid_points,
                                                              T)), 0, 1).to(device=device)

                    h0_dists, h1_dists, _ = return_mmd_distributions(
                        h0_paths, 
                        h1_paths, 
                        signature_kernel.compute_mmd, 
                        n_atoms=n_atoms, 
                        batch_size=n_paths, 
                        estimator=estimator,
                        u_stat=False, 
                        verbose=False
                    )

                    crit_val = np.sort(np.asarray(h0_dists))[int(len(h0_dists) * (1 - alpha))]
                    crit_vals_sim.append(crit_val)
                    errors_list.append(expected_type2_error(torch.tensor(h1_dists), crit_val))
                
                type2_errors_sim.append((np.mean(errors_list), np.std(errors_list)))

                if estimator == 'ub':
                    mean_h1 = Exp_MMD_3_ub(h0_param, h1_param, T=T)
                else:
                    mean_h1 = Exp_MMD_2_b(h0_param, h1_param, n_paths, T=T)
                var_h1 = trunc_vol_H_A_dist(h0_param, h1_param, T)
                scale_h1 = np.sqrt(var_h1 * 1/n_paths)
                loc_h1 = 1.0*mean_h1 * 1

                if scale_h1 == 0:
                    type2_errors_analytical.append(type2_errors_analytical[-1])
                else:
                    type2_errors_analytical.append(norm.cdf(threshold_est/n_paths, loc=loc_h1, scale=scale_h1))

                if estimator == 'ub':
                    type2_errors_gaussian.append(-1)
                else:
                    type2_errors_gaussian.append(norm.cdf(threshold_est/n_paths, loc=loc_h1,
                                                          scale=np.sqrt(Var_MMD_2_b(h0_param, h1_param, n_paths, T=T))))
                
            error_dict[f'{param1}'] = (type2_errors_sim, type2_errors_analytical, type2_errors_gaussian)


    return error_dict
        

            


    

