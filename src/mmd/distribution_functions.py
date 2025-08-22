import torch
import pickle
from tqdm import tqdm
import numpy as np
from src.mmd.level_functions import level_k_contribution, get_level_k_signatures_from_paths
from scipy.special import gamma


def return_mmd_distributions(h0_paths, h1_paths, mmd, n_atoms=128, batch_size=32, verbose=True, estimator='ub', u_stat=False, rand_ints=None):
    """
    Returns null and alternate MMD distributions for a given kernel

    :param h0_paths:    Bank of paths under the null hypothesis
    :param h1_paths:    Bank of paths under the alternate hypothesis
    :param mmd:         Method to calculate the MMD
    :param n_atoms:     Number of atoms in the corresponding distributions
    :param batch_size:  Number of paths to sample from each distribution
    :param verbose:     Whether to plot progress bar
    :return: Test statistic under H0 and test statistic under H1
    """

    assert h0_paths.shape[0] == h1_paths.shape[0]

    path_bank_size = h0_paths.shape[0]

    h0_dists = torch.zeros(n_atoms)
    h1_dists = torch.zeros(n_atoms)

    if rand_ints is None:
        rand_ints = np.zeros((n_atoms, 3, batch_size))
        for i in range(n_atoms):
            for j in range(3):
                rand_ints[i, j, :] = np.random.choice(path_bank_size, size=(batch_size), replace=False)
        rand_ints = torch.tensor(rand_ints).long()

    itr = tqdm(rand_ints) if verbose else rand_ints

    with torch.no_grad():
        for i, ii in enumerate(itr):
            x1, x2 = h0_paths[ii[0]], h0_paths[ii[1]]
            y = h1_paths[ii[-1]]

            h0_dists[i] = mmd(x1, x2, estimator=estimator, u_stat=u_stat)
            h1_dists[i] = mmd(x1, y, estimator=estimator, u_stat=u_stat)

    return h0_dists.tolist(), h1_dists.tolist(), rand_ints


def expected_type2_error(h1_dist: torch.Tensor, crit_value: float):
    """
    Calculates the expected type II error given a critical value from a null distribution and the alternate distribution

    :param h1_dist:     MMD distribution under the alternate hypothesis
    :param crit_value:  Critical value associated to the null distribution
    :return: P[Type 2 Error]
    """
    n_atoms = h1_dist.shape[0]
    num_fail = h1_dist <= crit_value
    return sum(num_fail.type(torch.float32))/n_atoms


def get_level_values(h0_paths, h1_paths, n_atoms, n_paths, ks, path_bank_size, verbose=True, exp=False, unbiased=True,
                     u_stat=False):

    """
    Compute level contributions (Gamma_{k})
    :param h0_paths: Bank of paths under the null hypothesis
    :param h1_paths: Bank of paths under the alternate hypothesis
    :param n_atoms: Number of empirical simulations to compute probability of errors
    :param n_paths: Batch size
    :param ks: List of levels
    :param path_bank_size: Number of available samples
    :param verbose: Flag indicating whether to display progress. Default is True
    :param exp: Flag indicating whether the randomised signature kernel with Exp(1) distribution is used. Default is False
    :param unbiased: Flag indicating whether to use unbiased estimator. Default is True
    :param u_stat: Flag indicating whether to use unbiased U-statistic estimator. Default is False
    :return: 2 arrays. The first contains the level contributions under the null and the second contains the level
             contributions under the altnerate.
    """

    h0_Mk_vals = torch.zeros((len(ks), n_atoms))
    h1_Mk_vals = torch.zeros((len(ks), n_atoms))

    rand_ints = torch.randint(0, path_bank_size, size=(n_atoms, 3, n_paths))

    if verbose:
        itr = tqdm(rand_ints)
    else:
        itr = rand_ints

    for i, ii in enumerate(itr):
        x1, x2 = h0_paths[ii[0]], h0_paths[ii[1]]
        y = h1_paths[ii[-1]]

        for j, _k in enumerate(ks):

            if exp:
                x1 = torch.multiply(torch.Tensor([gamma(_k / 2), 1]).to(device=h0_paths.device), x1)
                x2 = torch.multiply(torch.Tensor([gamma(_k / 2), 1]).to(device=h0_paths.device), x2)
                y = torch.multiply(torch.Tensor([gamma(_k / 2), 1]).to(device=h0_paths.device), y)

            h0_Mk_vals[j, i] = level_k_contribution(x1, x2, _k, unbiased=unbiased, u_stat=u_stat)
            h1_Mk_vals[j, i] = level_k_contribution(x1, y, _k, unbiased=unbiased, u_stat=u_stat)

    return h0_Mk_vals, h1_Mk_vals


def get_type1_type2_errors(signature_kernel, h0_paths, h1_paths, scaling, n_atoms, n_paths, estimator, alpha, device):

    """
    Compute the probability of a Type 1 error and a Type 2 error occurring
    :param signature_kernel: The signature kernel object
    :param h0_paths: Bank of paths under the null hypothesis
    :param h1_paths: Bank of paths under the alternate hypothesis
    :param scaling: Current scaling value
    :param n_atoms: Number of empirical simulations to compute probability of errors
    :param n_paths: Batch size
    :param estimator: String describing the estimator. Either 'ub' for unbiased or 'b' for biased
    :param alpha: Level of the test
    :param device: Device. Either 'cuda' or 'cpu'
    :return: P[Type 2 error] and P[Type 1 error]
    """

    h0_dists, h1_dists = return_mmd_distributions(
        torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h0_paths[:, :, :]),
        torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h1_paths[:, :, :]),
        signature_kernel.compute_mmd,
        n_atoms=n_atoms,
        batch_size=n_paths,
        estimator=estimator,
        verbose=False
    )

    h00_dists, h01_dists = return_mmd_distributions(
        torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h0_paths[:, :, :]),
        torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h0_paths[:, :, :]),
        signature_kernel.compute_mmd,
        n_atoms=n_atoms,
        batch_size=n_paths,
        estimator=estimator,
        verbose=False
    )

    crit_val = np.sort(np.asarray(h0_dists))[int(n_atoms * (1 - alpha))]
    type_2_error = 100 * expected_type2_error(torch.tensor(h1_dists), crit_val)

    crit_val2 = np.sort(np.asarray(h00_dists))[int(n_atoms * (1 - alpha))]
    type_1_error = 100 * expected_type2_error(torch.tensor(h01_dists), crit_val2)

    return type_2_error, type_1_error

def generate_error_probs_linear_kernel(signature_kernel, h0_paths, h1_paths, n_atoms, n_paths_list, alpha, scalings,
                                       estimator, num_sim, device, filename=None, folder=''):
    """
    Generate list of empirical probabilities of a Type 1 error and a Type 2 error occurring
    :param signature_kernel: The signature kernel object
    :param h0_paths: Bank of paths under the null hypothesis
    :param h1_paths: Bank of paths under the alternate hypothesis
    :param n_atoms: Number of empirical simulations to compute probability of errors
    :param n_paths_list: List containing batch sizes
    :param alpha: Level of the test
    :param scalings: List of scalings
    :param estimator: String describing the estimator. Either 'ub' for unbiased or 'b' for biased
    :param num_sim: Number of simulations to generate confidence intervals
    :param device: Device. Either 'cuda' or 'cpu'
    :param filename: Filename to save the lists. Default is None and in this case the lists are not saved
    :param folder: Folder in which to save the lists
    :return: 2 lists. The first contains the probability of a Type 1 error occurring and the second contains the
             probability of a Type 2 error occurring
    """

    assert estimator == 'b' or estimator == 'ub', "the estimator should be 'b' or 'ub' "

    type2_list = []
    type1_list = []
    for _ in tqdm(range(num_sim)):

        type_2_errors_dict = {}

        type_1_errors_dict = {}

        for n_paths in n_paths_list:

            type_2_errors = []

            type_1_errors = []

            for scaling in scalings:

                type2_error, type1_error = get_type1_type2_errors(signature_kernel, h0_paths, h1_paths, scaling,
                                                                  n_atoms, n_paths, estimator, alpha, device)
                type_2_errors.append(type2_error)
                type_1_errors.append(type1_error)

            type_2_errors_dict[n_paths] = type_2_errors
            type_1_errors_dict[n_paths] = type_1_errors
        type2_list.append(type_2_errors_dict)
        type1_list.append(type_1_errors_dict)

    if filename is not None:
        with open(f"{folder}type1error_{filename}", "wb") as fp:
            pickle.dump(type1_list, fp)

        with open(f"{folder}type2error_{filename}", "wb") as fp:
            pickle.dump(type2_list, fp)

    return type1_list, type2_list


def generate_error_probs_rbf_kernel(h0_paths, h1_paths, sigma, n_atoms, n_paths_list, alpha, scalings,
                                    estimator, num_sim, device, filename=None, folder=''):
    """
    Generate list of empirical probabilities of a Type 1 error and a Type 2 error occurring with RBF kernel applied.
    :param h0_paths: Bank of paths under the null hypothesis
    :param h1_paths: Bank of paths under the alternate hypothesis
    :param sigma: RBF kernel smoothing parameter
    :param n_atoms: Number of empirical simulations to compute probability of errors
    :param n_paths_list: List containing batch sizes
    :param alpha: Level of the test
    :param scalings: List of scalings
    :param estimator: String describing the estimator. Either 'ub' for unbiased or 'b' for biased
    :param num_sim: Number of simulations to generate confidence intervals
    :param device: Device. Either 'cuda' or 'cpu'
    :param filename: Filename to save the lists. Default is None and in this case the lists are not saved
    :param folder: Folder in which to save the lists
    :return: 2 lists. The first contains the probability of a Type 1 error occurring and the second contains the
             probability of a Type 2 error occurring
    """

    assert estimator == 'b' or estimator == 'ub', "the estimator should be 'b' or 'ub' "

    type2_list = []
    type1_list = []
    dyadic_order = 0

    for _ in tqdm(range(num_sim)):

        type_2_errors_dict = {}

        type_1_errors_dict = {}

        for n_paths in tqdm(n_paths_list):

            type_2_errors = []

            type_1_errors = []

            for scaling in scalings:

                static_kernel = RBFKernel(sigma=sigma, scaling=scaling)
                rbf_signature_kernel = SigKernel(static_kernel=static_kernel, dyadic_order=dyadic_order)

                type2_error, type1_error = get_type1_type2_errors(rbf_signature_kernel, h0_paths, h1_paths, 1,
                                                                  n_atoms, n_paths, estimator, alpha, device)
                type_2_errors.append(type2_error)
                type_1_errors.append(type1_error)

            type_2_errors_dict[n_paths] = type_2_errors
            type_1_errors_dict[n_paths] = type_1_errors
        type2_list.append(type_2_errors_dict)
        type1_list.append(type_1_errors_dict)

    if filename is not None:
        with open(f"{folder}type1error_{filename}", "wb") as fp:
            pickle.dump(type1_list, fp)

        with open(f"{folder}type2error_{filename}", "wb") as fp:
            pickle.dump(type2_list, fp)

    return type1_list, type2_list
