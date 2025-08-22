import numpy as np

def generate_scaled_brownian_motion_paths(S0, vol, num_sim, num_time_steps, T, seed=None):

    """
    Generate scaled Brownian motion sample paths.
    :param S0: Initial value.
    :param vol: Volatility parameter.
    :param num_sim: Number of simulations.
    :param num_time_steps: Number of time steps excluding initial value.
    :param T: Terminal time.
    :param seed: Numpy random seed. Default is None.
    :return: Array containing time-augmented paths.
    """
    
    np.random.seed(seed)
    h = np.divide(T, num_time_steps)
    normal_rvs = np.multiply(np.sqrt(h), np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim)))
    S = np.ones((num_time_steps + 1, num_sim))
    S[0, :] = np.ones(num_sim) * S0
    
    time_steps = [0]
    
    for i in range(1, num_time_steps + 1):
        S[i, :] = S[i-1, :] + np.multiply(vol, normal_rvs[i-1])
        time_steps.append(i * h)
        
    return np.concatenate((S[:, :, None],
                           np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                     axis=1)),
                          axis=2)