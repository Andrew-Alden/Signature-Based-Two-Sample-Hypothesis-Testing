from scipy import stats

def cdf(x: float, alpha, beta, N):
    """
    Calculates cdf of corresponding gamma distribution at a given point x

    :param x:       Point of evaluation
    :return:        Value of cdf at x
    """
    return stats.gamma.cdf(x*1, a=alpha, scale=beta)

def ppf(q: float, alpha, beta, N):
    """
    Calculates quantile function of corresponding distribution for a given level q

    :param q:   Quantile to calculate inverse cdf of
    :return:    Value of quantile function at q
    """
    return stats.gamma.ppf(q, a=alpha, scale=beta)

def pdf(x, alpha, beta, N, loc=0):
    return stats.gamma.pdf(x, a=alpha, scale=beta)


def gaussian_pdf(x, loc, scale):
    return stats.norm.pdf(x, loc=loc, scale=scale)

scale = lambda mean, var, N: mean**2/var
rate  = lambda mean, var, N: (N*var)/mean