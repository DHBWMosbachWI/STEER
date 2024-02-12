from collections import Counter
from scipy.interpolate import interp1d
import numpy as np


def cummulative_distribution_function(values: np.array):
    """
    Function that calculates the CDF for values in an numpy array

    Parameters
    ----------
    values: np.array
        the array that contains the values

    Returns
    -------
    A tuple of the form (x, cdf), where x represents the x-axes (values) and cdf the y-axes (probabilities)
    """
    counters = Counter(values)
    counts_sorted = sorted(counters.items(),
                           key=lambda pair: pair[0],
                           reverse=False)
    x = np.array(list(zip(*counts_sorted))[0])
    y = np.array(list(zip(*counts_sorted))[1])

    # Calc PDF (Probability Distribution Function)
    pdf = y / sum(y)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    return (x, cdf)


def inverse_transform_sampling(values: np.array, number_of_samples: int):
    """
    Function that returns for a given number of samples the inverse transform sampling
    using the inverse CDF Function

    Parameter
    ---------
    values: np.array
        the array that contains the values
    number_of_samples: int:
        the number of samples which the function should return using the inverse CDF
        Example: For 10, the function returns values for the probabilities [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    Returns
    -------
    A tuple of the form (x, icdf), where x represents the x-axes (probabilities) and icdf the y-axes (values to the probs)
    """
    x, cdf = cummulative_distribution_function(values)
    # generate the inverse cummulative distribution function
    i_cdf_f = interp1d(cdf,
                       x,
                       bounds_error=False,
                       fill_value=(x[0], x[len(x) - 1]))
    return (np.arange(0 + 1 / number_of_samples, 1 + 1 / number_of_samples,
                      1 / number_of_samples),
            i_cdf_f(
                np.arange(0 + 1 / number_of_samples, 1 + 1 / number_of_samples,
                          1 / number_of_samples)))
