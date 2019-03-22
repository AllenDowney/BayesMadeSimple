"""

Pmf: Represents a probability Mass Function (PMF).

Copyright 2019 Allen B. Downey

MIT License: https://opensource.org/licenses/MIT
"""

import numpy as np
import pandas as pd

import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d

    returns: modified d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


class Pmf(pd.Series):
    """Represents a probability Mass Function (PMF)."""
    
    def __init__(self, *args, **kwargs):
        """Initialize a Series.

        Note: this cleans up a weird Series behavior, which is
        that Series() and Series([]) yield different results.
        See: https://github.com/pandas-dev/pandas/issues/16737
        """
        if args:
            super().__init__(*args, **kwargs)
        else:
            underride(kwargs, dtype=np.float64)
            super().__init__([], **kwargs)

    @property
    def qs(self):
        """Get the quantities.

        returns: NumPy array
        """
        return self.index.values

    @property
    def ps(self):
        """Get the probabilities.

        returns: NumPy array
        """
        return self.values

    def _repr_html_(self):
        """Returns an HTML representation of the series.

        Mostly used for Jupyter notebooks.
        """
        df = pd.DataFrame(dict(probs=self))
        return df._repr_html_()

    def normalize(self):
        """Make the probabilities add up to 1 (modifies self).

        returns: normlizing constant
        """
        total = self.sum()
        self /= total
        return total

    def mean(self):
        """Computes expected value.

        returns: float
        """
        return np.sum(self.ps * self.qs)

    def choice(self, **options):
        """Makes a random choice.

        options: same as np.random.choice

        returns: NumPy array
        """
        options['p'] = self.ps
        return np.random.choice(self.qs, **options)

    def bar(self, **options):
        """Makes a bar plot.

        options: same as plt.bar
        """
        underride(options, label=self.name)
        plt.bar(self.qs, self.ps, **options)

    def __add__(self, x):
        """Computes the Pmf of the sum of values drawn from self and x.

        x: another Pmf or a scalar

        returns: new Pmf
        """
        if isinstance(x, Pmf):
            return pmf_add(self, x)
        else:
            return Pmf(self.qs + x, self.ps)

    __radd__ = __add__

    def update(self, likelihood, data):
        """Bayesian update.

        likelihood: function that takes (data, hypo) and returns
                    likelihood of data under hypo
        data: whatever format like_func understands

        returns: normalizing constant
        """
        for hypo in self.qs:
            self[hypo] *= likelihood(data, hypo)

        return self.normalize()

    def MAP(self):
        """Maximum aposteori probability.

        returns: the value with the highest probability
        """
        return self.idxmax()

    def quantile(self, ps):
        """Quantities corresponding to given probabilities.

        ps: sequence of probabilities

        return: sequence of quantities
        """
        cdf = self.sort_index().cumsum()
        interp = interp1d(cdf.values, cdf.index,
                          kind='next',
                          copy=False,
                          assume_sorted=True,
                          bounds_error=False,
                          fill_value=(self.qs[0], np.nan))
        return interp(ps)

    def credible_interval(self, p):
        """Credible interval containing the given probability.

        p: float 0-1

        returns: array of two quantities
        """
        tail = (1-p) / 2
        ps = [tail, 1-tail]
        return self.quantile(ps)


def pmf_add(pmf1, pmf2):
    """Distribution of the sum.

    pmf1: Pmf
    pmf2: Pmf

    returns: new Pmf
    """
    qs = np.add.outer(pmf1.qs, pmf2.qs).flatten()
    ps = np.multiply.outer(pmf1.ps, pmf2.ps).flatten()
    series = pd.Series(ps).groupby(qs).sum()
    return Pmf(series)


def pmf_from_seq(seq, normalize=True, sort=True, **options):
    """Make a PMF from a sequence of values.

    seq: any kind of sequence
    normalize: whether to normalize the Pmf, default True
    sort: whether to sort the Pmf by values, default True
    options: passed to the pd.Series constructor

    returns: Pmf object
    """
    series = pd.Series(seq).value_counts(sort=False)

    options['copy'] = False
    pmf = Pmf(series, **options)

    if sort:
        pmf.sort_index(inplace=True)

    if normalize:
        pmf.normalize()

    return pmf
