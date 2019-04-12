"""

Pmf: Represents a Probability Mass Function (PMF).
Cdf: Represents a Cumulative Distribution Function (CDF).

Copyright 2019 Allen B. Downey

MIT License: https://opensource.org/licenses/MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d

    :return: modified d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


class Pmf(pd.Series):
    """Represents a probability Mass Function (PMF)."""

    def __init__(self, *args, **kwargs):
        """Initialize a Pmf.

        Note: this cleans up a weird Series behavior, which is
        that Series() and Series([]) yield different results.
        See: https://github.com/pandas-dev/pandas/issues/16737
        """
        if args:
            super().__init__(*args, **kwargs)
        else:
            underride(kwargs, dtype=np.float64)
            super().__init__([], **kwargs)

    def copy(self, deep=True):
        """Make a copy.

        :return: new Pmf
        """
        return Pmf(self, copy=deep)

    def __getitem__(self, qs):
        """Look up qs and return ps."""
        try:
            return super().__getitem__(qs)
        except (KeyError, ValueError, IndexError):
            return 0

    @property
    def qs(self):
        """Get the quantities.

        :return: NumPy array
        """
        return self.index.values

    @property
    def ps(self):
        """Get the probabilities.

        :return: NumPy array
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

        :return: normalizing constant
        """
        total = self.sum()
        self /= total
        return total

    def mean(self):
        """Computes expected value.

        :return: float
        """
        #TODO: error if not normalized
        #TODO: error if the quantities are not numeric
        return np.sum(self.ps * self.qs)

    def median(self):
        """Median (50th percentile).

        :return: float
        """
        return self.quantile(0.5)

    def quantile(self, ps, **kwargs):
        """Quantiles.

        Computes the inverse CDF of ps, that is,
        the values that correspond to the given probabilities.

        :return: float
        """
        return self.make_cdf().quantile(ps, **kwargs)

    def var(self):
        """Variance of a PMF.

        :return: float
        """
        m = self.mean()
        d = self.qs - m
        return np.sum(d**2 * self.ps)

    def std(self):
        """Standard deviation of a PMF.

        :return: float
        """
        return np.sqrt(self.var())

    def choice(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `p` is provided.

        args: same as np.random.choice
        kwargs: same as np.random.choice

        :return: NumPy array
        """
        underride(kwargs, p=self.ps)
        return np.random.choice(self.qs, *args, **kwargs)

    def sample(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `weights` is provided.

        This function returns an array containing a sample of the quantities in this Pmf,
        which is different from Series.sample, which returns a Series with a sample of
        the rows in the original Series.

        args: same as Series.sample
        options: same as Series.sample

        :return: NumPy array
        """
        series = pd.Series(self.qs)
        underride(kwargs, weights=self.ps)
        sample = series.sample(*args, **kwargs)
        return sample.values

    def plot(self, **options):
        """Plot the Pmf as a line.

        :param options: passed to plt.plot
        :return:
        """
        underride(options, label=self.name)
        plt.plot(self.qs, self.ps, **options)

    def bar(self, **options):
        """Makes a bar plot.

        options: passed to plt.bar
        """
        underride(options, label=self.name)
        plt.bar(self.qs, self.ps, **options)

    def add(self, x):
        """Computes the Pmf of the sum of values drawn from self and x.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Pmf):
            return pmf_conv(self, x, np.add.outer)
        else:
            return Pmf(self.ps, index=self.qs + x)

    __add__ = add
    __radd__ = add

    def sub(self, x):
        """Computes the Pmf of the diff of values drawn from self and other.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Pmf):
            return pmf_conv(self, x, np.subtract.outer)
        else:
            return Pmf(self.ps, index=self.qs - x)

    subtract = sub
    __sub__ = sub

    def rsub(self, x):
        """Computes the Pmf of the diff of values drawn from self and other.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Pmf):
            return pmf_conv(x, self, np.subtract.outer)
        else:
            return Pmf(self.ps, index=x - self.qs)

    __rsub__ = rsub

    def mul(self, x):
        """Computes the Pmf of the product of values drawn from self and x.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Pmf):
            return pmf_conv(self, x, np.multiply.outer)
        else:
            return Pmf(self.ps, index=self.qs * x)

    multiply = mul
    __mul__ = mul
    __rmul__ = mul

    def div(self, x):
        """Computes the Pmf of the ratio of values drawn from self and x.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Pmf):
            return pmf_conv(self, x, np.divide.outer)
        else:
            return Pmf(self.ps, index=self.qs / x)

    divide = div
    __div = div
    __truediv__ = div

    def rdiv(self, x):
        """Computes the Pmf of the ratio of values drawn from self and x.

        x: another Pmf or a scalar or a sequence

        :return: new Pmf
        """
        if isinstance(x, Pmf):
            return pmf_conv(x, self, np.divide.outer)
        else:
            return Pmf(self.ps, index=x / self.qs)

    __rdiv__ = rdiv
    __rtruediv__ = rdiv

    def make_joint(self, other, **options):
        """Make joint distribution (assuming independence).

        :param self:
        :param other:
        :param options: passed to Pmf constructor

        :return: new Pmf
        """
        qs = pd.MultiIndex.from_product([self.qs, other.qs])
        ps = np.multiply.outer(self.ps, other.ps).flatten()
        return Pmf(ps, index=qs, **options)

    def marginal(self, i, name=None):
        """Gets the marginal distribution of the indicated variable.

        i: index of the variable we want
        name: string

        :return: Pmf
        """
        # TODO: rewrite this using MultiIndex operations
        pmf = Pmf(name=name)
        for vs, p in self.items():
            pmf[vs[i]] += p
        return pmf

    def conditional(self, i, j, val, name=None):
        """Gets the conditional distribution of the indicated variable.

        Distribution of vs[i], conditioned on vs[j] = val.

        i: index of the variable we want
        j: which variable is conditioned on
        val: the value the jth variable has to have
        name: string

        :return: Pmf
        """
        # TODO: rewrite this using MultiIndex operations
        pmf = Pmf(name=name)
        for vs, p in self.items():
            if vs[j] == val:
                pmf[vs[i]] += p

        pmf.normalize()
        return pmf

    def update(self, likelihood, data):
        """Bayesian update.

        likelihood: function that takes (data, hypo) and returns
                    likelihood of data under hypo
        data: whatever format like_func understands

        :return: normalizing constant
        """
        for hypo in self.qs:
            self[hypo] *= likelihood(data, hypo)

        return self.normalize()

    def max_prob(self):
        """Value with the highest probability.

        :return: the value with the highest probability
        """
        return self.idxmax()

    def make_cdf(self, normalize=True):
        """Make a Cdf from the Pmf.

        It can be good to normalize the cdf even if the Pmf was normalized,
        to guarantee that the last element of `ps` is 1.

        :return: Cdf
        """
        cdf = Cdf(self.cumsum())
        if normalize:
            cdf.normalize()
        return cdf

    def quantile(self, ps):
        """Quantities corresponding to given probabilities.

        ps: sequence of probabilities

        return: sequence of quantities
        """
        cdf = self.make_cdf()
        return cdf.quantile(ps)

    def credible_interval(self, p):
        """Credible interval containing the given probability.

        p: float 0-1

        :return: array of two quantities
        """
        tail = (1-p) / 2
        ps = [tail, 1-tail]
        return self.quantile(ps)

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, **options):
        """Make a PMF from a sequence of values.

        seq: any kind of sequence
        normalize: whether to normalize the Pmf, default True
        sort: whether to sort the Pmf by values, default True
        options: passed to the pd.Series constructor

        :return: Pmf object
        """
        series = pd.Series(seq).value_counts(sort=False)

        options['copy'] = False
        pmf = Pmf(series, **options)

        if sort:
            pmf.sort_index(inplace=True)

        if normalize:
            pmf.normalize()

        return pmf

    # Comparison operators

    def gt(self, x):
        """Probability that a sample from this Pmf > x.

        x: number

        :return: float probability
        """
        if isinstance(x, Pmf):
            return pmf_gt(self, x)
        else:
            return self[self.qs > x].sum()

    __gt__ = gt

    def lt(self, x):
        """Probability that a sample from this Pmf < x.

        x: number

        :return: float probability
        """
        if isinstance(x, Pmf):
            return pmf_lt(self, x)
        else:
            return self[self.qs < x].sum()

    __lt__ = lt

    def ge(self, x):
        """Probability that a sample from this Pmf >= x.

        x: number

        :return: float probability
        """
        if isinstance(x, Pmf):
            return pmf_ge(self, x)
        else:
            return self[self.qs >= x].sum()

    __ge__ = ge

    def le(self, x):
        """Probability that a sample from this Pmf <= x.

        x: number

        :return: float probability
        """
        if isinstance(x, Pmf):
            return pmf_le(self, x)
        else:
            return self[self.qs <= x].sum()

    __le__ = le

    def eq(self, x):
        """Probability that a sample from this Pmf == x.

        x: number

        :return: float probability
        """
        if isinstance(x, Pmf):
            return pmf_eq(self, x)
        else:
            return self[self.qs == x].sum()

    __eq__ = eq

    def ne(self, x):
        """Probability that a sample from this Pmf != x.

        x: number

        :return: float probability
        """
        if isinstance(x, Pmf):
            return pmf_ne(self, x)
        else:
            return self[self.qs != x].sum()

    __ne__ = ne


def pmf_conv(pmf1, pmf2, ufunc):
    """Convolve two PMFs.

    pmf1:
    pmf2:
    ufunc: elementwise function for arrays

    :return: new Pmf
    """
    qs = ufunc(pmf1.qs, pmf2.qs).flatten()
    ps = np.multiply.outer(pmf1.ps, pmf2.ps).flatten()
    series = pd.Series(ps).groupby(qs).sum()
    return Pmf(series)


def pmf_add(pmf1, pmf2):
    """Distribution of the sum.

    pmf1:
    pmf2:

    :return: new Pmf
    """
    return pmf_conv(pmf1, pmf2, np.add.outer)


def pmf_sub(pmf1, pmf2):
    """Distribution of the difference.

    pmf1:
    pmf2:

    :return: new Pmf
    """
    return pmf_conv(pmf1, pmf2, np.subtract.outer)


def pmf_mul(pmf1, pmf2):
    """Distribution of the product.

    pmf1:
    pmf2:

    :return: new Pmf
    """
    return pmf_conv(pmf1, pmf2, np.multiply.outer)

def pmf_div(pmf1, pmf2):
    """Distribution of the ratio.

    pmf1:
    pmf2:

    :return: new Pmf
    """
    return pmf_conv(pmf1, pmf2, np.divide.outer)

def pmf_outer(pmf1, pmf2, ufunc):
    """Computes the outer product of two PMFs.

    pmf1:
    pmf2:
    ufunc: function to apply to the qs

    :return: NumPy array
    """
    qs = ufunc.outer(pmf1.qs, pmf2.qs)
    ps = np.multiply.outer(pmf1.ps, pmf2.ps)
    return qs * ps


def pmf_gt(pmf1, pmf2):
    """Probability that a value from pmf1 is greater than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    :return: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.greater)
    return outer.sum()


def pmf_lt(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    :return: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.less)
    return outer.sum()


def pmf_ge(pmf1, pmf2):
    """Probability that a value from pmf1 is >= than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    :return: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.greater_equal)
    return outer.sum()


def pmf_le(pmf1, pmf2):
    """Probability that a value from pmf1 is <= than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    :return: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.less_equal)
    return outer.sum()


def pmf_eq(pmf1, pmf2):
    """Probability that a value from pmf1 equals a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    :return: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.equal)
    return outer.sum()


def pmf_ne(pmf1, pmf2):
    """Probability that a value from pmf1 is <= than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    :return: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.not_equal)
    return outer.sum()


class Cdf(pd.Series):
    """Represents a Cumulative Distribution Function (CDF)."""

    def __init__(self, *args, **kwargs):
        """Initialize a Cdf.

        Note: this cleans up a weird Series behavior, which is
        that Series() and Series([]) yield different results.
        See: https://github.com/pandas-dev/pandas/issues/16737
        """
        if args:
            super().__init__(*args, **kwargs)
        else:
            underride(kwargs, dtype=np.float64)
            super().__init__([], **kwargs)

    def copy(self, deep=True):
        """Make a copy.

        :return: new Pmf
        """
        return Cdf(self, copy=deep)

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, **options):
        """Make a CDF from a sequence of values.

        seq: any kind of sequence
        normalize: whether to normalize the Cdf, default True
        sort: whether to sort the Cdf by values, default True
        options: passed to the pd.Series constructor

        :return: CDF object
        """
        pmf = Pmf.from_seq(seq, normalize=False, sort=sort, **options)
        return pmf.make_cdf(normalize=normalize)

    @property
    def qs(self):
        """Get the quantities.

        :return: NumPy array
        """
        return self.index.values

    @property
    def ps(self):
        """Get the probabilities.

        :return: NumPy array
        """
        return self.values

    def _repr_html_(self):
        """Returns an HTML representation of the series.

        Mostly used for Jupyter notebooks.
        """
        df = pd.DataFrame(dict(probs=self))
        return df._repr_html_()

    def plot(self, **options):
        """Plot the Cdf as a line.

        :param options: passed to plt.plot
        :return:
        """
        underride(options, label=self.name)
        plt.plot(self.qs, self.ps, **options)

    def step(self, **options):
        """Plot the Cdf as a step function.

        :param options: passed to plt.step
        :return:
        """
        underride(options, label=self.name, where='post')
        plt.step(self.qs, self.ps, **options)

    def normalize(self):
        """Make the probabilities add up to 1 (modifies self).

        :return: normalizing constant
        """
        total = self.ps[-1]
        self /= total
        return total

    @property
    def forward(self, **kwargs):
        """Compute the forward Cdf

        :param kwargs: keyword arguments passed to interp1d

        :return array of probabilities
        """

        underride(kwargs, kind='previous',
                  copy=False,
                  assume_sorted=True,
                  bounds_error=False,
                  fill_value=(0, 1))

        interp = interp1d(self.qs, self.ps, **kwargs)
        return interp

    @property
    def inverse(self, **kwargs):
        """Compute the inverse Cdf

        :param kwargs: keyword arguments passed to interp1d

        :return array of quantities
        """
        underride(kwargs, kind='next',
                  copy=False,
                  assume_sorted=True,
                  bounds_error=False,
                  fill_value=(self.qs[0], np.nan))

        interp = interp1d(self.ps, self.qs, **kwargs)
        return interp

    # calling a Cdf like a function does forward lookup
    __call__ = forward

    # quantile is the same as an inverse lookup
    quantile = inverse

    def make_pmf(self, normalize=False):
        """Make a Pmf from the Cdf.

        :return: Cdf
        """
        ps = self.ps
        diff = np.ediff1d(ps, to_begin=ps[0])
        pmf = Pmf(pd.Series(diff, index=self.index.copy()))
        if normalize:
            pmf.normalize()
        return pmf

    def choice(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `p` is provided.

        args: same as np.random.choice
        options: same as np.random.choice

        :return: NumPy array
        """
        # TODO: Make this more efficient by implementing the inverse CDF method.
        pmf = self.make_pmf()
        return pmf.choice(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Makes a random sample.

        Uses the probabilities as weights unless `weights` is provided.

        This function returns an array containing a sample of the quantities in this Pmf,
        which is different from Series.sample, which returns a Series with a sample of
        the rows in the original Series.

        args: same as Series.sample
        options: same as Series.sample

        :return: NumPy array
        """
        # TODO: Make this more efficient by implementing the inverse CDF method.
        pmf = self.make_pmf()
        return pmf.sample(*args, **kwargs)

    def mean(self):
        """Expected value.

        :return: float
        """
        return self.make_pmf().mean()

    def var(self):
        """Variance.

        :return: float
        """
        return self.make_pmf().var()

    def std(self):
        """Standard deviation.

        :return: float
        """
        return self.make_pmf().std()

    def median(self):
        """Median (50th percentile).

        :return: float
        """
        return self.quantile(0.5)


