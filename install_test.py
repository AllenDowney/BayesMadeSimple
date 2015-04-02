"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import math
import numpy

from matplotlib import pyplot

import thinkplot
import thinkbayes


def RenderPdf(mu, sigma, n=101):
    """Makes xs and ys for a normal PDF with (mu, sigma).

    n: number of places to evaluate the PDF
    """
    xs = numpy.linspace(mu-4*sigma, mu+4*sigma, n)
    ys = [thinkbayes.EvalNormalPdf(x, mu, sigma) for x in xs]
    return xs, ys


def main():
    xs, ys = RenderPdf(100, 15)

    n = 34
    pyplot.fill_between(xs[-n:], ys[-n:], y2=0.0001, color='blue', alpha=0.2)
    s = 'Congratulations!\nIf you got this far,\nyou must be here.'
    d = dict(shrink=0.05)
    pyplot.annotate(s, [127, 0.002], xytext=[80, 0.005], arrowprops=d)

    thinkplot.Plot(xs, ys)
    thinkplot.Show(title='Distribution of IQ',
                   xlabel='IQ',
                   ylabel='PDF',
                   legend=False)


if __name__ == "__main__":
    main()
