"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2015 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes
import thinkplot


"""
This problem presents a solution to the "Bayesian Billiards Problem",
presented in this video:

https://www.youtube.com/watch?v=KhAUfqhLakw

Based on the formulation in this paper:

http://www.nature.com/nbt/journal/v22/n9/full/nbt0904-1177.html

Of a problem originally posed by Bayes himself.
"""

class Billiards(thinkbayes.Suite):

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        data: tuple (#wins, #losses)
        hypo: float probability of win
        """
        p = hypo
        win, lose = data
        like = p**win * (1-p)**lose
        return like


def ProbWinMatch(pmf):
    total = 0
    for p, prob in pmf.Items():
        total += prob * (1-p)**3
    return total


def main():
    ps = numpy.linspace(0, 1, 101)
    bill = Billiards(ps)
    bill.Update((5, 3))
    thinkplot.Pdf(bill)
    thinkplot.Save(root='billiards1',
                   xlabel='probability of win',
                   ylabel='PDF',
                   formats=['png'])

    bayes_result = ProbWinMatch(bill)
    print(thinkbayes.Odds(1-bayes_result))

    mle = 5 / 8
    freq_result = (1-mle)**3
    print(thinkbayes.Odds(1-freq_result))

    
if __name__ == '__main__':
    main()
