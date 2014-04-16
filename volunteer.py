"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import thinkbayes
import thinkplot

import numpy

class Volunteer(thinkbayes.Suite, thinkbayes.Joint):

    def Likelihood(self, data, hypo):
        if len(data) == 2:
            return self.Likelihood1(data, hypo)
        elif len(data) == 3:
            return self.Likelihood2(data, hypo)
        else:
            raise ValueError()

    def Likelihood1(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: 
        data: 
        """
        q, r = hypo
        p = q * r
        signed_up, reported = data
        yes = reported
        no = signed_up - reported

        like = p**yes * (1-p)**no
        return like

    def Likelihood2(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: 
        data: 
        """
        q, r = hypo

        signed_up, participated, reported = data

        yes = participated
        no = signed_up - participated
        like1 = q**yes * (1-q)**no

        yes = reported
        no = participated - reported
        like2 = r**yes * (1-r)**no

        return like1 * like2


def MarginalProduct(suite):
    pmf = thinkbayes.Pmf()
    for (q, r), prob in suite.Items():
        pmf.Incr(q*r, prob)
    return pmf


def main():
    probs = numpy.linspace(0, 1, 101)

    hypos = []
    for q in probs:
        for r in probs:
            hypos.append((q, r))

    suite = Volunteer(hypos)

    data = 140, 50
    suite.Update(data)

    data = 5, 3, 1
    suite.Update(data)

    p_marginal = MarginalProduct(suite)
    q_marginal = suite.Marginal(0)
    r_marginal = suite.Marginal(1)

    thinkplot.Pmf(q_marginal)
    thinkplot.Pmf(r_marginal)
    thinkplot.Pmf(p_marginal)
    thinkplot.Show()

    
if __name__ == '__main__':
    main()
