"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import thinkbayes
import thinkplot

import numpy

"""
Problem: students sign up to participate in a community service
project.  Some fraction, q, of the students who sign up actually
participate, and of those some fraction, r, report back.

Given a sample of students who sign up and the number who report
back, we can estimate the product q*r, but don't learn much about
q and r separately.

If we can get a smaller sample of students where we know who
participated and who reported, we can use that to improve the
estimates of q and r.

And we can use that to compute the posterior distribution of the
number of students who participated.

"""

class Volunteer(thinkbayes.Suite):

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: pair of (q, r)
        data: one of two possible formats
        """
        if len(data) == 2:
            return self.Likelihood1(data, hypo)
        elif len(data) == 3:
            return self.Likelihood2(data, hypo)
        else:
            raise ValueError()

    def Likelihood1(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: pair of (q, r)
        data: tuple (signed up, reported)
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

        hypo: pair of (q, r)
        data: tuple (signed up, participated, reported)
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


def MarginalDistribution(suite, index):
    """Extracts the marginal distribution of one parameter.

    suite: Suite
    index: which parameter

    returns: Pmf
    """
    pmf = thinkbayes.Pmf()
    for t, prob in suite.Items():
        pmf.Incr(t[index], prob)
    return pmf


def MarginalProduct(suite):
    """Extracts the distribution of the product of the parameters.

    suite: Suite

    returns: Pmf
    """
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

    # update the Suite with the larger sample of students who
    # signed up and reported
    data = 140, 50
    suite.Update(data)

    # update again with the smaller sample of students who signed
    # up, participated, and reported
    data = 5, 3, 1
    suite.Update(data)

    #p_marginal = MarginalProduct(suite)
    q_marginal = MarginalDistribution(suite, 0)
    r_marginal = MarginalDistribution(suite, 1)

    thinkplot.Pmf(q_marginal, label='q')
    thinkplot.Pmf(r_marginal, label='r')
    #thinkplot.Pmf(p_marginal)
    thinkplot.Show()

    
if __name__ == '__main__':
    main()
