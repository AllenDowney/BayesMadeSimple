"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkbayes
import thinkplot


class Train(thinkbayes.Suite):
    """Represents hypotheses about how many trains the company has.

    The likelihood function for the train problem is the same as
    for the Dice problem.
    """
    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: number of trains the carrier operates
        data: the number of the observed train
        """
        if hypo < data:
            return 0
        else:
            return 1.0/hypo



def main():
    hypos = range(1, 101)
    suite = Train(hypos)

    suite.Update(25)
    print('Posterior mean', suite.Mean())
    print('Posterior MLE', suite.MaximumLikelihood())
    print('Posterior CI 90', suite.CredibleInterval(90))

    thinkplot.PrePlot(1)
    thinkplot.Pmf(suite, linewidth=5)
    thinkplot.Save(root='train2',
                   xlabel='Number of trains',
                   ylabel='Probability',
                   formats=['png'])

    thinkplot.Pmf(suite, linewidth=5, color='0.8')
    suite.Update(42)
    print('Posterior mean', suite.Mean())
    print('Posterior MLE', suite.MaximumLikelihood())
    print('Posterior CI 90', suite.CredibleInterval(90))

    thinkplot.PrePlot(1)
    thinkplot.Pmf(suite, linewidth=5)
    thinkplot.Save(root='train3',
                   xlabel='Number of trains',
                   ylabel='Probability',
                   formats=['png'])


if __name__ == '__main__':
    main()
