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
        # fill this in!
        return 1



def main():
    hypos = xrange(100, 1001)
    suite = Train(hypos)

    suite.Update(321)

    thinkplot.PrePlot(1)
    thinkplot.Pmf(suite)
    thinkplot.Show(xlabel='Number of trains',
                   ylabel='Probability',
                   legend=False)


if __name__ == '__main__':
    main()
