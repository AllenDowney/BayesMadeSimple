"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

"""This file contains a partial solution to a problem from
MacKay, "Information Theory, Inference, and Learning Algorithms."

    Exercise 3.15 (page 50): A statistical statement appeared in
    "The Guardian" on Friday January 4, 2002:

        When spun on edge 250 times, a Belgian one-euro coin came
        up heads 140 times and tails 110.  'It looks very suspicious
        to me,' said Barry Blight, a statistics lecturer at the London
        School of Economics.  'If the coin weere unbiased, the chance of
        getting a result as extreme as that would be less than 7%.'

MacKay asks, "But do these data give evidence that the coin is biased
rather than fair?"

"""

import thinkbayes


class Euro(thinkbayes.Suite):

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: integer value of x, the probability of heads (0-100)
        data: tuple (#heads, #tails)
        """
        x = hypo / 100.0
        heads, tails = data
        like = x**heads * (1-x)**tails
        return like


def AverageLikelihood(suite, data):
    """Computes the average likelihood over all hypothesis in suite.

    Args:
      suite: Suite of hypotheses
      data: some representation of the observed data

    Returns:
      float
    """
    total = 0

    for hypo, prob in suite.Items():
        like = suite.Likelihood(data, hypo)
        total += prob * like

    return total


def main():
    fair = Euro()
    fair.Set(50, 1)

    bias = Euro()
    for x in range(0, 101):
        if x != 50:
            bias.Set(x, 1)
    bias.Normalize()

    # notice that we've changed the representation of the data
    data = 140, 110

    like_bias = AverageLikelihood(bias, data)
    print 'like_bias', like_bias

    like_fair = AverageLikelihood(fair, data)
    print 'like_fair', like_fair

    ratio = like_bias / like_fair
    print 'Bayes factor', ratio

    
if __name__ == '__main__':
    main()
