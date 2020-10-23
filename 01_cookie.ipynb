{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Bayesian Statistics Made Simple\n",
    "\n",
    "Code and exercises from my workshop on Bayesian statistics in Python.\n",
    "\n",
    "Copyright 2020 Allen Downey\n",
    "\n",
    "MIT License: https://opensource.org/licenses/MIT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The cookie problem\n",
    "\n",
    "> Suppose you have two bowls of cookies.  Bowl 1 contains 30 vanilla and 10 chocolate cookies.  Bowl 2 contains 20 vanilla of each.\n",
    ">\n",
    "> You choose one of the bowls at random and, without looking into the bowl, choose one of the cookies at random.  It turns out to be a vanilla cookie.\n",
    ">\n",
    "> What is the chance that you chose Bowl 1?\n",
    "\n",
    "Assume that there was an equal chance of choosing either bowl and an equal chance of choosing any cookie in the bowl.\n",
    "\n",
    "Here are the hypotheses and prior probabilities."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "hypos = 'Bowl 1', 'Bowl 2'\n",
    "probs = 1/2, 1/2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To compute the answer, I'll use a Pandas `Series` to represent the hypotheses and their probabilities."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "prior = pd.Series(probs, hypos)\n",
    "prior"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This `Series` represents a probability mass function (PMF)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we compute the likelihood of the data under each hypothesis.\n",
    "\n",
    "* The chance of getting a vanilla cookie from Bowl 1 is 3/4.\n",
    "\n",
    "* The chance of getting a vanilla cookie from Bowl 2 is 1/2."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "likelihood = 3/4, 1/2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The next step is to multiply the priors by the likelihoods:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "unnorm = prior * likelihood\n",
    "unnorm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The result is called `unnorm` because it is an \"unnormalized posterior\".\n",
    "\n",
    "To compute the posteriors, we have to divide through by $P(D)$, which is the sum of the unnormalized posteriors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "prob_data = unnorm.sum()\n",
    "prob_data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Notice that we get 5/8, which is what we got by computing $P(D)$ directly.\n",
    "\n",
    "Now we divide by `prob_data` to get the normalized posteriors:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "posterior = unnorm / prob_data\n",
    "posterior"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The posterior probability for Bowl 1 is 0.6, which is what we got using Bayes's Theorem explicitly.\n",
    "\n",
    "As a bonus, we also get the posterior probability of Bowl 2, which is 0.4.\n",
    "\n",
    "The posterior probabilities add up to 1, which they should, because the hypotheses are \"complementary\"; that is, either one of them is true or the other, but not both.\n",
    "\n",
    "When we add up the unnormalized posteriors and divide through, we force the posteriors to add up to 1.  This process is called \"normalization\", which is why the total probability of the data is also called the \"[normalizing constant](https://en.wikipedia.org/wiki/Normalizing_constant#Bayes'_theorem)\"\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Exercise 1\n",
    "\n",
    "Suppose we put the first cookie back, stir, draw another cookie from the same bowl, and it's a chocolate cookie.  What is the probability we drew both cookies from Bowl 1?\n",
    "\n",
    "Hint: The prior for the second update is the posterior from the first update."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "prior2 = posterior\n",
    "prior2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now \n",
    "\n",
    "1. Compute the likelihood of the data under each hypothesis,\n",
    "2. Multiply the new prior by the likelihoods.\n",
    "3. Divide through by the total probability of the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Solution goes here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Solution goes here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Solution goes here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Solution goes here"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 101 Bowls\n",
    "\n",
    "Suppose instead of 2 bowls there are 101 bowls:\n",
    "\n",
    "* Bowl 0 contains no vanilla cookies,\n",
    "\n",
    "* Bowl 1 contains 1% vanilla cookies,\n",
    "\n",
    "* Bowl 2 contains 2% vanilla cookies,\n",
    "\n",
    "and so on, up to\n",
    "\n",
    "* Bowl 99 contains 99% vanilla cookies, and\n",
    "\n",
    "* Bowl 100 contains all vanilla cookies.\n",
    "\n",
    "As in the previous problem, there are only two kinds of cookies, vanilla and chocolate.  So Bowl 0 is all chocolate cookies, Bowl 1 is 99% chocolate, and so on.\n",
    "\n",
    "Suppose we choose a bowl at random, choose a cookie at random, and it turns out to be vanilla.  What is the probability that the cookie came from Bowl $x$, for each value of $x$?\n",
    "\n",
    "To represent the prior, I'll use a Pandas `Series` with 101 equally spaced quantities from 0 to 1."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "xs = np.linspace(0, 1, num=101)\n",
    "prob = 1/101\n",
    "\n",
    "prior = pd.Series(prob, xs)\n",
    "prior.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here's what it looks like."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "prior.plot()\n",
    "\n",
    "plt.xlabel('Fraction of vanilla cookies')\n",
    "plt.ylabel('Probability')\n",
    "plt.title('Prior');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that we have a prior, we need to compute likelihoods.\n",
    "\n",
    "Here are the likelihoods for a vanilla cookie:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "likelihood_vanilla = xs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "And for a chocolate cookie."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "likelihood_chocolate = 1 - xs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To compute unnormalized posteriors, we multiply the priors and the likelihoods."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "unnorm = prior * likelihood_vanilla"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To normalize, we divide through by the total probability of the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "posterior = unnorm / unnorm.sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here's what the posterior looks like."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "posterior.plot()\n",
    "\n",
    "plt.xlabel('Fraction of vanilla cookies')\n",
    "plt.ylabel('Probability')\n",
    "plt.title('Posterior, one vanilla');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Suppose we put the first cookie back, stir the bowl, draw from the same bowl again, and get a vanilla cookie again.\n",
    "\n",
    "What's are the posterior probabilities now?\n",
    "\n",
    "We can do another update, using the posterior from the first draw as the prior for the second draw."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "prior2 = posterior\n",
    "unnorm2 = prior2 * likelihood_vanilla\n",
    "posterior2 = unnorm2 / unnorm2.sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "And here's what it looks like."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "posterior2.plot()\n",
    "\n",
    "plt.xlabel('Fraction of vanilla cookies')\n",
    "plt.ylabel('Probability')\n",
    "plt.title('Posterior, two vanilla');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Exercise 2\n",
    "\n",
    "Suppose we put the second cookie back, stir the bowl, draw from the same bowl again, and get a chocolate cookie.\n",
    "\n",
    "Now what are the posterior probabilities?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Solution goes here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Solution goes here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Solution goes here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
