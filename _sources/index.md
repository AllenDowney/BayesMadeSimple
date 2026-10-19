---
myst:
  html_meta:
    description: "A beginner's tutorial on Bayesian statistics in Python: three notebooks that use grid algorithms and pandas to solve problems with Bayes's theorem."
og:description: "A beginner's tutorial on Bayesian statistics in Python: three notebooks that use grid algorithms and pandas to solve problems with Bayes's theorem."
---

# Bayesian Statistics Made Simple: A Tutorial in Python

[Allen Downey](https://allendowney.com)

Bayesian statistical methods are becoming more common, but there are not many resources to help beginners get started. People who know Python can use their programming skills to get a head start.

In this tutorial, I introduce Bayesian methods using grid algorithms, which help develop understanding and prepare for MCMC, which is a powerful algorithm for real-world problems.

It is based on my book, [*Think Bayes*](https://allendowney.github.io/ThinkBayes2/), and my blog, [*Probably Overthinking It*](https://allendowney.com/blog).

**Level:** beginner. You should be comfortable with basic Python; some experience with NumPy and pandas helps. You don't need any background in statistics.

**Length:** about 90 minutes.

## The notebooks

The pages on this site show the notebooks with solutions. To do the exercises, run the versions with blanks on Colab:

1. [The cookie problem](01_cookie): Bayes's theorem as a table of priors, likelihoods and posteriors; then the 101 Bowls problem. [Exercises on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/01_cookie.ipynb)
2. [The Euro problem](02_euro): estimating the probability that a coin lands heads; uniform and triangle priors, the posterior mean, and credible intervals. [Exercises on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/02_euro.ipynb)
3. [The Bayesian bandit problem](03_bandit): updating beliefs about several slot machines, and using them to decide which one to play next. [Exercises on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/03_bandit.ipynb)

[Slides for this tutorial are here](https://docs.google.com/presentation/d/e/2PACX-1vTUIf7LJJpUd4NzInBGRyHnHqoZ4E736sqd6Iwq_ne3_aDXdJlNgO8O57_USzQzFfDx0gA44fniKe5R/pub).

## Recordings

I presented this tutorial at PyCon US every year from 2012 to 2016 and at SciPy 2019. The most recent recording is from [SciPy 2019](https://www.youtube.com/watch?v=-X0BiV9n_fQ). The code in the videos is an earlier version; the [repository README](https://github.com/AllenDowney/BayesMadeSimple) lists the version that goes with each video.

## Next steps

To go further, read [*Think Bayes*](https://allendowney.github.io/ThinkBayes2/), which is free online, or see [my other workshops](https://allendowney.github.io/#workshops).

The code is on [GitHub](https://github.com/AllenDowney/BayesMadeSimple), under the MIT License.
