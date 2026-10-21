# Bayesian Statistics Made Simple

[![tests](https://github.com/AllenDowney/BayesMadeSimple/actions/workflows/tests.yml/badge.svg)](https://github.com/AllenDowney/BayesMadeSimple/actions/workflows/tests.yml)

[Allen Downey](https://allendowney.com)

Bayesian statistical methods are becoming more common, but there are not many resources to help beginners get started. People who know Python can use their programming skills to get a head start.

In this tutorial, I introduce Bayesian methods using grid algorithms, which help develop understanding and prepare for MCMC, which is a powerful algorithm for real-world problems.

It is based on my book, [*Think Bayes*](https://allendowney.github.io/ThinkBayes2/), and my blog, [*Probably Overthinking It*](https://allendowney.com/blog).

This is one of [my workshops and tutorials](https://allendowney.github.io/#workshops).

**Level:** beginner. You should be comfortable with basic Python; some experience with NumPy and pandas helps. You don't need any background in statistics.

**Length:** about 90 minutes.

## Outline

1. **The cookie problem:** Bayes's theorem as a table of priors, likelihoods and posteriors, using a pandas `Series`; then the 101 Bowls problem, which leads to distributions of quantities.
2. **The Euro problem:** estimating the probability that a coin lands heads; uniform and triangle priors, the posterior mean, and credible intervals.
3. **The Bayesian bandit:** updating beliefs about several slot machines as you play them, and using those beliefs to decide which one to play next (Thompson sampling).

Each notebook comes in two versions: one with blanks for the exercises, and one with solutions.

| Notebook | Exercises | Solutions |
|---|---|---|
| 1. The cookie problem | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/01_cookie.ipynb) | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/01_cookie_soln.ipynb) |
| 2. The Euro problem | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/02_euro.ipynb) | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/02_euro_soln.ipynb) |
| 3. The Bayesian bandit | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/03_bandit.ipynb) | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/03_bandit_soln.ipynb) |

[Slides for this tutorial are here](https://docs.google.com/presentation/d/e/2PACX-1vTUIf7LJJpUd4NzInBGRyHnHqoZ4E736sqd6Iwq_ne3_aDXdJlNgO8O57_USzQzFfDx0gA44fniKe5R/pub).

## Recordings

I presented this tutorial at PyCon US every year from 2012 to 2016 and at SciPy 2019. The code in the videos is not the current version; the tag after each link is the version of the code that goes with it.

* [SciPy 2019](https://www.youtube.com/watch?v=-X0BiV9n_fQ): tag [`scipy-2019`](https://github.com/AllenDowney/BayesMadeSimple/tree/scipy-2019)
* [PyCon 2016](https://www.youtube.com/watch?v=TpgiFIGXcT4): tag [`pycon-2016`](https://github.com/AllenDowney/BayesMadeSimple/tree/pycon-2016)
* [PyCon 2015](https://www.youtube.com/watch?v=5W715nfJNJw): tag [`pycon-2015`](https://github.com/AllenDowney/BayesMadeSimple/tree/pycon-2015)
* [PyCon 2014](https://www.youtube.com/watch?v=z_sIDLVisvA): tag [`pycon-2014`](https://github.com/AllenDowney/BayesMadeSimple/tree/pycon-2014)
* [PyCon 2013](https://www.youtube.com/watch?v=fH_as3eZ5YU) and [PyCon 2012](https://www.youtube.com/watch?v=bobeo5kFz1g): the code from those years is not in this repository.

## Running the notebooks

The easiest way to run the notebooks is on Colab: click one of the links above. You need a Google account, but you don't have to install anything.

To run them on your own computer, you need Python 3 with Jupyter, NumPy, pandas, SciPy, Matplotlib, and Seaborn. You can [download the code in this zip file](https://github.com/AllenDowney/BayesMadeSimple/archive/master.zip), or clone the repository:

```
git clone https://github.com/AllenDowney/BayesMadeSimple
cd BayesMadeSimple
```

Then install the packages with pip:

```
pip install -r requirements.txt
```

Or, if you use conda or mamba, create an environment:

```
mamba env create -f environment.yml
mamba activate BayesMadeSimple
```

Then run `jupyter lab` and open `01_cookie.ipynb`.

## Next steps

To go further, read [*Think Bayes*](https://allendowney.github.io/ThinkBayes2/), which is free online, or see [my other workshops](https://allendowney.github.io/#workshops), including ones on Bayesian decision analysis and on PyMC.

## License

The code is available under the [MIT License](LICENSE). The text is available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
