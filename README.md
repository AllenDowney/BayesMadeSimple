## Bayesian Statistics Made Simple

[![tests](https://github.com/AllenDowney/BayesMadeSimple/actions/workflows/tests.yml/badge.svg)](https://github.com/AllenDowney/BayesMadeSimple/actions/workflows/tests.yml)


I presented this tutorial at PyCon US every year from 2012 to 2016 and at SciPy 2019. The most recent recording is [SciPy 2019](https://www.youtube.com/watch?v=-X0BiV9n_fQ); earlier ones are from PyCon [2016](https://www.youtube.com/watch?v=TpgiFIGXcT4), [2015](https://www.youtube.com/watch?v=5W715nfJNJw), [2014](https://www.youtube.com/watch?v=z_sIDLVisvA), [2013](https://www.youtube.com/watch?v=fH_as3eZ5YU), and [2012](https://www.youtube.com/watch?v=bobeo5kFz1g).

This is one of [my workshops and tutorials](https://allendowney.github.io/#workshops).


[Allen Downey](https://allendowney.com)

Bayesian statistical methods are becoming more common, but there are not many resources to help beginners get started. People who know Python can use their programming skills to get a head start. 

In this tutorial, I introduce Bayesian methods using grid algorithms, which help develop understanding and prepare for MCMC, which is a powerful algorithm for real-world problems.

It is based on my book, [*Think Bayes*](https://greenteapress.com/wp/think-bayes/), 
a [class I teach at Olin College](https://sites.google.com/site/compbayes18/), and my blog, [“Probably Overthinking It.”](https://allendowney.com/blog)

[Slides for this tutorial are here](https://docs.google.com/presentation/d/e/2PACX-1vTUIf7LJJpUd4NzInBGRyHnHqoZ4E736sqd6Iwq_ne3_aDXdJlNgO8O57_USzQzFfDx0gA44fniKe5R/pub).


### The notebooks

Each notebook comes in two versions: one with blanks for the exercises, and one with solutions.

| Notebook | Exercises | Solutions |
|---|---|---|
| 1. The cookie problem | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/01_cookie.ipynb) | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/01_cookie_soln.ipynb) |
| 2. The Euro problem | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/02_euro.ipynb) | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/02_euro_soln.ipynb) |
| 3. The Bayesian bandit | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/03_bandit.ipynb) | [Run on Colab](https://colab.research.google.com/github/AllenDowney/BayesMadeSimple/blob/master/03_bandit_soln.ipynb) |

The code in the videos is not the current version. The SciPy 2019 video matches the tag [`scipy-2019`](https://github.com/AllenDowney/BayesMadeSimple/tree/scipy-2019), and the PyCon videos match [`pycon-2016`](https://github.com/AllenDowney/BayesMadeSimple/tree/pycon-2016), [`pycon-2015`](https://github.com/AllenDowney/BayesMadeSimple/tree/pycon-2015), and [`pycon-2014`](https://github.com/AllenDowney/BayesMadeSimple/tree/pycon-2014).


### Running the notebooks

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
