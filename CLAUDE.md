# BayesMadeSimple

Allen Downey's tutorial "Bayesian Statistics Made Simple": an introduction to Bayesian methods using grid algorithms in Python (NumPy and pandas). Given at PyCon US 2012–2016 and SciPy 2019. The README is also the GitHub Pages site (`allendowney.github.io/BayesMadeSimple/`, Jekyll minimal theme via `_config.yml`).

**Guide for the 2026 refresh:** the "Refresh checklist" in `~/GreenTeaPress/plans/workshops.md`. Progress goes on `PROJECT_BOARD.md` here (Task 1) and in GreenTeaPress Task 24.

## Which notebooks are the workshop

The current version (October 2020) is three notebooks, each with a student version (blanks) and a solution version:

- `01_cookie.ipynb` / `01_cookie_soln.ipynb`
- `02_euro.ipynb` / `02_euro_soln.ipynb`
- `03_bandit.ipynb` / `03_bandit_soln.ipynb`

They need only NumPy, pandas, SciPy, Matplotlib and Seaborn; no data files and no helper modules.

The `zigzag*.ipynb` notebooks are from a separate talk ("Bayesian Zig Zag", PyMC3) and are not part of the workshop or its tests. Earlier versions of the workshop were removed from `master` in 2026; they are in the tags below and in git history.

## Tags

- `scipy-2019`: the code in the SciPy 2019 video.
- `pycon-2014`, `pycon-2015`, `pycon-2016`: the code in the PyCon videos (2012 and 2013 predate the repo).
- `pre-refresh-2026`: the state before the 2026 refresh.

## Testing

Environment: `mamba env create -f environment.yml` (name `BayesMadeSimple`). Run the solution notebooks top to bottom (not the student versions, which have blanks):

```
pytest --nbmake 01_cookie_soln.ipynb 02_euro_soln.ipynb 03_bandit_soln.ipynb
```

The target environment is Google Colab's current Python and libraries.

## Conventions

- Don't change the substance of the text; list content questions under "Content issues" on the board for Allen to decide.
- Keep student and solution versions in sync: same cells, solutions replaced by blanks.
