# BayesMadeSimple Project Board

Numbered tasks for tracking work. Each task has a permanent number; add new tasks at the end. Update status as work progresses.

**Status values:** `Pending`, `In progress`, `DONE`

---

## Task 1: Refresh

**Status:** In progress (2026-10-19)

**Context:** BayesMadeSimple is a "current" workshop (GreenTeaPress Task 24) and the pilot for the refresh checklist in `~/GreenTeaPress/plans/workshops.md` ("Refresh checklist"). That checklist is the guide; revise it after this pilot. 358 stars; last content change 2020-10-23. The most recent recording is [SciPy 2019](https://www.youtube.com/watch?v=-X0BiV9n_fQ).

**0. Set up**
- [x] `PROJECT_BOARD.md` and `CLAUDE.md` (2026-10-19)
- [x] Tags (pushed 2026-10-19): `scipy-2019` at `bfce408` (2019-04-25, the last commit before the 2019-07-15 "Post workshop updates", which changed the `Pmf` API used in the video, e.g. `d6 + d6` → `d6.add_dist(d6)`); `pre-refresh-2026` at `39c99ac`; PyCon 2014–2016 tags (Decisions, below)
- [x] README: each recording lists its tag
- [x] Inventory (below)

**1. Runs in a current environment**
- [x] `environment.yml` rewritten (2026-10-19): Python 3.12, conda-forge, the five libraries plus Jupyter, pytest, nbmake; the old 2020 edits (pymc3, arviz for Zig Zag) dropped. Local env: `mamba env create -f environment.yml` (created as `BayesMadeSimple`: NumPy 2.5, pandas 3.0, SciPy 1.18, Matplotlib 3.11, Seaborn 0.13)
- [x] `requirements.txt` (unpinned; Colab already has all five libraries) and `requirements-dev.txt` (adds Jupyter, pytest, nbmake). Colab is on Python 3.12 since 2025-08 (release notes)
- [x] First code cell installs what Colab lacks: nothing to install, no data or helper files, so no install cell
- [x] Breakage and deprecation warnings: none. All six notebooks (student versions too) run with Deprecation/Future/PendingDeprecation/User/Runtime warnings as errors (NumPy 2.5, pandas 3.0, SciPy 1.18, Matplotlib 3.11)
- [x] README install section rewritten: Colab first, then pip or mamba; Binder and Anaconda/Python 3.7 instructions removed; notebook table with Colab links; which tag matches which video

**2. Tests and CI**
- [x] `pytest --nbmake` on all six notebooks (the student versions run too; their blanks are placeholders)
- [x] GitHub Actions `.github/workflows/tests.yml` (push to master, PR, monthly, manual; Python 3.12; pip; 20-minute timeout), badge in the README

**3. Colab**
- [x] "Click here to run this notebook on Colab" cell after the title of each notebook (solutions also link to the version with blanks), and in the README table
- [ ] Allen runs one notebook on Colab end to end

**4. Text**
- [x] Typos: "20 vanilla of each" → "20 of each", "What's are", "Maximum Aposteori Probability" → "maximum a posteriori probability", "return" → "returns", "wining", "an end itself" → "an end in itself", a missing period; `credible_interval` docstring said "Compute the mean of a PMF"
- [x] Links: README `https:allendowney.com` fixed, http → https (Green Tea Press, blog, MacKay), LessWrong wiki link updated to its new URL; everything else under "Content issues"

**5. README in the standard shape**
- [x] README rewritten (2026-10-19): abstract, level (beginner) and length (90 minutes, Allen), outline with Colab table, slides, recordings with the matching tag for each, running instructions, next steps (*Think Bayes* 2e, hub), license; `LICENSE` file (MIT, as the notebooks say)

**6. Site and visibility**
- [x] `jb/` (Jupyter Book 1, 2026-10-19): `build.sh [publish]` runs `prep_notebooks.py` (copies the solution notebooks in without `_soln`, makes each one's first H2 its title, adds the page description) and `jb build` (executes the notebooks). The pages link to the versions with blanks on Colab. Build tools are in `environment.yml`
- [x] SEO checklist in the test build: sitemap lists only the 4 pages; each page has one description, one `og:description`, `rel="canonical"`; landing H1 differs from the site title; page-view counter in `extra_footer`; social image is the *Think Bayes* 2e cover. No `jupytext.toml` (no `.md` pairs)
- [ ] Allen reviews the four descriptions (`jb/index.md` front matter, `jb/prep_notebooks.py`)
- [ ] Publish: `./build.sh publish`, switch Pages from `master` / (Jekyll) to `gh-pages`, delete the root `_config.yml`; `Sitemap:` line in `~/AllenDowney.github.io/robots.txt`; Allen submits the sitemap in Search Console; tick in GreenTeaPress Task 22
- [ ] Hub entry, repo description and topics

**7. Close**
- [ ] Tag `2026-refresh`; note in GreenTeaPress Task 24

### Inventory (2026-10-19)

**The workshop (October 2020 version, 3 notebooks, student + solution):**

| Notebook | Topic | Imports |
|---|---|---|
| `01_cookie` / `01_cookie_soln` | Cookie problem, 101 bowls; Bayes tables with pandas `Series` | numpy, pandas, matplotlib, seaborn |
| `02_euro` / `02_euro_soln` | Euro problem (MacKay) | + `scipy.interpolate` |
| `03_bandit` / `03_bandit_soln` | Bayesian bandit | + `scipy.interpolate`, `collections` |

No data files, no helper modules, no install cell, no Colab links. Kernel metadata says Python 3.8.5. Student versions have no outputs; solutions are partly executed. The README predates this version: it describes the 2019 notebooks (`empyrical-dist`, Anaconda Python 3.7, Binder).

**Earlier versions (leftovers; deleted from `master` 2026-10-19 except Zig Zag):**

- *SciPy 2019 version* (uses `empiricaldist` and a `!pip install` cell for Colab): `cookie_soln`, `dice_soln`, `euro_soln`, `bandit_soln`, `world_cup_soln` (solutions, renamed in the 2020 "Reorg"), plus the student versions `03_euro.ipynb` and `04_bandit.ipynb`. Their student versions of cookie, dice and world cup were deleted in 2020. `03_euro.ipynb` next to `02_euro.ipynb` is confusing.
- *2014–2016 PyCon version* (Python 2-era, GPL, `thinkbayes`/`thinkbayes2`/`thinkplot`): `world_cup01.ipynb` (Python 2.7 kernel), `world_cup01soln.ipynb`, `world_cup02.ipynb`; scripts `dice.py`, `dice_soln.py`, `euro.py`, `euro_soln.py`, `euro2.py`, `euro2_soln.py`, `train.py`, `train2.py`, `train_soln.py`, `sat.py`, `sat_soln.py` (reads `sat_ranks.csv`, `sat_scale.csv`), `lincoln.py`, `volunteer.py`, `billiards.py`, `install_test.py`; helper copies `thinkbayes.py`, `thinkbayes2.py`, `thinkplot.py`.
- *2019 Pmf prototypes:* `distribution.py`, `empyrical_dist.py` (became the `empiricaldist` package), used by `flip.ipynb`.
- *"Bayesian Zig Zag"* (hockey / Poisson, grid methods and PyMC3; 2018–2020, a separate talk): `zigzag.ipynb`, `zigzag2.ipynb`, `zigzag3.ipynb` (with `empiricaldist`, `pymc3`, `arviz` install cell); `debug.ipynb` (PyMC3 + `thinkbayes2`).
- `tutorial.md`: an older copy of the install instructions (Python 2 or 3, Binder).

**Local state not in git (discarded 2026-10-19):**

- Uncommitted edits from 2020-10: `zigzag*.ipynb` (re-run outputs, kernel version, `!pip install arviz`). Not tagged.
- Untracked: `zigzag1.png`–`zigzag19.png`, `zigzag3a/3b`, `zigzag16a/16b`, `poisson_model1/2.png` and `.xml` (slide figures, probably for the Zig Zag talk); `.ipynb_checkpoints/`. No `.gitignore`.

**Runs on current libraries:** the three solution notebooks execute top to bottom in the `BayesMadeSimple` env with no errors or warnings (2026-10-19, `nbconvert --execute` to a scratch copy; the repo notebooks are unchanged).

### Decisions (Allen, 2026-10-19)

- Tags pushed, plus `pycon-2014` (`27751fc`), `pycon-2015` (`3137373`), `pycon-2016` (`6ebedce`): the last commit before each tutorial. PyCon 2012 and 2013 predate the repo (first commit 2014-03-19).
- Leftovers deleted from `master` (they stay in the tags and history): the SciPy 2019 notebooks, the 2014–2016 scripts and data, the `thinkbayes`/`thinkplot` copies, the Pmf prototypes with `flip.ipynb`, `debug.ipynb`, `tutorial.md`.
- Zig Zag notebooks stay as committed; the uncommitted 2020 re-runs, slide figures and checkpoints were discarded; `.gitignore` added.
- October 2020 version: venue unknown; leave it off the README for now.

### Content issues

(For Allen to decide.)

Resolved 2026-10-19 (Allen): Olin class mention dropped from the README; *Think Bayes* links to the 2e site; the four notebook text fixes made (cookie: the two "what we got" sentences reworded; Euro: "0 to 100" → "0 to 1", "takes a Pmf" → "takes a `Series` that represents a PMF", "We have already seen … MAP" → "One way … is the MAP"); `LICENSE` added.

Open:

1. The slides (Google Slides, 2019?) may not match the 2020 notebooks; check before the next time you give it.
2. Copyright lines in the notebooks say 2020.
3. The October 2020 version's venue, if any.
