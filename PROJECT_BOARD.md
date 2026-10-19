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
- [ ] README: "The code in the SciPy 2019 video matches tag `scipy-2019`" (with step 5)
- [x] Inventory (below)

**1. Runs in a current environment**
- [x] `environment.yml` rewritten (2026-10-19): Python 3.12, conda-forge, the five libraries plus Jupyter, pytest, nbmake; the old 2020 edits (pymc3, arviz for Zig Zag) dropped. Local env: `mamba env create -f environment.yml` (created as `BayesMadeSimple`: NumPy 2.5, pandas 3.0, SciPy 1.18, Matplotlib 3.11, Seaborn 0.13)
- [ ] `requirements.txt` for pip/Colab users; confirm Colab's current Python
- [ ] First code cell installs what Colab lacks (probably nothing for the three workshop notebooks)
- [ ] Fix breakage and deprecation warnings
- [ ] Replace Binder instructions with Colab

**2. Tests and CI**
- [ ] `pytest --nbmake` on the three solution notebooks
- [ ] GitHub Actions (push, PR, monthly), status badge

**3. Colab**
- [ ] "Open in Colab" links in each notebook and the README
- [ ] Allen runs one notebook on Colab end to end

**4. Text**
- [ ] Typos and broken links only; everything else under "Content issues"

**5. README in the standard shape**
- [ ] Abstract, level, length, outline with Colab links, slides, recordings (with tags), license, links to *Think Bayes* and the hub

**6. Site and visibility**
- [ ] Pages site is the README (Jekyll); SEO checklist items that apply (description, title)
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

(For Allen to decide; none yet.)
