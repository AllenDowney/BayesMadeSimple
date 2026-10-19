"""Copy the solution notebooks into jb/ and prepare them for the book.

For each ../NN_name_soln.ipynb, writes NN_name.ipynb here (so the page URL has no _soln):
- the first H2 heading (e.g. "## The cookie problem") becomes the page's H1, in place
  of the "# Bayesian Statistics Made Simple" that all three notebooks share;
- H3 headings in that first section move up to H2;
- the page description goes in the notebook metadata (see GreenTeaPress plans/seo.md).
"""

import nbformat

DESCRIPTIONS = {
    '01_cookie': "Bayes's theorem as a table: solve the cookie problem and the 101 Bowls problem in Python, "
                 "using a pandas Series for priors, likelihoods and posteriors.",
    '02_euro': "Estimate the probability that a coin lands heads with a Bayesian update in Python: "
               "uniform and triangle priors, MAP, posterior mean and credible intervals.",
    '03_bandit': "Bayesian bandits in Python: update beliefs about slot machines as you play them, "
                 "then use Thompson sampling to choose which machine to play next.",
}


def prepare(name, description):
    nb = nbformat.read(f'../{name}_soln.ipynb', 4)

    first = nb.cells[0]
    assert first.source.startswith('# Bayesian Statistics Made Simple'), name
    for cell in nb.cells[1:]:
        if cell.cell_type == 'markdown' and cell.source.startswith('## '):
            heading, _, rest = cell.source.partition('\n')
            cell.source = rest.lstrip('\n')
            break
    title = heading[3:].strip()
    first.source = first.source.replace('# Bayesian Statistics Made Simple', f'# {title}', 1)

    # The first section's H2 is now the H1, so its H3s (e.g. "### Exercise 1") move up to H2
    start = nb.cells.index(cell)
    for later in nb.cells[start + 1:]:
        if later.cell_type != 'markdown':
            continue
        if later.source.startswith('## '):
            break
        later.source = '\n'.join(line[1:] if line.startswith('### ') else line
                                  for line in later.source.split('\n'))

    nb.metadata['myst'] = {'html_meta': {'description': description}}
    nb.metadata['og:description'] = description
    nbformat.write(nb, f'{name}.ipynb')
    print(f'{name}: {title} ({len(description)} characters)')


for name, description in DESCRIPTIONS.items():
    prepare(name, description)
