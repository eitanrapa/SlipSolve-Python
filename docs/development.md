# Development

## Setting up

```bash
git clone https://github.com/eitanrapa/SlipSolve-Python.git
cd SlipSolve-Python
pip install -e ".[dev]"
```

or create the conda environment (`conda env create -f environment.yml`), which
includes the test dependencies, and run `pip install -e .` inside it.

## Running the tests

```bash
pytest                 # the whole suite, from the repository root
pytest -n auto         # in parallel, with pytest-xdist
pytest tests/test_slip_greens.py -q
```

Every test is synthetic: the data are forward-modelled from planted slip, so no
data files, downloads or environment variables are needed. A few tests cross-check
the triangular dislocation against `cutde`, an independent implementation, and are
skipped when it is not installed. The layered tests build their tables by the
package's own wavenumber integration, on small crusts.

Some tests deserve a note when changing the code:

- `tests/test_slip_greens.py` pins the sign of slip (positive strike-slip is
  left-lateral). Getting it backwards does not crash anything; it makes a polarity
  bound enforce the wrong sense of motion.
- `tests/test_slip_inversion.py` plants a slip patch, forward-models it through two
  look geometries, samples, inverts and requires the patch back. Its synthetic data
  go from displacement to phase and back to LOS, so a sign error in the conversion
  cannot cancel against the same error in the inversion.
- `tests/slip_synthetic.py` holds the shared builders (look vectors, synthetic LOS
  grids, a tapered slip patch, correlated noise); the tests import it directly.

## Building the documentation

The site is built with [MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/); the API pages
are generated from the docstrings by
[mkdocstrings](https://mkdocstrings.github.io/) through static analysis, so the
package does not need to be importable to build them.

```bash
pip install -e ".[docs]"
mkdocs serve             # live preview at http://127.0.0.1:8000
mkdocs build --strict    # what the continuous integration runs
```

A GitHub Actions workflow (`.github/workflows/docs.yml`) builds the site on every
push and pull request, and deploys it to GitHub Pages on a push to `main`.

Docstrings are rendered as Markdown with the numpy docstring style: use single
backticks for code, not reStructuredText roles, and fenced or indented blocks for
examples. Math in the guide pages is written as `\( ... \)` and `\[ ... \]`.
