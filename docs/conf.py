# Sphinx configuration of the cuGMEC documentation (Read the Docs: see ../.readthedocs.yaml).
#
#   pip install -r docs/requirements.txt
#   sphinx-build -b html docs docs/_build/html

project = "cuGMEC"
author = "S. Y. Liu, P. Y. Jiang, G. Y. Fu and contributors"
copyright = "2025-2026, the cuGMEC authors"
release = "1.27.2609"

extensions = ["myst_parser", "sphinx.ext.mathjax", "sphinx_copybutton", "sphinxcontrib.bibtex"]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "plain"
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_inline"]
myst_dmath_double_inline = True
myst_heading_anchors = 3
numfig = True
math_numfig = True
math_eqref_format = "({number})"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "scripts"]

html_theme = "sphinx_rtd_theme"
html_title = "cuGMEC"
html_logo = None
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
