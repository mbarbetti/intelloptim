"""Sphinx configuration for the public intelloptim documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

from intelloptim.version import __version__

project = "intelloptim"
author = "Matteo Barbetti"
copyright = "2026, Matteo Barbetti"
release = __version__
version = ".".join(__version__.split(".")[:2])
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "description"
napoleon_google_docstring = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}
html_theme = "furo"
html_title = "intelloptim"
html_static_path = []
