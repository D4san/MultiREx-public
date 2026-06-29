# Configuration file for the Sphinx documentation builder.

import os
import sys
import sphinx_rtd_theme

# -- paquete multirex -------------
sys.path.insert(0, os.path.abspath('..'))

# -- Project information ---------------------------------------------------
project = 'MultiREx documentation'
copyright = '2025, David Duque-Castaño and Jorge I. Zuluaga'
author = 'David Duque-Castaño and Jorge I. Zuluaga'
release = 'v0.2.3'

# -- General configuration -------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',   # Para generar documentación automáticamente de docstrings
    'sphinx.ext.napoleon',  # Para soportar Google/NumPy style docstrings
    'nbsphinx',           # Para incluir notebooks en la documentación
    'sphinx.ext.mathjax',  # Para soportar matemáticas en la documentación
    # 'sphinx.ext.viewcode',  # (Opcional) Agrega enlaces "View Code" en la documentación
    # 'sphinx_autodoc_typehints',  # (Opcional) Muestra anotaciones de tipo en la doc
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -----------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_context = {
    "display_github": True,
    "github_user": "D4san",
    "github_repo": "MultiREx-public",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

## -- Css customization ---------------------------------------------------
def setup(app):
    app.add_css_file("custom.css")
