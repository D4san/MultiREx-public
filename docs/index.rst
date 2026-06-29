MultiREx - Planetary Transmission Spectra Generator
===================================================

.. image:: https://img.shields.io/pypi/v/multirex?color=blue
   :target: https://pypi.org/project/multirex/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/dw/multirex
   :target: https://pypi.org/project/multirex/
   :alt: PyPI downloads

.. image:: https://img.shields.io/pypi/l/multirex
   :target: https://pypi.org/project/multirex/
   :alt: License

.. image:: https://img.shields.io/pypi/implementation/multirex
   :target: https://pypi.org/project/multirex/
   :alt: Python implementation

.. image:: https://img.shields.io/pypi/pyversions/multirex
   :target: https://pypi.org/project/multirex/
   :alt: Python versions

.. image:: https://img.shields.io/badge/Article-MNRAS-blue.svg?style=flat
   :target: https://academic.oup.com/mnras/article/539/2/1528/8109635
   :alt: MNRAS Article

.. image:: https://img.shields.io/badge/GitHub-Repository-black?logo=github
   :target: https://github.com/D4san/MultiREx-public
   :alt: GitHub Repository

.. image:: https://readthedocs.org/projects/multirex-documentation/badge/?version=latest
   :target: https://multirex-documentation.readthedocs.io/en/latest/
   :alt: Documentation Status

MultiREx is a Python library designed for generating synthetic exoplanet
transmission spectra. This tool extends the functionalities of the
`TauREx <https://taurex3.readthedocs.io/en/latest/index.html>`_
library, reorganizing and enabling the massive generation of spectra and
observations with added noise. The package was originally devised for training
large machine learning models to identify biosignatures in noisy spectra, but it
can also be used for other purposes.

For the science behind the model, please refer to the following paper:

David S. Duque-Castaño, Jorge I. Zuluaga, and Lauren Flor-Torres (2024),
**Machine-assisted classification of potential biosignatures in Earth-like
exoplanets using low signal-to-noise ratio transmission spectra**,
`MNRAS <https://academic.oup.com/mnras/article/539/2/1528/8109635>`__.

The notebooks used to develop this work are available in the
`examples/papers/DZF-MLBiosignatureClassification <https://github.com/D4san/MultiREx-public/tree/main/examples/papers/DZF-MLBiosignatureClassification>`_ directory.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   notebooks/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Examples:

   notebooks/parameter_exploration
   notebooks/chemical_equilibrium

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules
   

Key Features of MultiREx
------------------------

- **Planetary System Assembly**: Combine different stars, planets, and atmospheres
  to explore various system configurations.
- **Customizable Atmospheres**: Configure and add a wide range of atmospheric
  compositions to planets.
- **Synthetic Spectrum Generation**: Produce realistic spectra based on your
  system's attributes.
- **Astronomical Observation Simulation**: Use ``randinstrument`` to simulate
  spectral observations with noise determined by signal-to-noise ratios.
- **Multiverse Analysis**: Automate the generation of multiple spectra with
  random parameter variations for extensive statistical or ML-based analyses.

Downloading and Installing MultiREx
-----------------------------------

``MultiREx`` is available on the Python Package Index (PyPI) and can be installed
on Linux using:

.. code-block:: bash

   sudo pip install multirex

This command installs all dependencies and downloads useful data, scripts, and
constants.

.. note::

   If you do not have access to ``sudo``, you can install ``MultiREx`` in
   your local environment (usually at ``~/.local/``). In that case, you
   need to add the local Python installation path to your ``PATH`` environment
   variable. For instance, add the following line to your ``~/.bashrc`` or
   ``~/.bash_profile``:

   .. code-block:: bash

      export PATH=$HOME/.local/bin:$PATH

If you are a developer or want to work directly with the package sources, clone
``MultiREx`` from the `GitHub repository <https://github.com/D4san/MultiREx-public>`_:

.. code-block:: bash

   git clone https://github.com/D4san/MultiREx-public

To install the package from the sources, run:

.. code-block:: bash

   cd MultiREx-public
   python3 setup.py install

Running MultiREx in Google Colab
--------------------------------

To run ``MultiREx`` in Google Colab, execute:

.. code-block:: shell

   !pip install -Uq multirex

After installing, **reset the Colab session** before importing the package. This
avoids unintended behavior from the package ``pybtex``. Once reset, simply import:

.. code-block:: python

   import multirex as mrex

Quickstart
----------

To start using ``MultiREx``, import the package:

.. code-block:: python

   import multirex as mrex

``MultiREx`` models a planetary system with three main components: a star, a
planet, and a planetary atmosphere.

First, define a star:

.. code-block:: python

   star = mrex.Star(temperature=5777, radius=1, mass=1)
   # Radius and mass are in solar units.

Then, define a planet:

.. code-block:: python

   planet = mrex.Planet(radius=1, mass=1)
   # Radius and mass are in Earth units.

Now give the planet an atmosphere. This is a basic example of an N2 atmosphere
with 100 ppm of CO2 and 1 ppm of CH4:

.. code-block:: python

   atmosphere = mrex.Atmosphere(
       temperature=288,    # in K
       base_pressure=1e5,  # in Pa
       top_pressure=1,     # in Pa
       fill_gas="N2",      # the gas that fills the atmosphere
       composition=dict(
           CO2=-4,         # log10(mix-ratio)
           CH4=-6,
       )
   )
   planet.set_atmosphere(atmosphere)

Assemble the system:

.. code-block:: python

   system = mrex.System(star=star, planet=planet, sma=1)
   # sma (semimajor axis) is in AU (astronomical units).

Create a transmission model:

.. code-block:: python

   system.make_tm()

Plot the transmission spectrum:

.. code-block:: python

   wns = mrex.Physics.wavenumber_grid(wl_min=0.6, wl_max=10, resolution=1000)
   fig, ax = system.plot_contributions(wns, xscale='log')

.. image:: https://github.com/D4san/MultiREx-public/blob/main/examples/resources/contributions-transmission-spectrum.png?raw=true
   :alt: Contributions in transmission spectrum
   :align: center
   :width: 500

All of these functionalities are also available in ``Taurex``. However, the
interface in ``MultiREx`` is more intuitive and is best suited for the package’s
true strength: creating large ensembles of random planetary systems.

For instance, you can create a random planetary system by specifying parameter
ranges:

.. code-block:: python

   system = mrex.System(
       star=mrex.Star(
           temperature=(4000, 6000),
           radius=(0.5, 1.5),
           mass=(0.8, 1.2),
       ),
       planet=mrex.Planet(
           radius=(0.5, 1.5),
           mass=(0.8, 1.2),
           atmosphere=mrex.Atmosphere(
               temperature=(290, 310),       # K
               base_pressure=(1e5, 10e5),     # Pa
               top_pressure=(1, 10),          # Pa
               fill_gas="N2",
               composition=dict(
                   CO2=(-5, -4),  # log10(mix-ratio)
                   CH4=(-6, -5),
               ),
           )
       ),
       sma=(0.5, 1)  # in AU
   )

Using this approach, you can generate thousands of spectra for training machine
learning models or other statistical analyses.

.. image:: https://github.com/D4san/MultiREx-public/blob/main/examples/resources/synthetic-transmission-spectra.png?raw=true
   :alt: Synthetic transmission spectra
   :align: center
   :width: 500

Further Examples
----------------

Several Jupyter notebooks illustrating both basic and advanced functionalities
are available in the `examples/ <https://github.com/D4san/MultiREx-public/tree/main/examples>`_ folder.
Additionally, notebooks used to generate figures for our papers are in
`examples/papers/ <https://github.com/D4san/MultiREx-public/tree/main/examples/papers>`_.

A Note About TauREx
-------------------

MultiREx is built on the spectral calculation capabilities of
`TauREx <https://taurex3.readthedocs.io/en/latest/index.html>`_. 
If you use ``MultiREx`` in your research, please also cite TauREx:

A. F. Al-Refaie, Q. Changeat, I.P. Waldmann, and G. Tinetti,
"**TauREx III: A fast, dynamic, and extendable framework for retrievals**,"
arXiv:1912.07759 (2019).

TauREx requires molecular opacities or cross-sections from sources like
`ExoMol <https://www.exomol.com/data/search/>`_ or
`ExoTransmit <https://github.com/elizakempton/Exo_Transmit/tree/master/Opac>`_.
We have pre-downloaded some of these molecules; others can be downloaded with
``multirex.Util.get_gases()``.

What’s New
----------

For a detailed list of the latest features, see
`What's new <https://github.com/D4san/MultiREx-public/blob/master/WHATSNEW.md>`__.
