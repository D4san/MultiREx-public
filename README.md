# MultiREx
## Planetary transmission spectra generator

<!-- This are visual tags that you may add to your package at the beginning with useful information on your package --> 
[![version](https://img.shields.io/pypi/v/multirex?color=blue)](https://pypi.org/project/multirex/)
[![downloads](https://img.shields.io/pypi/dw/multirex)](https://pypi.org/project/multirex/)
[![license](https://img.shields.io/pypi/l/multirex)](https://pypi.org/project/multirex/)
[![implementation](https://img.shields.io/pypi/implementation/multirex)](https://pypi.org/project/multirex/)
[![pythonver](https://img.shields.io/pypi/pyversions/multirex)](https://pypi.org/project/multirex/)
<!-- 
[![arXiv](http://img.shields.io/badge/arXiv-2207.08636-orange.svg?style=flat)](http://arxiv.org/abs/2207.08636)
[![ascl](https://img.shields.io/badge/ascl-2205.016-blue.svg?colorB=262255)](https://ascl.net/2205.016)
-->

MultiREx is a Python library designed for generating synthetic exoplanet transmission spectra. This tool extends the functionalities of the `Taurex` library (see below), reorganizing and enabling the massive generation of spectra and observations with added noise. The package was originally devised for training large machine learning models at identifying the presence of biosignatures in noisy spectra. However, it should also be used for other purposes.

For the science behind the model please refer to the following paper:

> David S. Duque-Castaño, Jorge I. Zuluaga, and Lauren Flor-Torres (2024), **Machine-assisted classification of potential biosignatures in earth-like exoplanets using low signal-to-noise ratio transmission spectra**, submitted to MNRAS.
<!--
[Astronomy and Computing 40 (2022) 100623](https://www.sciencedirect.com/science/article/pii/S2213133722000476), [arXiv:2207.08636](https://arxiv.org/abs/2207.08636).
-->

## Downloading and Installing `MultiREx` 

`MultiREx` is available at the `Python` package index and can be installed in Linux using:

```bash
$ sudo pip install multirex
```
as usual this command will install all dependencies and download some useful data, scripts and constants.

> **NOTE**: If you don't have access to `sudo`, you can install `MultiREx` in your local environmen (usually at `~/.local/`). In that case you need to add to your `PATH` environmental variable the location of the local python installation. For that purpose add to the configuration files `~/.bashrc` or `~/.bash_profile`, the line `export PATH=$HOME/.local/bin:$PATH`

If you are a developer or want to work directly with the package sources, clone `MultiREx` from the `GitHub` repository:

```bash
$ git clone https://github.com/D4san/MultiREx-public
```

To install the package from the sources use:

```bash
$ cd MultiREx-public
$ python3 setup.py install
```

## Running `MultiREx` in `GoogleColab`

To run MultiREx in Google Colab you should execute:
```python
!pip install -Uq multirex
```

After installing you should reset session before importing the package. This is to avoid the unintended behavior of the package `pybtex`. After reset you should not reinstall the package, just import it:

```python
import multirex as mrex
```

## Quickstart

To start using `MultiREx` you must import the package:

```python
import multirex as mrex
```

To start with, we need to provide to `MultiREx` the properties of the three components of any transmission model: A star, a planet and a planetary atmosphere.


```python
star=mrex.Star(temperature=5777,radius=1,mass=1)
```

Radius and mass are in solar units.

Now let's create the planet:
```python
planet=mrex.Planet(radius=1,mass=1)
```
Radius and mass are in units of Earth properties. 

Now it's time to give the planet an atmosphere. This is a basic example of an N2 atmosphere having 100 ppm of CO2 and 1 ppm of CH4:

```python
atmosphere=mrex.Atmosphere(
    temperature=288, # in K
    base_pressure=1e5, # in Pa
    top_pressure=1, # in Pa
    fill_gas="N2", # the gas that fills the atmosphere
    composition=dict(
        CO2=-4, # This is the log10(mix-ratio)
        CH4=-6,
    )
)
planet.set_atmosphere(atmosphere)
```

Now we can ensamble the system:

```python
system=mrex.System(star=star,planet=planet,sma=1)
```

Semimajor axis of the planeta (`sma`) is given in au (astronomical units).

We are ready to see some spectrum. For this purpose we need to create a transmission model:

```python
system.make_tm()
```

Once initialized, let's plot the transmission spectrum over a given grid of wavenumbers or wavelengths:

```python
wns = mrex.Physics.wavenumber_grid(wl_min=0.6,wl_max=10,resolution=1000)
fig, ax = system.plot_contributions(wns,xscale='log')
```

<p align="center"><img src="https://github.com/D4san/MultiREx-public/blob/main/examples/resources/contributions-transmission-spectrum.png?raw=true" alt="Contributions in transmission spectra"/></p>

All of these functionalities are also available in `Taurex`. However, the interface to `MultiREx` is much more intuitive and, more importantly, it is also best suited for the real superpower of the package: the capaciy to create large ensamble of random planetary systems. 

For creating a random planetary system starting with a range of the relevant parameters (ranges given between parenthesis), we use the command:

```python
system=mrex.System(
    star=mrex.Star(
        temperature=(4000,6000),
        radius=(0.5,1.5),
        mass=(0.8,1.2),
    ),
    planet=mrex.Planet(
        radius=(0.5,1.5),
        mass=(0.8,1.2),
        atmosphere=mrex.Atmosphere(
            temperature=(290,310), # in K
            base_pressure=(1e5,10e5), # in Pa
            top_pressure=(1,10), # in Pa
            fill_gas="N2", # the gas that fills the atmosphere
            composition=dict(
                CO2=(-5,-4), # This is the log10(mix-ratio)
                CH4=(-6,-5),
            )
        )
    ),
    sma=(0.5,1)
)
```

In this simple example, we assume that all key parameters (stellar mass and radius, planetary mass and radius, surface planetary temperature, semimajor axis, etc.) are physically and statistically independent. This is not true, but it works for testing the basic features of the package.

Using this system as a template we may generate thousands of spectra that can be used, for instance, for training machine learning algorithms. For an in depth explanation of how to use those advanced functionalities of `MultiREx` please check the  [quick start guide](https://github.com/D4san/MultiREx-public/blob/main/examples/multirex-quickstart.ipynb).

In the figure below we show some of the resulting synthetic spectra, along with the corresponding theoretical spectrum corresponding to a particular set of random values for the key system parameters.

<p align="center"><img src="https://github.com/D4san/MultiREx-public/blob/main/examples/resources/synthetic-transmission-spectra.png?raw=true" alt="Synthetic transmission spectra"/></p>

## Further examples

In order to illustrate the basic and advanced functionalities of `MultiREx` we provided with the package repository several example `Jupyter` notebooks (see directory `examples/`). Additionally, all the notebooks used to generate the results and create the figures for our papers are also available in the `GitHub` repo (see directory `examples/papers`). 

## Key features of `MultiREx`

- **Planetary System Assembly**: Facilitates the combination of different planets, stars, and atmospheres to explore various stellar system configurations.

- **Customizable Atmospheres**: Allows the addition and configuration of varied atmospheric compositions for the planets.

- **Synthetic Spectrum Generation**: Produces realistic spectra based on the attributes and conditions of planetary systems.

- **Astronomical Observation Simulation**: Includes `randinstrument` to simulate spectral observations with noise levels determined by the signal-to-noise ratio (SNR).

- **Multiverse analysis**: Automates the generation of multiple spectra that randomly vary in specific parameters, providing a wide range of results for analysis.

### A note about `Taurex`

MultiREx is built on the spectral calculation capabilities and the basic structure of [Taurex](https://taurex3-public.readthedocs.io/en/latest/index.html). If you use `MultiREx` in your research or publications, please also cite Taurex as follows:

> A. F. Al-Refaie, Q. Changeat, I.P. Waldmann, and G. Tinetti **TauREx III: A fast, dynamic, and extendable framework for retrievals**,  arXiv preprint [arXiv:1912.07759 (2019)](https://arxiv.org/abs/1912.07759).

It is necessary to load the opacities or cross-sections of the molecules used in the formats that Taurex utilizes, which can be obtained from:
- [ExoMol](https://www.exomol.com/data/search/)
- [ExoTransmit](https://github.com/elizakempton/Exo_Transmit/tree/master/Opac)

We have pre-downloaded some of these molecules, and others can be downloaded using the command `multirex.Util.get_gases()`

## What's new

For a detailed list of the newest characteristics of the code see the file [What's new](https://github.com/D4san/MultiREx-public/blob/master/WHATSNEW.md).

------------

This package has been designed and written by David Duque-Castaño and Jorge I. Zuluaga (C) 2024