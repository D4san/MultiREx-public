# MultiREx
## Planetary spectra generator


<!-- This are visual tags that you may add to your package at the beginning with useful information on your package --> 
[![version](https://img.shields.io/pypi/v/multirex?color=blue)](https://pypi.org/project/multirex/)
[![downloads](https://img.shields.io/pypi/dw/multirex)](https://pypi.org/project/multirex/)


## Download and Install

```bash
pip install multirex
```

## Description
MultiREx is a Python library designed for the generation of synthetic exoplanet spectra. This tool extends the functionalities of the Taurex library, reorganizing and enabling the massive generation of spectra and observations with noise.

### Taurex
MultiREx builds on the spectral calculation capabilities and the basic structure of [Taurex](https://taurex3-public.readthedocs.io/en/latest/index.html). If you use `MultiREx` in your research or publications, please cite Taurex as follows:

**TauREx III: A fast, dynamic, and extendable framework for retrievals**  
A. F. Al-Refaie, Q. Changeat, I.P. Waldmann, and G. Tinetti  
ApJ, submitted in 2019

It is necessary to load the opacities or cross-sections of the molecules used in the formats that Taurex utilizes, which can be obtained from:
- [ExoMol](https://www.exomol.com/data/search/)
- [ExoTransmit](https://github.com/elizakempton/Exo_Transmit/tree/master/Opac)

> We have pre-downloaded some of these molecules, and others can be downloaded using the command `multirex.Util.get_gases()`

## Key Features of MultiREx

- **Planetary System Assembly**: Facilitates the combination of different planets, stars, and atmospheres to explore various stellar system configurations.
- **Customizable Atmospheres**: Allows the addition and configuration of varied atmospheric compositions for the planets.
- **Synthetic Spectrum Generation**: Produces realistic spectra based on the attributes and conditions of planetary systems.
- **Astronomical Observation Simulation**: Includes `randinstrument` to simulate spectral observations with noise levels determined by the signal-to-noise ratio (SNR).
- **`explore_multiverse` Function**: Automates the generation of multiple spectra that randomly vary in specific parameters, providing a wide range of results for analysis.

## Examples

For usage tutorials, access the `/examples` folder. Specifically, this folder contains a subfolder named `/examples/research`, which demonstrates how MultiREx can be used in research that utilizes the generated data to train Machine Learning models.


## What's new


Version 0.1.*:

- First version of the package.

------------

This package has been designed and written by David Duque-Castaño and Jorge I. Zuluaga (C) 2024
