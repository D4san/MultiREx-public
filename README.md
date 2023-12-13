# MultiREx
## Planetary spectra generator


<!-- This are visual tags that you may add to your package at the beginning with useful information on your package --> 
[![version](https://img.shields.io/pypi/v/multirex?color=blue)](https://pypi.org/project/multirex/)
[![downloads](https://img.shields.io/pypi/dw/multirex)](https://pypi.org/project/multirex/)


## Download and install

Describe here how the package can be downloaded and install it in
different arquitectures.

If you are using `PyPI` installation it's as simple as:

```
pip install multirex
```


## Descripción
MultiREx es una biblioteca en Python diseñada para la generación de espectros sintéticos de exoplanetas. Esta herramienta extiende las funcionalidades de la librería Taurex, reorganizando y permitiendo la generación masiva de espectros y observaciones con ruido.

## Taurex
MultiREx se basa en las capacidades de cálculo de espectros y la estructura base de [Taurex](https://taurex3-public.readthedocs.io/en/latest/index.html). Si utilizas `MultiREx` en tu investigación o publicaciones, por favor cita a Taurex de la siguiente manera:

TauREx III: A fast, dynamic and extendable framework for retrievals
A. F. Al-Refaie, Q. Changeat, I.P. Waldmann, y G. Tinetti
ApJ, presentado en 2019

Por este motivo es necesario cargar las opaciedades o secciones eficaces de las moléculas a utilizar en los formatos que TauREx utiliza, los cuales pueden ser obtenidos en :
- [ExoMol](https://www.exomol.com/data/search/).
- [ExoTransmit](https://github.com/elizakempton/Exo_Transmit/tree/master/Opac).



## Características Principales de MultiREx

- **Ensamblaje de Sistemas Planetarios**: Facilita la combinación de diferentes planetas, estrellas y atmósferas para explorar una variedad de configuraciones de sistemas estelares.
- **Atmósferas Personalizables**: Permite la adición y configuración de composiciones atmosféricas variadas para los planetas.
- **Generación de Espectros Sintéticos**: Produce espectros realistas basados en los atributos y condiciones de los sistemas planetarios.
- **Simulación de Observaciones Astronómicas**: Incluye `randinstrument` para simular observaciones de espectros con niveles de ruido determinados por la relación señal-ruido (SNR).
- **Función `explore_multiverse`**: Automatiza la generación de múltiples espectros que varían aleatoriamente en parámetros específicos, proporcionando un amplio rango de resultados para análisis.

## Uso Básico
Aquí un ejemplo simple de cómo usar MultiREx para crear un planeta y generar un espectro sintético:

```python
import mutirex as mrex

# Crear un planeta con propiedades específicas
trappis1e = mrex.Planet(albedo=0.3, radius=0.920, mass=0.692)
trappis1e.set_atmosphere(mrex.Atmosphere(temperature=289, base_pressure=1e5,
                                         top_pressure=1e-3,
                                         composition={"H2O": 1e-2, "CO2": 1e-1, "CH4": (1e-10, 1e-1),
                                                      "O3": (1e-10, 1e-1)}, fill_gas="N2"))

# Configurar una estrella
trappist1 = mrex.Star(temperature=2566, radius=0.1192, mass=0.1192)

# Crear un sistema que incluye el planeta y la estrella
systemtrappist1 = mrex.System(trappis1e, trappist1, distance_parsecs=12.42988,
                              planet_distance=0.02925, orbital_period=6.1010,
                              transit_time=0.9293/60/60)


# Uso de explore_multiverse para generar datos
systemtrappist1.explore_multiverse(wn_grid=wn_grid, snr=3, path=path, n_iter=10,
                                   labels=[["O3"],["CH4"]], header=True, n_observations=1,
                                   spectrum=True, observations=True)
```

En este ejemplo, explore_multiverse se utiliza para generar 10 conjuntos diferentes de espectros y observaciones del sistema systemtrappist1, variando las condiciones de cada conjunto y guardando los resultados en el directorio especificado.


## What's new

If your package will be frequently updated with new features include a
section describing the new features added to it:

Version 0.1.*:

- First version of the package.

------------

This package has been designed and written by David Duque-Castaño and Jorge I. Zuluaga (C) 2023