import numpy as np
import matplotlib
import pytest
from multirex.spectra import Planet, Atmosphere, Star, System

# Use non-interactive backend for tests
matplotlib.use('Agg')

def test_plot_mixing_ratio():
    planet = Planet(
        radius=1.0,
        mass=1.0,
        atmosphere=Atmosphere(
            temperature=300,
            base_pressure=1e5,
            top_pressure=1e-4,
            fill_gas='H2',
            composition={
                'He': -0.2,
                'CO2': -3.0,
                'CH4': -4.0
            }
        )
    )
    star = Star(temperature=5772, radius=1.0, mass=1.0)
    system = System(planet=planet, star=star, sma=1.0)
    system.make_tm()

    # Generate spectrum to initialize chemistry profiles in TauREx
    from multirex.spectra import Physics
    wns = Physics.wavenumber_grid(wl_min=10000/3000, wl_max=10000/300, resolution=50)
    system.generate_spectrum(wns)

    # Test plotting all gases
    fig, ax = system.plot_mixing_ratio(showfig=False)
    assert fig is not None

    # Test plotting selected gases
    fig, ax = system.plot_mixing_ratio(list_gases=['He', 'CO2'], showfig=False)
    assert fig is not None

    # Test with invalid gas (warnings should happen but not crash)
    fig, ax = system.plot_mixing_ratio(list_gases=['CO2', 'N2'], showfig=False)
    assert fig is not None
