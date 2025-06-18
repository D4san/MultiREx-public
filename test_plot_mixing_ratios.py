import numpy as np
from multirex.spectra import Planet, Atmosphere, Star, System

# Create a planet with an atmosphere
planet = Planet(
    radius=1.0,  # Earth radii
    mass=1.0,    # Earth masses
    atmosphere=Atmosphere(
        temperature=300,  # K
        base_pressure=1e5,  # Pa
        top_pressure=1e-4,  # Pa
        fill_gas='H2',
        composition={
            'H2': 0.78,  # log10(mixing ratio)
            'He': -0.2,  # log10(mixing ratio)
            'CO2': -3.0, # log10(mixing ratio)
            'CH4': -4.0  # log10(mixing ratio)
        }
    )
)

# Create a star
star = Star(
    temperature=5772,  # K
    radius=1.0,       # Solar radii
    mass=1.0          # Solar masses
)

# Create a system
system = System(
    planet=planet,
    star=star,
    sma=1.0  # AU
)

# Generate the transmission model
system.make_tm()

# Test plotting all gases
print("\nPlotting all gases...")
fig, ax = system.plot_mixing_ratios(showfig=True, syslegend=True)

# Test plotting specific gases
print("\nPlotting selected gases (H2, CO2)...")
fig, ax = system.plot_mixing_ratios(
    list_gases=['H2', 'CO2'],
    showfig=True,
    syslegend=True
)

# Test plotting with invalid gas
print("\nPlotting with invalid gas (N2)...")
fig, ax = system.plot_mixing_ratios(
    list_gases=['H2', 'N2'],  # N2 is not in the composition
    showfig=True,
    syslegend=True
)

# Test without showing figure
print("\nPlotting without showing figure...")
fig, ax = system.plot_mixing_ratios(
    list_gases=['H2', 'CO2'],
    showfig=False,
    syslegend=True
)

print("\nAll tests completed!")
from multirex.spectra import Planet, Atmosphere, Star, System

# Create a planet with an atmosphere
planet = Planet(
    radius=1.0,  # Earth radii
    mass=1.0,    # Earth masses
    atmosphere=Atmosphere(
        temperature=300,  # K
        base_pressure=1e5,  # Pa
        top_pressure=1e-4,  # Pa
        fill_gas='H2',
        composition={
            'H2': 0.78,  # log10(mixing ratio)
            'He': -0.2,  # log10(mixing ratio)
            'CO2': -3.0, # log10(mixing ratio)
            'CH4': -4.0  # log10(mixing ratio)
        }
    )
)

# Create a star
star = Star(
    temperature=5772,  # K
    radius=1.0,       # Solar radii
    mass=1.0          # Solar masses
)

# Create a system
system = System(
    planet=planet,
    star=star,
    sma=1.0  # AU
)

# Generate the transmission model
system.make_tm()

# Test plotting all gases
print("\nPlotting all gases...")
fig, ax = system.plot_mixing_ratios(showfig=True, syslegend=True)

# Test plotting specific gases
print("\nPlotting selected gases (H2, CO2)...")
fig, ax = system.plot_mixing_ratios(
    list_gases=['H2', 'CO2'],
    showfig=True,
    syslegend=True
)

# Test plotting with invalid gas
print("\nPlotting with invalid gas (N2)...")
fig, ax = system.plot_mixing_ratios(
    list_gases=['H2', 'N2'],  # N2 is not in the composition
    showfig=True,
    syslegend=True
)

# Test without showing figure
print("\nPlotting without showing figure...")
fig, ax = system.plot_mixing_ratios(
    list_gases=['H2', 'CO2'],
    showfig=False,
    syslegend=True
)

print("\nAll tests completed!")
