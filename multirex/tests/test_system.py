import numpy as np
import pytest
from multirex.spectra import Planet, Atmosphere, Star, System, Physics

def create_sample_system():
    # Helper function to create a system with fixed parameters
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    planet = Planet(seed=42, radius=1, mass=1, atmosphere=atm)
    star = Star(seed=42, temperature=5800, radius=1, mass=1)
    system = System(planet, star, seed=42, sma=1.0)
    return system

def test_system_make_tm():
    system = create_sample_system()
    # Call make_tm() and verify that the transmission model is generated
    system.make_tm()
    assert system.transmission is not None

def test_generate_spectrum():
    system = create_sample_system()
    system.make_tm()
    wn_grid = Physics.wavenumber_grid(1, 10, 100)
    bin_wn, bin_rprs = system.generate_spectrum(wn_grid)
    assert isinstance(bin_wn, np.ndarray)
    assert isinstance(bin_rprs, np.ndarray)

def create_sample_system_ggchem():
    # Helper function to create a system with GGChem
    ggchem_params_test = {
        'metallicity': 1.0,
        'selected_elements': ['C', 'O', 'H', 'N'],
        'ratio_elements': ['C'],
        'abundance_profile': 'solar', # Changed from 'equilibrium' to 'solar' as per example
        'ratios_to_O': [0.1], # Changed from {} to [0.1]
        'equilibrium_condensation': True # Added this parameter
    }
    atm = Atmosphere(seed=100, temperature=1200, base_pressure=1e5, top_pressure=1e0,
                     chemistry_type='ggchem', ggchem_params=ggchem_params_test)
    planet = Planet(seed=101, radius=1.2, mass=0.8, atmosphere=atm)
    star = Star(seed=102, temperature=5500, radius=0.9, mass=0.9)
    system = System(planet, star, seed=103, sma=0.05)
    return system

def test_system_make_tm_with_ggchem():
    """Test that make_tm() runs successfully with GGChem."""
    system = create_sample_system_ggchem()
    # Call make_tm() and verify that the transmission model is generated
    system.make_tm()
    assert system.transmission is not None
    # Check if the chemistry object in Taurex is indeed GGChem
    # This requires Taurex model to be built and accessible
    # For now, we assume if make_tm() runs without error with 'ggchem', it's a good sign.
    # A more specific check would be: isinstance(system.transmission.chemistry, GGChem)
    # but this depends on how Taurex structures its model and if GGChem is directly accessible.

def test_generate_spectrum_with_ggchem():
    """Test that a spectrum can be generated with GGChem."""
    system = create_sample_system_ggchem()
    system.make_tm()
    assert system.transmission is not None, "Transmission model should be created before generating spectrum."
    wn_grid = Physics.wavenumber_grid(min_wn=300, max_wn=3000, npoints=50)
    bin_wn, bin_rprs = system.generate_spectrum(wn_grid)
    assert isinstance(bin_wn, np.ndarray), "Wavenumber grid should be a numpy array."
    assert len(bin_wn) > 0, "Wavenumber grid should not be empty."
    assert isinstance(bin_rprs, np.ndarray), "R/Rs array should be a numpy array."
    assert len(bin_rprs) > 0, "R/Rs array should not be empty."
    assert len(bin_wn) == len(bin_rprs), "Wavenumber and R/Rs arrays must have the same length."
    assert not np.isnan(bin_rprs).any(), "Spectrum should not contain NaN values."
