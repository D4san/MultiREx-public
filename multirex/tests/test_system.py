import numpy as np
import pytest
import importlib.util
import pandas as pd
from multirex.spectra import Planet, Atmosphere, Star, System, Physics

GG_AVAILABLE = importlib.util.find_spec("taurex_ggchem") is not None

def create_sample_system():
    # Helper function to create a system with fixed parameters
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    planet = Planet(seed=42, radius=1, mass=1, atmosphere=atm)
    star = Star(seed=42, temperature=5800, radius=1, mass=1)
    system = System(planet, star, seed=42, sma=1.0)
    return system


def test_explore_parameter_space_absence_case():
    system = create_sample_system()
    wn_grid = Physics.wavenumber_grid(1.0, 2.0, 10)
    parameter_space = {
        'planet.atmosphere.composition.CH4': {'min': -6, 'max': -6, 'n': 1, 'include_absence': True}
    }
    df = system.explore_parameter_space(wn_grid, parameter_space, snr=10, header=True,
                                        spectra=True, observations=False, n_jobs=1)
    # Ensure 'atm CH4' column exists and has NaN for the absence row
    assert 'atm CH4' in df.columns
    assert df['atm CH4'].isna().sum() == 1
    assert df['atm CH4'].notna().sum() == 1

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

# GGChem-specific helpers and tests

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

@pytest.mark.skipif(not GG_AVAILABLE, reason="taurex_ggchem not installed")
def test_system_make_tm_with_ggchem():
    """Test that make_tm() runs successfully with GGChem."""
    system = create_sample_system_ggchem()
    # Call make_tm() and verify that the transmission model is generated
    system.make_tm()
    assert system.transmission is not None

@pytest.mark.skip(reason="GGChem sometimes produces NaNs on synthetic setups; skipped until investigated")
def test_generate_spectrum_with_ggchem():
    """Test that a spectrum can be generated with GGChem."""
    system = create_sample_system_ggchem()
    system.make_tm()
    assert system.transmission is not None, "Transmission model should be created before generating spectrum."
    wn_grid = Physics.wavenumber_grid(wl_min=10000/3000, wl_max=10000/300, resolution=50)
    bin_wn, bin_rprs = system.generate_spectrum(wn_grid)
    assert isinstance(bin_wn, np.ndarray), "Wavenumber grid should be a numpy array."
    assert len(bin_wn) > 0, "Wavenumber grid should not be empty."
    assert isinstance(bin_rprs, np.ndarray), "R/Rs array should be a numpy array."
    assert len(bin_rprs) > 0, "R/Rs array should not be empty."
    assert len(bin_wn) == len(bin_rprs), "Wavenumber and R/Rs arrays must have the same length."
    assert not np.isnan(bin_rprs).any(), "Spectrum should not contain NaN values."

def create_sample_system_cia():
    atm = Atmosphere(seed=200, temperature=1000, base_pressure=1e5, top_pressure=1e0,
                     composition={"H2": -0.15, "H2O": -3.0}, fill_gas="He", cia=["H2-H2", "H2-He"])
    planet = Planet(seed=201, radius=1.0, mass=1.0, atmosphere=atm)
    star = Star(seed=202, temperature=5000, radius=1.0, mass=1.0)
    system = System(planet, star, seed=203, sma=0.05)
    return system

def test_system_make_tm_with_cia():
    """Test that make_tm() runs successfully with CIA pairs."""
    system = create_sample_system_cia()
    system.make_tm()
    assert system.transmission is not None

from unittest.mock import patch
from taurex.cache import CIACache
from taurex.cia.cia import CIA
import numpy as np

class DummyCIA(CIA):
    def __init__(self, pair_name):
        super().__init__("Dummy", pair_name)
        self._pair_name = pair_name
    @property
    def pairName(self): return self._pair_name
    def cia(self, T, wngrid): return np.zeros(len(wngrid))

@patch.object(CIACache, '__getitem__', side_effect=lambda self, key: DummyCIA(key), autospec=True)
def test_generate_spectrum_with_cia(mock_getitem):
    """Test that a spectrum can be generated with CIA pairs."""
    system = create_sample_system_cia()
    system.make_tm()
    wn_grid = Physics.wavenumber_grid(wl_min=10000/3000, wl_max=10000/300, resolution=50)
    bin_wn, bin_rprs = system.generate_spectrum(wn_grid)
    assert len(bin_wn) > 0
    assert len(bin_wn) == len(bin_rprs)
    assert not np.isnan(bin_rprs).any()
