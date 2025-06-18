#########################################
#  __  __      _ _   _ ___ ___          #
# |  \/  |_  _| | |_(_) _ \ __|_ __     #
# | |\/| | || | |  _| |   / _|\ \ /     #
# |_|  |_|\_,_|_|\__|_|_|_\___/_\_\     #
# Planetary spectra generator           #
#########################################

"""
MultiREx: A Python library for generating synthetic exoplanet transmission spectra.

This module provides classes and functions for creating planetary systems,
generating synthetic spectra, and analyzing the results. It extends the
functionalities of the TauREx library, enabling the massive generation of
spectra and observations with added noise.

The main classes in this module are:
    - Physics: Utility functions for spectrum generation and manipulation
    - Planet: Represents a planet with physical properties and atmosphere
    - Atmosphere: Defines atmospheric properties and composition
    - Star: Represents a star with physical properties
    - System: Combines a planet and star to generate transmission spectra
    - Multiverse: Generates multiple spectra with random parameter variations
"""

#########################################
# EXTERNAL PACKAGES
#########################################
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from tqdm import tqdm
import itertools
import copy
from joblib import Parallel, delayed

import taurex.log
from taurex.binning import FluxBinner, SimpleBinner
from taurex.cache import OpacityCache, CIACache
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.contributions import AbsorptionContribution, RayleighContribution
from taurex.model import TransmissionModel
from taurex.planet import Planet as tauP
from taurex.stellar import PhoenixStar, BlackbodyStar
from taurex.temperature import Isothermal
from taurex_ggchem import GGChem # Import GGChem

import multirex.utils as Util

from astropy.constants import M_jup, M_earth, R_jup, R_earth, R_sun, M_sun

#########################################
# LOAD DATA
#########################################
# Predefine the opacity path with the data included in the package
taurex.log.disableLogging()
OpacityCache().clear_cache()
xsec_path = os.path.join(os.path.dirname(__file__), 'data')
OpacityCache().set_opacity_path(xsec_path)

#########################################
# MAIN CLASSES
#########################################
class Physics:    
    def wavenumber_grid(wl_min, wl_max, resolution):
        """Generate a wave number grid from a wavelength range and resolution.
        
        This function converts a wavelength range (in microns) to a wavenumber grid (in cm^-1).
        The conversion uses the formula: wavenumber = 10000/wavelength, where wavelength is in microns
        and wavenumber is in cm^-1.

        Args:
            wl_min (float): 
                Minimum wavelength in microns.
            
            wl_max (float): 
                Maximum wavelength in microns.
            
            resolution (int): 
                Number of points in the resulting grid.
        
        Returns:
            wn (np.array): 
                Wave number grid in cm^-1, sorted in ascending order.

        Notes:
            To convert back from wavenumber (cm^-1) to wavelength:

            >>> wl = 10000/wn  # in microns

            Or to get wavelength in meters:

            >>> wl = 10000/(wn*1e6)  # in meters
        """
        return np.sort(10000/np.logspace(np.log10(wl_min),np.log10(wl_max),resolution))

    def generate_value(value):
        """Generate a value based on the input type.
        
        This utility function handles different input types to generate values:
        
                - If given a single value, returns that value
                - If given a tuple range (min, max), returns a random value in that range
                - If given a list, returns a random choice from the list
                - If given None, returns None
        
        Args:
            value: The input value which can be:
                None: Returns None
                tuple (min, max): Returns a random value between min and max
                list: Returns a random element from the list
                Any other type: Returns the value unchanged
                
        Returns:
            The generated value based on the input type
            
        Examples:
            >>> Physics.generate_value(5)
            5
            >>> Physics.generate_value((1, 10))  # Returns random value between 1 and 10
            7.3546
            >>> Physics.generate_value(['red', 'green', 'blue'])  # Returns random element
            'green'
            >>> Physics.generate_value(None)
            None
        """
        if value is None:
            return None
        elif (isinstance(value, tuple) and
            len(value) == 2):        
            return np.random.uniform(value[0], value[1])
        elif isinstance(value, list):
            return np.random.choice(value)
        else:
            return value
            
    def generate_parameter_space_values(value):
        """Generate a sequence of values for parameter space exploration.
        
        This utility function handles different input types to generate a sequence of values:
        
        - If given a single value, returns a list with just that value
        - If given a tuple range (min, max), returns a list with a random value in that range
        - If given a list, returns the list unchanged
        - If given a dict with keys 'min', 'max', 'n', and optionally 'distribution',
          returns a sequence of n values between min and max with the specified distribution
        - If given None, returns None
        
        Args:
            value: The input value which can be:
                None: Returns None
                tuple (min, max): Returns a list with a random value between min and max
                list: Returns the list unchanged
                dict: With keys:
                    'min': Minimum value
                    'max': Maximum value
                    'n': Number of points
                    'distribution': 'linear' or 'log' (default: 'linear')
                Any other type: Returns a list with just that value
                
        Returns:
            list: A list of values based on the input type
            
        Examples:
            >>> Physics.generate_parameter_space_values(5)
            [5]
            >>> Physics.generate_parameter_space_values((1, 10))  # Returns random value between 1 and 10
            [7.3546]
            >>> Physics.generate_parameter_space_values([1, 2, 3])
            [1, 2, 3]
            >>> Physics.generate_parameter_space_values({'min': -10, 'max': -1, 'n': 10, 'distribution': 'linear'})
            [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
            >>> Physics.generate_parameter_space_values({'min': 100, 'max': 1000, 'n': 4, 'distribution': 'log'})
            [100, 215.44, 464.16, 1000]
        """
        if value is None:
            return None
        elif isinstance(value, dict) and 'min' in value and 'max' in value and 'n' in value:
            min_val = value['min']
            max_val = value['max']
            n_points = value['n']
            distribution = value.get('distribution', 'linear')
            
            if distribution.lower() == 'log':
                if min_val <= 0 or max_val <= 0:
                    raise ValueError("Log distribution requires positive values")
                return list(np.logspace(np.log10(min_val), np.log10(max_val), n_points))
            else:  # linear distribution
                return list(np.linspace(min_val, max_val, n_points))
        elif isinstance(value, tuple) and len(value) == 2:
            return [np.random.uniform(value[0], value[1])]
        elif isinstance(value, list):
            return value
        else:
            return [value]
        
    def generate_df_SNR_noise(df, n_repeat, SNR, seed=None):
        """
        Generates a new DataFrame by applying Gaussian noise in a
        vectorized manner to the spectra, and then concatenates this
        result with another DataFrame containing other columns of information.

        Args:
            df (DataFrame): DataFrame with parameters and spectra. It must have attributes 'params' and 'data'.
                Example: df.params, df.data
            n_repeat (int): How many times each spectrum is replicated.
            SNR (float): Signal-to-noise ratio for each observation.
            seed (int, optional): Seed for the random number generator. Default is None.

        Returns:
            DataFrame: New DataFrame with parameters and spectra with noise added in
                the same format as the input DataFrame. The returned DataFrame
                has the attributes df.params and df.data.
        """
        if not hasattr(df, "params"):
            print("Warning: 'params' attribute not found in the DataFrame.")
            df_params = pd.DataFrame()
            if not hasattr(df, "data"):
                print("Warning: 'data' attribute not found in the DataFrame.", 
                    "The DataFrame will be considered as having 'data' attribute.")
                df_spectra = df
        else:
            if not hasattr(df, "data"):
                raise ValueError("The DataFrame must have a 'data' attribute.")
            else:
                df_params = df.params
                df_spectra = df.data

        if not isinstance(df_spectra, pd.DataFrame):
            raise ValueError("df_spectra must be a pandas DataFrame.")
        if not isinstance(df_params, pd.DataFrame):
            raise ValueError("df_params must be a pandas DataFrame.")
        if (not isinstance(n_repeat, int) or
            n_repeat <= 0):
            raise ValueError("n_repeat must be a positive integer.")
        if (not isinstance(SNR, (int, float)) or
            SNR <= 0):
            raise ValueError("SNR must be a positive number.")
        if (seed is not None and
            (not isinstance(seed, int) or
                seed < 0)):
            raise ValueError("seed must be a non-negative integer.")

        if seed is not None:
            np.random.seed(seed)  
        
        # Replicate the spectra DataFrame according to the replication factor
        df_spectra_replicated = pd.DataFrame(
            np.repeat(df_spectra.values, n_repeat, axis=0),
            columns=df_spectra.columns
            )
        
        # Calculate the signal and noise for each spectrum and replicate it
        signal_max = df_spectra.max(axis=1)
        signal_min = df_spectra.min(axis=1)
        signal_diff = signal_max - signal_min
        noise_per_spectra = signal_diff / SNR 
        noise_replicated = np.repeat(
            noise_per_spectra.values[:, np.newaxis],
            n_repeat,
            axis=0
            )
        
        # apply Gaussian noise vectorized
        gaussian_noise = np.random.normal(
            0, noise_replicated, df_spectra_replicated.shape
            )
        
        df_spectra_replicated += gaussian_noise

        # Replicate the DataFrame of other parameters to match the number
        # of rows of df_spectra_replicated
        
        df_other_columns_replicated = pd.DataFrame(
            np.repeat(df_params.values,n_repeat, axis=0),
            columns=df_params.columns
            )

        df_other_columns_replicated.insert(0, 'noise', noise_replicated.flatten())
        df_other_columns_replicated.insert(1, 'SNR', SNR)
        
        df_final = pd.concat(
            [df_other_columns_replicated.reset_index(drop=True),
            df_spectra_replicated.reset_index(drop=True)],
            axis=1
            )
        
        warnings.filterwarnings("ignore")
        df_final.data = df_final.iloc[:, -df_spectra_replicated.shape[1]:]
        df_final.params = df_final.iloc[:, :df_other_columns_replicated.shape[1]]
        warnings.filterwarnings("default")
        return df_final

    def spectrum2altitude(spectrum, Rp, Rs):
        """Converts the transit depth to the atmospheric effective altitude.

        Args:
            spectrum (float): Transit depth.
            Rp (float): Planet radius in Earth radii.
            Rs (float): Star radius in solar radii.
        
        Returns:
            float: Atmospheric effective altitude in km.
        """
        effalts = (np.sqrt(spectrum)*Rs*R_sun.value - Rp*R_earth.value)/1e3
        return effalts

    def df2spectra(observation):
        """Convert observations dataframe to spectra
        """
        wls = np.array(observation.columns[2:],dtype=float)
        spectra = np.array(observation.iloc[:,2:])
        noise = np.array(observation['noise'])
        return noise, wls, spectra

# For legacy code compatibility
wavenumber_grid = Physics.wavenumber_grid
generate_value = Physics.generate_value
generate_parameter_space_values = Physics.generate_parameter_space_values
generate_df_SNR_noise = Physics.generate_df_SNR_noise

class Atmosphere:
    """Represents a plane parallel atmosphere with specified properties and composition.
    
    This class allows you to define an atmosphere with properties like temperature
    and pressure, as well as its chemical composition. The composition can be specified
    manually molecule by molecule or by using an equilibrium chemistry model like GGChem.
    
    Attributes:
        seed (int): Random seed for reproducibility.
        temperature (float): Temperature of the atmosphere in Kelvin.
        base_pressure (float): Base (bottom) pressure of the atmosphere in Pa.
        top_pressure (float): Top pressure of the atmosphere in Pa.
        composition (dict): Composition of the atmosphere with gases and their
            mixing ratios in log10 values (e.g., {"H2O": -3, "CO2": -2}).
            Only used if chemistry_type is 'manual'.
        fill_gas (str or list): Gas or list of gases used as filler in the
            atmosphere composition to ensure the total mixing ratio equals 1.
            Only used if chemistry_type is 'manual'.
        chemistry_type (str): Type of chemistry model to use ('manual' or 'ggchem').
        ggchem_params (dict): Parameters for GGChem if chemistry_type is 'ggchem'.
        original_params (dict): The original parameters used to initialize the
            atmosphere, including any ranges specified for random generation.
    
    Note:
        The mixing ratios in the composition dictionary are in log10 scale.
        For example, a value of -3 corresponds to a mixing ratio of 10^-3 = 0.001.
    """
    """Represents a plane parallel atmosphere with specified properties and composition.
    
    This class allows you to define an atmosphere with properties like temperature
    and pressure, as well as its chemical composition. The composition is specified
    as a dictionary of gases with their mixing ratios in log10 values. The class
    supports both fixed values and random generation from ranges.
    
    Attributes:
        seed (int): Random seed for reproducibility.
        temperature (float): Temperature of the atmosphere in Kelvin.
        base_pressure (float): Base (bottom) pressure of the atmosphere in Pa.
        top_pressure (float): Top pressure of the atmosphere in Pa.
        composition (dict): Composition of the atmosphere with gases and their
            mixing ratios in log10 values (e.g., {"H2O": -3, "CO2": -2}).
        fill_gas (str or list): Gas or list of gases used as filler in the
            atmosphere composition to ensure the total mixing ratio equals 1.
        original_params (dict): The original parameters used to initialize the
            atmosphere, including any ranges specified for random generation.
    
    Note:
        The mixing ratios in the composition dictionary are in log10 scale.
        For example, a value of -3 corresponds to a mixing ratio of 10^-3 = 0.001.
    """
    def __init__(self, seed=None, temperature=None, 
                 base_pressure=None, top_pressure=None, 
                 composition=None, fill_gas=None,
                 chemistry_type='manual', ggchem_params=None):
        """Initialize an Atmosphere object.
        
        Args:
            seed (int, optional): Random seed for reproducibility. If None, current time is used.
            temperature (float or tuple, optional): Temperature of the atmosphere in Kelvin.
                Can be a single value or a range (min, max) for random generation.
            base_pressure (float or tuple, optional): Base pressure of the atmosphere in Pa.
                Can be a single value or a range (min, max) for random generation.
            top_pressure (float or tuple, optional): Top pressure of the atmosphere in Pa.
                Can be a single value or a range (min, max) for random generation.
            composition (dict, optional): Composition of the atmosphere with gases and
                their mixing ratios in log10 values. For example: {"H2O": -3, "CO2": [-2,-1]}
                where values can be fixed or ranges for random generation.
                Used only if chemistry_type is 'manual'.
            fill_gas (str or list, optional): Gas or list of gases used as filler in the
                atmosphere composition to ensure the total mixing ratio equals 1.
                Used only if chemistry_type is 'manual'.
            chemistry_type (str, optional): Type of chemistry model to use. 
                Defaults to 'manual'. Can be 'ggchem'.
            ggchem_params (dict, optional): Parameters for GGChem if chemistry_type is 'ggchem'.
                Example: {'metallicity': 1.0, 'selected_elements': ['C','O','H','N'], ...}
        
        Note:
            The base_pressure must be greater than top_pressure, as base refers to
            the bottom of the atmosphere (higher pressure) and top refers to the
            upper boundary (lower pressure).
        """
        self._original_params = dict(
            seed = seed,
            temperature = temperature,
            base_pressure = base_pressure,
            top_pressure = top_pressure,
            composition=  composition if composition is not None else dict(),
            fill_gas = fill_gas,
            chemistry_type = chemistry_type, # New attribute
            ggchem_params = ggchem_params # New attribute
        )

        self._seed = seed if seed is not None else int(time.time())
        np.random.seed(self._seed)
        
        # Initialize attributes with None to avoid validation errors during initialization
        self._temperature = None
        self._base_pressure = None
        self._top_pressure = None
        self._fill_gas = fill_gas
        self._chemistry_type = chemistry_type # New attribute
        self._ggchem_params = ggchem_params if ggchem_params is not None else {} # New attribute, ensure it's a dict
        
        # Use setter methods to properly initialize with validation
        if temperature is not None:
            self.set_temperature(temperature)
        if base_pressure is not None:
            self.set_base_pressure(base_pressure)
        if top_pressure is not None:
            self.set_top_pressure(top_pressure)
        if self.chemistry_type == 'manual':
            if composition is not None:
                self.set_composition(composition)
            else:
                self._composition = dict()
        elif self.chemistry_type == 'ggchem':
            self._composition = {} # Manual composition not used with GGChem
            if composition is not None:
                warnings.warn("Manual 'composition' provided but chemistry_type is 'ggchem'. Manual composition will be ignored.")
            if fill_gas is not None:
                 warnings.warn("Manual 'fill_gas' provided but chemistry_type is 'ggchem'. Manual fill_gas will be ignored.")
            
    @property
    def original_params(self):
        return self._original_params

    @property
    def seed(self):
        return self._seed
    
    def set_seed(self, value):
        """Sets the seed used for randomness."""
        self._seed = value
        self._original_params["seed"] = value
        np.random.seed(value)
    
    @property
    def temperature(self):
        return self._temperature

    def set_temperature(self, value):     
        """
        Sets the temperature of the atmosphere, as an isothermal profile.
        Parameters:
        value (float or tuple): Temperature of the atmosphere in K (single value or range).
        """   
        #validations
        if (isinstance(value, tuple) and
            len(value) == 2):
            if (value[0] < 0 or
                value[1] < 0):
                raise ValueError("Temperature values must be positive")
        elif (isinstance(value, (int, float)) and
                value < 0):
            raise ValueError("Temperature value must be positive.")
        
        self._temperature = generate_value(value)
        self._original_params["temperature"] = value

    @property
    def base_pressure(self):
        """
        :noindex:
        """
        return self._base_pressure

    def set_base_pressure(self, value):
        """
        Sets the base pressure of the atmosphere.
        Parameters:
        value (float or tuple): Base pressure of the atmosphere in Pa (single value or range).
        """
        #validations
        if (isinstance(value, tuple) and
            len(value) == 2):
            if (value[0] < 0 or
                value[1] < 0):
                raise ValueError("Base pressure values must be positive")
        elif (isinstance(value, (int, float)) and
              value < 0):
            raise ValueError("Base pressure value must be positive.")
            # validate if top pressure is smaller than base pressure
        
        self._base_pressure = generate_value(value)
        
        if (self._top_pressure is not None):
            if self._base_pressure <= self._top_pressure:
                raise ValueError("Base pressure must be greater than top pressure.")
        
        self._original_params["base_pressure"] = value

    @property
    def top_pressure(self):
        return self._top_pressure

    def set_top_pressure(self, value):        
        """
        Sets the top pressure of the atmosphere.
        Parameters:
        value (float or tuple): Top pressure of the atmosphere in Pa (single value or range).
        """
        # validations 
        if (isinstance(value, tuple) and
            len(value) == 2):
            if (value[0] < 0 or
                value[1] < 0):
                raise ValueError("Top pressure values must be positive")
        elif (isinstance(value, (int, float))
              and value < 0):
            raise ValueError("Top pressure value must be positive.")        
                
        self._top_pressure = generate_value(value)
        
        if (self._base_pressure is not None):
            if self._top_pressure >= self._base_pressure:
                raise ValueError("Top pressure must be smaller than base pressure.")
        
        self._original_params["top_pressure"] = value

    @property
    def composition(self):
        return self._composition

    def set_composition(self, gases):
        """
        Sets the composition of the atmosphere.
        Parameters:
        gases (dict): Composition of the atmosphere with gases and mix ratios in log10 values. 
        (eg.{"H2O":  -3, "CO2": [-2,-1]})
        """
        # If using GGChem, manual composition setting is ignored or handled differently.
        if self.chemistry_type == 'ggchem':
            warnings.warn("Composition is managed by GGChem when chemistry_type is 'ggchem'. Manual composition setting will be ignored.")
            self._composition = {}
            self._original_params["composition"] = {}
            return

        self._composition = dict()
        for gas, mix_ratio in gases.items():
            self.add_gas(gas, mix_ratio)
        self.validate_composition()

    @property
    def fill_gas(self):
        return self._fill_gas

    def set_fill_gas(self, gas):
        """
        Sets the filler gas of the atmosphere.
        Parameters:
        gas (str or list): Gas or list of gases used
        as filler in the atmosphere composition.
        """
        self._fill_gas = gas
        self._original_params["fill_gas"] = gas

    def add_gas(self, gas, mix_ratio):
        """
        Adds a gas to the atmosphere composition with a log10 mix ratio.
        If the gas already exists, its value is updated.
        Parameters:
        gas (str): Gas name.
        mix_ratio (float or tuple): Mix ratio of the gas in log10.
        """
        # If using GGChem, manual gas addition is not applicable.
        if self.chemistry_type == 'ggchem':
            warnings.warn("Cannot add gas manually when chemistry_type is 'ggchem'.")
            return

        if gas in self._composition:
            old_value = self._composition[gas]
            print((
                f"{gas} already exists in the composition. "
                f"Its old value was {old_value}. "
                f"It will be updated to {mix_ratio}."
                ))
        
        # Handle log10 values by converting to actual mixing ratios
        value = generate_value(mix_ratio)
        self._composition[gas] = value
            
        self._original_params["composition"][gas] = mix_ratio
        self.validate_composition()

    def remove_gas(self, gas):
        """
        Removes a gas from the atmosphere composition.
        Parameters:
        gas (str): Gas name.
        """
        if gas not in self._composition:
            print((
                f"{gas} does not exist in the composition. "
                f"No action will be taken."
                ))
            return
        del self._composition[gas]
        del self._original_params["composition"][gas]
        self.validate_composition()
        
    def validate_composition(self):
        """
        Validates that the sum of gas mix ratios in the atmosphere composition does not exceed 1.
        Also checks if the maximum possible values from ranges could exceed 1 and issues a warning.
        """
        # If using GGChem, this validation might not be applicable or needs adjustment.
        if self.chemistry_type == 'ggchem':
            # GGChem handles its own internal consistency for elemental abundances.
            # Manual mix ratio validation is skipped.
            return

        # Convert log values to actual mixing ratios for validation
        actual_mix_ratios = [10**value for value in self._composition.values()]
        total_mix_ratio = sum(actual_mix_ratios)
        
        if (total_mix_ratio > 1 or
            total_mix_ratio < 0):
            raise ValueError((f"The sum of mix ratios must be between 0 and 1."
                            f" Actual value: {total_mix_ratio}"))
        
        # Check if the maximum possible values from ranges could exceed 1
        max_possible_values = []
        for gas, mix_ratio in self._original_params["composition"].items():
            if isinstance(mix_ratio, tuple) and len(mix_ratio) == 2:
                # Get the maximum value from the range
                max_possible_values.append(10**max(mix_ratio))
            elif isinstance(mix_ratio, (int, float)):
                max_possible_values.append(10**mix_ratio)
        
        if max_possible_values:
            max_total = sum(max_possible_values)
            if max_total > 1:
                warnings.warn(f"The maximum possible sum of mix ratios from\
                     ranges could exceed 1. Max possible sum: {max_total:.6f}")

    def get_params(self):
        """Returns the current parameters of the atmosphere.
        
        Returns:
            dict: A dictionary containing the atmosphere's parameters including temperature,
                base_pressure, top_pressure, composition, fill_gas, and seed.
        """
        return dict(
            temperature = self._temperature,
            base_pressure = self._base_pressure,
            top_pressure = self._top_pressure,
            composition = self._composition,
            fill_gas = self._fill_gas,
            seed = self._seed,
            chemistry_type = self._chemistry_type, # New attribute
            ggchem_params = self._ggchem_params # New attribute
        )

    def reshuffle(self):
        """
        Regenerates the atmosphere based on original values or range of values.
        """
        self._seed = self._original_params.get("seed", int(time.time()))
        np.random.seed(self._seed)
        self.set_temperature(self._original_params["temperature"])
        self.set_base_pressure(self._original_params["base_pressure"])
        self.set_top_pressure(self._original_params["top_pressure"])
        self.set_chemistry_type(self._original_params.get("chemistry_type", 'manual')) # New attribute
        self.set_ggchem_params(self._original_params.get("ggchem_params", {})) # New attribute
        
        if self.chemistry_type == 'manual':
            self.set_composition(self._original_params.get("composition", {}))
            self.set_fill_gas(self._original_params.get("fill_gas"))
        elif self.chemistry_type == 'ggchem':
            # For GGChem, composition and fill_gas are managed by GGChem itself.
            # Clear manual composition and fill_gas to avoid conflicts.
            self._composition = {}
            self._original_params['composition'] = {}
            self._fill_gas = None 
            self._original_params['fill_gas'] = None
            # GGChem parameters are set via ggchem_params, no further action here unless they need randomization.
        
    def validate(self):
        """
        Validates the atmosphere's essential properties are defined.
        If chemistry_type is 'manual', fill_gas is also required.
        If chemistry_type is 'ggchem', fill_gas is not required.
        """
        essential_attrs = [
            '_temperature', '_base_pressure', 
            '_top_pressure'
            ]        
        # Add _fill_gas to essential_attrs only if chemistry_type is 'manual'
        if self.chemistry_type == 'manual':
            essential_attrs.append('_fill_gas')
            
        missing_attrs = [
            attr for attr in essential_attrs 
            if getattr(self, attr) is None
            ]
        if missing_attrs:
            # Provide more specific feedback if fill_gas is missing for manual chemistry
            if '_fill_gas' in missing_attrs and self.chemistry_type == 'manual':
                print("Atmosphere Missing attributes: fill_gas is required when chemistry_type is 'manual'.")
            else:
                print("Atmosphere Missing attributes:",
                    [attr[1:] for attr in missing_attrs if attr != '_fill_gas' or self.chemistry_type == 'manual'])
            return False

        #valid ranges for temperature, base_pressure, and top_pressure
        if not all([
            (isinstance(self._temperature, (int, float))
                and self._temperature > 0),
            (isinstance(self._base_pressure, (int, float))
                and self._base_pressure > 0),
            (isinstance(self._top_pressure, (int, float))
                and self._top_pressure > 0),
            self._base_pressure > self._top_pressure
            ]):
            print("Atmosphere has invalid attribute values.")
            return False
        return True

    def __getstate__(self):
        """
        Return the state of the object for pickling.

        Returns:
            dict: The state dictionary of the Planet object.
        """
        # Copy the object's __dict__ (all attributes) into state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Restore the state of the object from the unpickled state.

        Args:
            state (dict): The state dictionary to restore.
        """
        self.__dict__.update(state)

    @property
    def chemistry_type(self):
        """Get the type of chemistry model used ('manual' or 'ggchem')."""
        return self._chemistry_type

    def set_chemistry_type(self, value):
        """Set the type of chemistry model.
        
        Args:
            value (str): Chemistry type, e.g., 'manual' or 'ggchem'.
        """
        if value not in ['manual', 'ggchem']:
            raise ValueError("chemistry_type must be 'manual' or 'ggchem'")
        self._chemistry_type = value
        self._original_params["chemistry_type"] = value

    @property
    def ggchem_params(self):
        """Get the parameters for GGChem (if chemistry_type is 'ggchem')."""
        return self._ggchem_params

    def set_ggchem_params(self, value):
        """Set the parameters for GGChem.
        
        Args:
            value (dict): Dictionary of parameters for GGChem.
        """
        if not isinstance(value, dict) and value is not None:
            raise ValueError("ggchem_params must be a dictionary or None")
        self._ggchem_params = value if value is not None else {}
        self._original_params["ggchem_params"] = value


class Planet:
    """Represents a planet with specified properties and an optional atmosphere.
    
    This class allows you to define a planet with physical properties like radius
    and mass, and optionally attach an atmosphere with specific composition.
    The class supports both fixed values and random generation from ranges.
    
    Attributes:
        seed (int): Random seed for reproducibility.
        radius (float): Radius of the planet in Earth radii.
        mass (float): Mass of the planet in Earth masses.
        atmosphere (Atmosphere): An Atmosphere object defining the planet's atmosphere.
        original_params (dict): The original parameters used to initialize the planet,
            including any ranges specified for random generation.
    """

    def __init__(self, seed=None, radius=None, mass=None, atmosphere=None):
        """Initialize a Planet object.
        
        Args:
            seed (int, optional): Random seed for reproducibility. If None, current time is used.
            radius (float or tuple, optional): Radius of the planet in Earth radii.
                Can be a single value or a range (min, max) for random generation.
            mass (float or tuple, optional): Mass of the planet in Earth masses.
                Can be a single value or a range (min, max) for random generation.
            atmosphere (Atmosphere, optional): An Atmosphere object defining the planet's atmosphere.
                If None, the planet will have no atmosphere until one is set.
        """
        self._original_params = dict(
            seed=seed, radius=radius, mass=mass
        ) 
        self._seed = seed if seed is not None else int(time.time())
        np.random.seed(self._seed)

        self._radius = generate_value(radius)
        self._mass = generate_value(mass)
        
        if atmosphere is not None:
            self.set_atmosphere(atmosphere)
        else:
            self._atmosphere = None     

    @property
    def original_params(self):
        return self._original_params
        
    @property
    def radius(self):
        return self._radius
    
    def set_radius(self, value):
        """
        Sets the radius of the planet.
        Parameters:
        value (float or tuple): Radius of the planet in Earth radii (single value or range).
        """
        # validations
        if (isinstance(value, tuple) and
            len(value) == 2):
            if (value[0] < 0 or
                value[1] < 0):
                raise ValueError("Radius values must be positive")
        elif (isinstance(value, (int, float)) and
              value < 0):
            raise ValueError("Radius value must be positive.")
        
        self._radius = generate_value(value)
        self._original_params["radius"] = value

    @property
    def mass(self):
        return self._mass

    def set_mass(self, value):
        """
        Define the mass of the planet.
        Parameters:
        value (float or tuple): Mass of the planet in Earth masses (single value or range).
        """
        # validations
        if (isinstance(value, tuple) and
            len(value) == 2):
            if (value[0] < 0 or
                value[1] < 0):
                raise ValueError("Mass values must be positive")
        elif (isinstance(value, (int, float)) and
              value < 0):
            raise ValueError("Mass value must be positive.")
        
        self._mass = generate_value(value)
        self._original_params["mass"] = value

    @property
    def seed(self):
        return self._seed

    def set_seed(self, value):
        """Sets the seed used for randomness."""
        self._seed = value
        self._original_params["seed"] = value
        np.random.seed(value)

    @property
    def atmosphere(self):
        return self._atmosphere

    def set_atmosphere(self, value):
        """
        Define the atmosphere of the planet.
        
        Parameters:
        value (Atmosphere): An Atmosphere multirex object.
        """        
        # validate value is an Atmosphere object of multirex
        if value is not None and not isinstance(value, Atmosphere):
            raise ValueError("Atmosphere must be an Atmosphere object.")
        self._atmosphere = value

    def validate(self):
        """
        Validates that all essential attributes of the planet are defined.

        Returns:
        bool: True if all attributes are defined, False otherwise.
        """
        essential_attrs = ['_radius', '_mass', '_atmosphere']
        missing_attrs = [
            attr for attr in essential_attrs
            if getattr(self, attr) is None
            ]
        
        if missing_attrs:
            print("Planet Missing attributes:",
                 [attr[1:] for attr in missing_attrs])
            return False
        if (self._atmosphere is not None and
            not self._atmosphere.validate()):      
            return False
        return True
        
    def get_params(self):
        """Gets the current parameters of the planet and its atmosphere.
        
        Returns:
            dict: A dictionary of the planet's parameters and its atmosphere's parameters.
        """
        params = dict(
            p_radius = self._radius,
            p_mass = self._mass,
            p_seed = self._seed
        )
        if self.atmosphere is not None:
            params.update(
                {("atm "+i): self.atmosphere.get_params()[i] 
                 for i in self.atmosphere.get_params()}
            )
            #remove composition and add as individual parameters
            params.pop("atm composition")
            params.update(
                {("atm "+i): self.atmosphere.get_params()["composition"][i]
                 for i in self.atmosphere.get_params()["composition"]}
            )
        return params

    def reshuffle(self, atmosphere=False):
        """
        Regenerates the planet's attributes using the original values and optionally updates the atmosphere, excluding albedo.
        """
        self._seed = self._original_params.get("seed", int(time.time()))
        np.random.seed(self._seed)
        self.set_radius(self._original_params["radius"])
        self.set_mass(self._original_params["mass"])
        if atmosphere and self._atmosphere:
            self._atmosphere.reshuffle()

    def __getstate__(self):
        """
        Return the state of the object for pickling.

        Returns:
            dict: The state dictionary of the Planet object.
        """
        # Copy the object's __dict__ (all attributes) into state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Restore the state of the object from the unpickled state.

        Args:
            state (dict): The state dictionary to restore.
        """
        self.__dict__.update(state)


class Star:
    """Represents a star with specified properties.
    
    This class allows you to define a star with physical properties like temperature,
    radius, and mass. The class supports both fixed values and random generation from ranges.
    It can use either a blackbody model or the more sophisticated Phoenix stellar model.
    
    Attributes:
        seed (int): Random seed for reproducibility.
        temperature (float): Temperature of the star in Kelvin.
        radius (float): Radius of the star in solar radii.
        mass (float): Mass of the star in solar masses.
        phoenix (bool): Whether the star uses the Phoenix stellar model (True) or
            a simple blackbody model (False).
        phoenix_path (str, optional): Path to the Phoenix model files. This parameter automates 
            the management of Phoenix model files. Providing a path that lacks a 'Phoenix' folder 
            prompts the automatic download of necessary model files into a newly created 'Phoenix' 
            folder at the specified path. An empty string ("") uses the current working directory.
        original_params (dict): The original parameters used to initialize the star,
            including any ranges specified for random generation.
    
    Note:
        When using the Phoenix stellar model, the appropriate model files will be
        automatically downloaded if they don't exist at the specified path.
    """
    def __init__(self, seed=None, temperature=None,
                 radius=None, mass=None, phoenix_path=None):
        """Initialize a Star object.
        
        Args:
            seed (int, optional): Random seed for reproducibility. If None, current time is used.
            temperature (float or tuple, optional): Temperature of the star in Kelvin.
                Can be a single value or a range (min, max) for random generation.
            radius (float or tuple, optional): Radius of the star in solar radii.
                Can be a single value or a range (min, max) for random generation.
            mass (float or tuple, optional): Mass of the star in solar masses.
                Can be a single value or a range (min, max) for random generation.
            phoenix_path (str, optional): Path to the Phoenix model files. If provided,
                the star will use the Phoenix stellar model instead of a blackbody model.
                If the path doesn't contain Phoenix model files, they will be automatically
                downloaded. An empty string uses the current working directory.
        """
        self._original_params = dict(
            seed=seed,
            temperature=temperature,
            radius=radius,
            mass=mass
        )
        
        self._seed = seed if seed is not None else int(time.time())
        np.random.seed(self._seed)

        self._temperature = generate_value(temperature)
        self._radius = generate_value(radius)
        self._mass = generate_value(mass)
        
        if phoenix_path is not None:
            phoenix_path= Util.get_stellar_phoenix(phoenix_path)
            self.phoenix_path=phoenix_path
            self.phoenix=True
            self._original_params["phoenix"]=self.phoenix
        else:
            self.phoenix=False
            self._original_params["phoenix"]=self.phoenix
            
        
    @property
    def seed(self):
        return self._seed

    def set_seed(self, value):
        """
        Sets the seed used for randomness and reproducibility.
        Parameters:
            value (int): Seed value.
        """
        self._seed = value
        self._original_params["seed"] = value
        np.random.seed(value)

    @property
    def temperature(self):
        return self._temperature

    def set_temperature(self, value):
        """
        Sets the star's temperature. 
        Parameters:
            value (float or tuple): Temperature in Kelvin.
        """        
        # validation 
        if (isinstance(value, tuple) and
            len(value) == 2):
            if (value[0] < 0 or
                value[1] < 0):
                raise ValueError("Temperature values must be positive")
        elif (isinstance(value, (int, float)) and
              value < 0):
            raise ValueError("Temperature value must be positive.")
        
        self._temperature = generate_value(value)
        self._original_params["temperature"] = value

    @property
    def radius(self):
        return self._radius

    def set_radius(self, value):
        """
        Sets the star's radius. Can be a single value or a range for random generation.
        Parameters:
            value (float or tuple): Radius in solar radii.
        """        
        # validation 
        if (isinstance(value, tuple) and
            len(value) == 2):
            if (value[0] < 0
                or value[1] < 0):
                raise ValueError("Radius values must be positive")
        elif (isinstance(value, (int, float)) and
              value < 0):
            raise ValueError("Radius value must be positive.")
        
        self._radius = generate_value(value)
        self._original_params["radius"] = value

    @property
    def mass(self):
        return self._mass

    def set_mass(self, value):
        """
        Sets the star's mass. Can be a single value or a range for random generation.
        Parameters:
            value (float or tuple): Mass in solar masses.
        """
        # validate     
        if (isinstance(value, tuple) and
            len(value) == 2):
            if (value[0] < 0 or
                value[1] < 0):
                raise ValueError("Mass values must be positive")
        elif (isinstance(value, (int, float)) and
              value < 0):
            raise ValueError("Mass value must be positive.")  
              
        self._mass = generate_value(value)
        self._original_params["mass"] = value

    def get_params(self):
        """Retrieves the current parameters of the star.
        
        Returns:
            dict: A dictionary containing the star's parameters.
        """
        return {
            "s temperature": self._temperature,
            "s radius": self._radius,
            "s mass": self._mass,
            "s seed": self._seed
        }

    def reshuffle(self):
        """
        Regenerates the star's attributes using the original values.
        """
        self.set_seed(self._original_params.get("seed", int(time.time())))
        self.set_temperature(self._original_params["temperature"])
        self.set_radius(self._original_params["radius"])
        self.set_mass(self._original_params["mass"])
        
        
    def validate(self):
        """
        Validates that all essential attributes of the star are defined.

        Returns:
            bool: True if all essential attributes are defined and valid, False otherwise.
        """
        essential_attrs = ['_temperature', '_radius', '_mass']
        missing_attrs = [attr for attr in essential_attrs 
                         if getattr(self, attr) is None]

        if missing_attrs:
            print("Star is missing essential attributes:", [attr[1:] for attr in missing_attrs])
            return False

        return True

    def __getstate__(self):
        """
        Return the state of the object for pickling.

        Returns:
            dict: The state dictionary of the Planet object.
        """
        # Copy the object's __dict__ (all attributes) into state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Restore the state of the object from the unpickled state.

        Args:
            state (dict): The state dictionary to restore.
        """
        self.__dict__.update(state)
    
class System:
    """Represents a planetary system consisting of a planet orbiting a star.
    
    This class combines a Planet and a Star object to create a complete planetary system.
    It provides methods to generate transmission spectra, analyze spectral contributions
    from different atmospheric components, and simulate observations with noise.
    
    Attributes:
        planet (Planet): The planet in the system.
        star (Star): The star in the system.
        sma (float): Semi-major axis of the planet's orbit in AU.
        seed (int): Random seed for reproducibility.
        transmission (TransmissionModel): The TauREx transmission model for the system,
            created after calling make_tm().
        original_params (dict): The original parameters used to initialize the system,
            including any ranges specified for random generation.
    
    Note:
        After creating a System object, you must call make_tm() to generate the
        transmission model before generating spectra or observations.
    """

    def __init__(self, planet, star, seed=None, sma=None):
        """Initialize a System object.
        
        Args:
            planet (Planet): The planet in the system. Must be a valid Planet object.
            star (Star): The star in the system. Must be a valid Star object.
            seed (int, optional): Random seed for reproducibility. If None, current time is used.
            sma (float or tuple, optional): Semi-major axis of the planet's orbit in AU.
                Can be a single value or a range (min, max) for random generation.
                
        Note:
            After creating a System object, you must call make_tm() to generate the
            transmission model before generating spectra or observations.
        """
        self._original_params = dict(
            seed=seed,
            sma=sma
        )
        
        self._seed = seed if seed is not None else int(time.time())
        np.random.seed(self._seed)

        self.set_planet(planet)
        self.set_star(star)            
        self.set_sma(sma)
        
        self._transmission=None     # transmission model
        

    @property
    def original_params(self):
        return self._original_params
    
    @property
    def seed(self):
        return self._seed
    
    def set_seed(self, value):
        """Sets the seed used for randomness."""
        self._seed = value
        self._original_params["seed"] = value
        np.random.seed(value)
        
    @property
    def planet(self):
        return self._planet
    
    def set_planet(self, value):
        """
        Define the planet of the system.
        Parameters:
        value (Planet): A Planet object of multirex.
        """
        # validation
        if (value is not None and
            not isinstance(value, Planet)):
            raise ValueError("Planet must be a Planet object.")
        self._planet = value
        
    @property
    def star(self):
        return self._star
    
    def set_star(self, value):
        """
        Define the star of the system.
        Args:
        value (Star): A Star object of multirex.
        """
        # validate value
        if (value is not None and
            not isinstance(value, Star)):
            raise ValueError("Star must be a Star object.")
        self._star = value
        
    @property
    def sma(self):
        return self._sma
    
    def set_sma(self, value):
        """
        Define the semi-major axis of the planet's orbit.
        Args:
        value (float or tuple): Semi-major axis of the planet's orbit in AU (single value or range).
        """
        # validate value
        if (isinstance(value, tuple)
            and len(value) == 2):
            if (value[0] < 0 or
                value[1] < 0):
                raise ValueError("Semi-major axis values must be positive")
        elif (isinstance(value, (int, float)) and
              value < 0):
            raise ValueError("Semi-major axis value must be positive.")
        
        self._sma = generate_value(value)
        self._original_params["sma"] = value
        
    def get_params(self):
        """Get the current parameters of the system.
        
        Returns:
            dict: 
                A dictionary containing the system's parameters including semi-major axis, seed, and all parameters from the planet and star.
        """
        params = {
            "sma": self._sma,
            "seed": self._seed
        }        
        params.update(self.planet.get_params())
        params.update(self.star.get_params())
        return params
    
    def validate(self):
        """
        Validates that all essential attributes of the system are defined.
        
        Returns:
        bool: True if all essential attributes are defined, False otherwise.
        """
        essential_attrs = ['_sma']
        missing_attrs = [attr for attr in essential_attrs 
                         if getattr(self, attr) is None]
        if missing_attrs:
            print("System is missing essential attributes:",
                  [attr[1:] for attr in missing_attrs])
            return False
        
        #validate planet and star
        
        if not self._planet.validate():
            print("System configuration error: The planet configuration is invalid.")
            return False
        if not self._star.validate():
            print("System configuration error: The star configuration is invalid.")
            return False
        return True

    def reshuffle(self):
        """
        Regenerates the system's attributes using the original values.
        """
        self._seed = self._original_params.get("seed",
                                               int(time.time()))
                 
        np.random.seed(self._seed)
        self.set_sma(self.original_params["sma"])
        self.planet.reshuffle(atmosphere=True)
        self.star.reshuffle()

    def make_tm(self):
        """Generate a transmission model for the system.
        
        This method creates a TauREx transmission model based on the properties of the
        planet, star, and atmosphere. It is a necessary step before generating any spectra
        or observations. If you make any changes to the system properties, you must call
        this method again to update the transmission model.
        
        The method configures:
        - The planet's physical properties
        - The star's properties (using Phoenix model if specified)
        - The atmosphere's temperature profile (isothermal)
        - The atmosphere's chemistry based on the composition
        - Contributions from absorption and Rayleigh scattering
        
        Returns:
            None: The transmission model is stored internally and can be accessed
                through the transmission property.
                
        Raises:
            ValueError: If the system configuration is invalid (e.g., missing essential
                attributes or invalid parameter values).
        """
        
        #check if the system is valid
        if not self.validate():
            print("System is not valid. A transmission model cannot be generated.")
            return
                
        #convert mass and radius to jupiter and earth units
        rconv= R_jup.value/R_earth.value
        mconv= M_jup.value/M_earth.value
        
        # Taurex planet
        tauplanet=tauP(planet_distance=self.sma,
                    planet_mass=self.planet.mass / mconv,
                    planet_radius=self.planet.radius / rconv,
                    )
                
        #Taurex star        
        if self.star.phoenix:
            taustar=PhoenixStar(temperature=self.star.temperature,
                            radius=self.star.radius,
                            mass=self.star.mass,
                            phoenix_path=self.star.phoenix_path)
        else:
            taustar=BlackbodyStar(temperature=self.star.temperature,
                            radius=self.star.radius,
                            mass=self.star.mass)        
        
        # Taurex temperature model
        tautemperature=Isothermal(T=self.planet.atmosphere.temperature)
        
        ## Taurex chemistry
        atmosphere = self.planet.atmosphere
        if atmosphere.chemistry_type == 'ggchem':
            if not atmosphere.ggchem_params:
                raise ValueError("ggchem_params must be provided in Atmosphere when chemistry_type is 'ggchem'.")
            # Ensure ggchem_params is a dictionary before unpacking
            current_ggchem_params = atmosphere.ggchem_params.copy() if isinstance(atmosphere.ggchem_params, dict) else {}
            
            # Here you could add logic to generate random values for ggchem_params if they are defined as ranges
            # For example, if current_ggchem_params['metallicity'] = (0.5, 1.5)
            # then: current_ggchem_params['metallicity'] = Physics.generate_value(current_ggchem_params['metallicity'])
            # This requires Physics.generate_value to be compatible or a new helper function.
            # For now, assuming ggchem_params contains direct values.
            
            # Validate required GGChem parameters (example)
            required_gg_params = ['metallicity', 'selected_elements', 'ratio_elements', 'abundance_profile', 'ratios_to_O']
            for req_param in required_gg_params:
                if req_param not in current_ggchem_params:
                    #warnings.warn(f"Required GGChem parameter '{req_param}' not found in ggchem_params. Using default if available or may error.")
                    pass # GGChem might have defaults, or raise its own error. Better to let GGChem handle this.
            
            try:
                tauchem = GGChem(**current_ggchem_params)
            except TypeError as e:
                raise ValueError(f"Error initializing GGChem with parameters {current_ggchem_params}: {e}. Ensure all parameters are valid for GGChem.")
        
        elif atmosphere.chemistry_type == 'manual':
            tauchem = TaurexChemistry(fill_gases=atmosphere.fill_gas)
            if atmosphere.composition:
                for gas, mix_ratio_log10 in atmosphere.composition.items():
                    # Convert log10 mixing ratio to actual value for TauREx
                    actual_mix_ratio = 10**mix_ratio_log10
                    tauchem.addGas(ConstantGas(molecule_name=gas,
                                                mix_ratio=actual_mix_ratio))
            elif not atmosphere.fill_gas:
                # If no composition and no fill_gas, TaurexChemistry might be empty or default to something.
                warnings.warn("Manual chemistry selected with no composition and no fill_gas. TaurexChemistry might be empty or use defaults.")
        else:
            raise ValueError(f"Unknown chemistry_type: {atmosphere.chemistry_type}. Must be 'manual' or 'ggchem'.")
        
        ## Transmission model
        tm = TransmissionModel(
            planet=tauplanet,
            temperature_profile=tautemperature,
            chemistry=tauchem,
            star=taustar,
            atm_max_pressure=self.planet.atmosphere.base_pressure,
            atm_min_pressure=self.planet.atmosphere.top_pressure)
        tm.add_contribution(AbsorptionContribution())
        tm.add_contribution(RayleighContribution())
        tm.build()
        
        self._transmission=tm
        
        ## OFF 
        #load the zscale in km
        #self._zscale= self.transmission.altitude_profile*1e-3
        
    @property
    def transmission(self):
        """ Get the transmission model of the system."""
        return self._transmission
    
    
    def generate_spectrum(self, wn_grid):
        """Generate a transmission spectrum based on a wave number grid.
        
        This method uses the system's transmission model to generate a synthetic
        spectrum at the specified wave numbers. The transmission model must be
        created first by calling make_tm().
        
        Args:
            wn_grid (numpy.ndarray): Wave number grid in cm^-1. Can be created using
            the Physics.wavenumber_grid() method.
        
        Returns:
            tuple: A tuple containing:            
                bin_wn (numpy.ndarray): Binned wave number grid in cm^-1.
                bin_rprs (numpy.ndarray): Binned spectrum in (Rp/Rs)^2 units,
                representing the transit depth at each wavelength.
                
        Raises:
            ValueError: If no transmission model has been generated. Call make_tm()
                before using this method.
                
        Examples:
            >>> system = System(planet, star, sma=1.0)
            >>> system.make_tm()
            >>> wn_grid = Physics.wavenumber_grid(1.0, 10.0, 1000)
            >>> wn, spectrum = system.generate_spectrum(wn_grid)
        """
        
        #validate the transmission model
        if self._transmission is None:
            print("A transmission model has not been generated.")
            return
                
        # Create a binner
        bn = FluxBinner(wngrid=wn_grid)
        # Generate the spectrum
        bin_wn, bin_rprs, _, _ = bn.bin_model(
            self.transmission.model(wngrid=wn_grid))
                
        return bin_wn, bin_rprs
    
    def generate_contributions(self, wn_grid):
        """
        Generate a differentiated spectrum contribution based on a wave number grid.
        
        Args:
            wn_grid (array): Wave number grid.
        
        Returns:
            tuple: A tuple containing:
                - bin_wn (array): Wave number grid.
                - bin_rprs (dict): Fluxes in rp^2/rs^2 per contribution and molecule.
        """
        
        #validate the transmission model
        if self._transmission is None:
            print("A transmission model has not been generated.")
            return
        
        # Create a binner
        bn = FluxBinner(wngrid=wn_grid)
        
        # Generate the full spectrum
        self.transmission.model(wngrid=wn_grid)
        model = self.transmission.model_full_contrib(wngrid=wn_grid)
        
        bin_rprs = {}
        for aporte in model[1]:
            bin_rprs[aporte] = {}
            for j in range(len(model[1][aporte])):
                chem = [model[1][aporte][j][i] for i in range(1, 4)]
                contrib = [model[0], chem[0], chem[1], chem[2]]
                bin_wn, bin_rprs[aporte][model[1][aporte][j][0]], _, _ \
                    = bn.bin_model(contrib)               
        
        return bin_wn, bin_rprs 
       
    def generate_observations(self, wn_grid, snr, n_observations=1):
        """Generate simulated observations with noise based on the system's spectrum.
        
        This method generates synthetic observations by adding gaussian noise to the
        system's transmission spectrum. The noise level is determined by the specified
        signal-to-noise ratio (SNR). Multiple observations can be generated at once.
        
        Args:
            wn_grid (numpy.ndarray): Wave number grid in cm^-1, defining the wavelengths
                at which the observations are made. Can be created using the
                Physics.wavenumber_grid() method.
            snr (float): Signal-to-noise ratio, used to determine the level of noise
                added to the observations. Higher values result in less noise.
            n_observations (int, optional): Number of noisy observations to generate.
                Defaults to 1.
        
        Returns:
            pandas.DataFrame: DataFrame containing the simulated observations with added noise.
            The DataFrame has the following structure:

                - Columns labeled with wavelengths (from wn_grid) containing the fluxes
                  in (Rp/Rs)^2 units with added noise.
                - 'SNR' column indicating the signal-to-noise ratio used.
                - 'noise' column showing the noise level added to each observation.
                
                The DataFrame also has two special attributes:
                - df.params: Contains the system parameters and noise information.
                - df.data: Contains only the spectral data (wavelength columns).
        
        Raises:
            ValueError: If no transmission model has been generated. Call make_tm()
                before using this method.
                
        Examples:
            >>> system = System(planet, star, sma=1.0)
            >>> system.make_tm()
            >>> wn_grid = Physics.wavenumber_grid(1.0, 10.0, 1000)
            >>> observations = system.generate_observations(wn_grid, snr=10, n_observations=5)
        """
        
        
        # Validate the transmission model
        if self._transmission is None:
            print("A transmission model has not been generated.")
            return
        self.make_tm()
        
        # Generate the spectrum dataframe
        bin_wn,bin_rprs=self.generate_spectrum(wn_grid)
        columns = list(10000 / np.array(bin_wn))
        bin_rprs_reshaped = bin_rprs.reshape(1, -1)
        spec_df = pd.DataFrame(bin_rprs_reshaped, columns=columns)
        
        # Generate dataframe with noisy observations
        observations = generate_df_SNR_noise(spec_df, n_observations, snr)  
        
        return observations

    # plots 
    def plot_spectrum(self,  wn_grid, showfig=True, xscale='linear', syslegend=True):
        """
        Plot the spectrum.
        
        Args:
            wn_grid (array): Wave number grid (in cm-1).
            showfig (bool, optional): Whether to show the plot. Defaults to True.
            xscale (str, optional): Scale for x-axis ('linear' or 'log'). Defaults to 'linear'.
            syslegend (bool, optional): Whether to show system legend. Defaults to True.
        
        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure): Figure of the plot.
                - ax (matplotlib.axes): Axes of the plot.
        """                     
        wns, spectrum = self.generate_spectrum(wn_grid)
        wls = 1e4/wns

        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = ax.twinx()
        
        ax.plot(wls, spectrum*1e6)
        ax2.plot(wls, 
                 Physics.spectrum2altitude(
                     spectrum,
                     self.planet.radius,
                     self.star.radius),
                alpha=0)
        
        ax.set_xlabel("Wavelength [μm]")
        ax.set_ylabel("Transit depth [ppm]")
        ax2.set_ylabel("Effective altitude [km]")
        ax2.tick_params(axis='y')

        if xscale == "log":
            ax.set_xscale('log')
            from matplotlib.ticker import FuncFormatter
            formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
            ax.xaxis.set_major_formatter(formatter)
            formatter = FuncFormatter(lambda y, _: '{:.1g}'.format(y))
            ax.xaxis.set_minor_formatter(formatter)
            ax.grid(axis='x', which='minor', ls='--')
            ax.grid(axis='x', which='major')
            ax.grid(axis='y', which='major')
        else:
            ax.grid()

        ax.margins(x=0)
    
        if syslegend:
            text = ax.text(0.01,0.98,self.__str__(),fontsize=8,
                verticalalignment='top',transform=ax.transAxes)
            text.set_bbox(dict(facecolor='w', 
                            alpha=1, 
                            edgecolor='w',
                            boxstyle='round,pad=0.1'))
            
        if showfig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax

    ## plot contributions
    def plot_contributions(self, wn_grid, showfig=True, showspectrum=True, xscale='linear', syslegend=True):
        """
        Plot the spectrum for each contribution and molecule.
        
        Args:
            wn_grid (array): Wave number grid (in cm-1).
            showfig (bool, optional): Whether to show the plot. Defaults to True.
            showspectrum (bool, optional): Whether to show the total spectrum. Defaults to True.
            xscale (str, optional): Scale for x-axis ('linear' or 'log'). Defaults to 'linear'.
            syslegend (bool, optional): Whether to show system legend. Defaults to True.
        
        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure): Figure of the plot.
                - ax (matplotlib.axes): Axes of the plot.
        """
        wns, contributions =self.generate_contributions(wn_grid)
        wls = 1e4/wns
                   
        fig, ax = plt.subplots(figsize=(10, 5))

        # Twin axis showing the scale-height
        ax2 = ax.twinx()
        ax2.set_ylabel("Effective altitude [km]")
        ax2.tick_params(axis='y')
    
        for aporte in contributions:
            for mol in contributions[aporte]:
                ax.plot(wls,
                        contributions[aporte][mol]*1e6,
                        label=aporte+": "+mol,
                        )
                ax2.plot(wls,
                         Physics.spectrum2altitude(
                             contributions[aporte][mol],
                             self.planet.radius,self.star.radius
                            ),
                         color='c',
                         alpha=0)
                
        ax.set_xlabel("Wavelength [μm]")
        ax.set_ylabel("Transit depth [ppm]")
        
        # add other y axis in the right with the zscale
        if showspectrum:
            ax.plot(wls, 
                    self.generate_spectrum(wn_grid)[1]*1e6,
                    label="Total Spectrum",
                    color="black",
                    alpha=0.5,
                    ls="--",
                    )
                
        ax.legend(loc='upper right')

        if xscale == "log":
            ax.set_xscale('log')
            from matplotlib.ticker import FuncFormatter
            formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
            ax.xaxis.set_major_formatter(formatter)
            formatter = FuncFormatter(lambda y, _: '{:.1g}'.format(y))
            ax.xaxis.set_minor_formatter(formatter)
            ax.grid(axis='x', which='minor', ls='--')
            ax.grid(axis='x', which='major')
            ax.grid(axis='y', which='major')
        else:
            ax.grid()

        ax.margins(x=0)
    
        if syslegend:
            text = ax.text(0.01,0.98,self.__str__(),fontsize=8,
                verticalalignment='top',transform=ax.transAxes)
            text.set_bbox(dict(facecolor='w', 
                            alpha=1, 
                            edgecolor='w',
                            boxstyle='round,pad=0.1'))
        
        if showfig:
            plt.show()
        else:
            plt.close(fig)
  
        return fig, ax

    def plot_mixing_ratio(self, list_gases=None, showfig=True,  min_mix=None):
        """
        Plot mixing ratio profiles.

        Args:
            list_gases (list of str, optional): Subset of active gases to plot.
                If None, plot all gases in self.transmission.chemistry.activeGases.
            showfig (bool): Whether to call plt.show() after plotting.
            min_mix (float, optional): Minimum mixing ratio (in linear units) to set as lower bound on the X-axis (log scale).
                If None, X-axis lower bound is chosen automatically.

        Returns:
            fig (matplotlib.figure.Figure): The created figure.
            ax (matplotlib.axes.Axes): The primary axes object.

        """
        import warnings
        import matplotlib.pyplot as plt

        # Ensure a transmission model exists
        if self._transmission is None:
            raise ValueError("Transmission model not generated. Call make_tm() first.")
        chem = self._transmission.chemistry
        active = chem.activeGases             # list of gas names
        mix_profiles = chem.activeGasMixProfile  # array of shape (n_gases, n_levels)

        # Determine which gases to plot (using indices)
        if list_gases is None:
            selected = list(range(len(active)))
        else:
            selected = []
            for i, gas in enumerate(active):
                if gas in list_gases:
                    selected.append(i)
            for gas in list_gases:
                if gas not in active:
                    warnings.warn(f"Gas '{gas}' not active; skipping.")
        if not selected:
            raise ValueError("No valid gases to plot.")

        # Extract profiles: pressure [Pa] and altitude [km]
        P = self._transmission.pressureProfile

        # Create figure and primary axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot mixing ratio for each selected gas
        for idx in selected:
            gas = active[idx]
            mix = mix_profiles[idx]
            ax.plot(mix, P, label=gas)

        # Format primary axis: mixing ratio and pressure
        ax.set_xlabel("Mixing Ratio (log scale)")
        ax.set_xscale('log')
        ax.set_xlim(right = 1)
        if min_mix is not None:
            ax.set_xlim(left=min_mix)
        ax.set_ylabel("Pressure [Pa]")
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        # Show or close figure
        if showfig:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def clone_shuffled(self):
        """
        Creates a new System instance using the original initialization parameters,
        which will regenerate (reshuffle) the random values.
        
        Returns:
            System: A freshly initialized System instance.
        """
        # Para el planeta, se usan los parámetros originales (original_params)
        cloned_atmosphere = None
        if self.planet.atmosphere is not None:
            orig_atm = self.planet.atmosphere.original_params
            cloned_atmosphere = Atmosphere(
                seed=orig_atm["seed"],
                temperature=orig_atm["temperature"],
                base_pressure=orig_atm["base_pressure"],
                top_pressure=orig_atm["top_pressure"],
                composition=orig_atm["composition"],
                fill_gas=orig_atm["fill_gas"]
            )
        cloned_planet = Planet(
            seed=self.planet._original_params["seed"],
            radius=self.planet._original_params["radius"],
            mass=self.planet._original_params["mass"],
            atmosphere=cloned_atmosphere
        )
        cloned_star = Star(
            seed=self.star._original_params["seed"],
            temperature=self.star._original_params["temperature"],
            radius=self.star._original_params["radius"],
            mass=self.star._original_params["mass"],
            phoenix_path=self.star.phoenix_path if hasattr(self.star, 'phoenix_path') else None
        )
        return System(cloned_planet, cloned_star, seed=self._seed, sma=self._sma)

    def explore_multiverse(self, wn_grid, snr=10, n_universes=1, labels=None, header=False,
                       n_observations=1, spectra=True, observations=True, path=None, n_jobs=1):
        """
        Explore the multiverse by generating spectra and observations, and optionally save them
        in Parquet format.

        Args:
            wn_grid (array): Wave number grid.
            snr (float, optional): Signal-to-noise ratio. Defaults to 10.
            n_universes (int, optional): Number of universes to explore. One planet per universe
                is generated with properties drawn from the priors. Defaults to 1.
            labels (list, optional): Labels for atmospheric composition. Example: [["CO2", "CH4"], "CH4"].
                Defaults to None.
            header (bool, optional): Whether to include header information (system parameters) in the output.
                Defaults to False.
            n_observations (int, optional): Number of observations to generate per spectrum.
                Defaults to 1.
            spectra (bool, optional): Whether to save the spectra. Defaults to True.
            observations (bool, optional): Whether to save the observations. Defaults to True.
            path (str, optional): Path to save the files. If not provided, files are not saved.
            n_jobs (int, optional): Number of parallel jobs to run. Defaults to 1 (sequential execution).
                Use -1 to utilize all available cores.

        Returns:
            dict: Dictionary containing 'spectra' and/or 'observations' DataFrames depending on the arguments.
                - spectra (DataFrame): Spectra of the universes.
                - observations (DataFrame): Observations of the universes.

        Example:
            >>> system = System(planet, star, sma=1.0)
            >>> results = system.explore_multiverse(wn_grid, snr=10, n_universes=5, header=True)
        """
        # Validate the transmission model
        if self._transmission is None:
            raise ValueError("A transmission model has not been generated.")
        
        if not any([spectra, observations]):
            raise ValueError("At least one of 'spectra' or 'observations' must be True.")
        
        def process_universe(i):
            """
            Process a single universe.

            This function clones the current system (using the clone() method),
            generates the transmission model, extracts the spectrum, and prepares the header
            with the system parameters.

            Args:
                i (int): Index of the universe (not used internally).

            Returns:
                tuple: A tuple containing:
                    - header (dict): System parameters (if header is True).
                    - spec_df (DataFrame): The generated spectrum as a DataFrame.
            """
            # Clone the system to have an independent instance
            system_copy = self.clone_shuffled()
            system_copy.make_tm()
            bin_wn, bin_rprs = system_copy.generate_spectrum(wn_grid)
            columns = list(10000 / np.array(bin_wn))
            spec_df = pd.DataFrame(bin_rprs.reshape(1, -1), columns=columns)
            
            current_header = {}
            if header:
                current_header = system_copy.get_params()
            if labels is not None:
                valid_labels = []
                for label in labels:
                    if isinstance(label, str) and label in system_copy.transmission.chemistry.gases:
                        valid_labels.append(label)
                    elif isinstance(label, list):
                        valid_sublabels = [
                            sublabel for sublabel in label
                            if sublabel in system_copy.transmission.chemistry.gases
                        ]
                        if valid_sublabels:
                            valid_labels.append(valid_sublabels)
                current_header["label"] = valid_labels if valid_labels else []
            return current_header, spec_df

        # Process all universes either sequentially or in parallel
        if n_jobs == 1:
            results = [process_universe(i) for i in range(n_universes)]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_universe)(i) for i in range(n_universes)
            )
        
        # Separate headers and spectra from the results
        header_list = [res[0] for res in results]
        spectra_list = [res[1] for res in results]
        all_spectra_df = pd.concat(spectra_list, axis=0, ignore_index=True)
        all_header_df = pd.DataFrame(header_list)
        
        final_spectra_df = pd.concat([all_header_df, all_spectra_df], axis=1)
        warnings.filterwarnings("ignore")
        final_spectra_df.data = final_spectra_df.iloc[:, -all_spectra_df.shape[1]:]
        final_spectra_df.params = final_spectra_df.iloc[:, :all_header_df.shape[1]]
        warnings.filterwarnings("default")
        
        if observations:
            print(f"Generating observations for {n_universes} spectra...")
            all_observations_df = generate_df_SNR_noise(final_spectra_df, n_observations, snr)
            if path is not None:
                all_observations_df_copy = all_observations_df.copy()
                all_observations_df_copy.columns = all_observations_df_copy.columns.astype(str)
                all_observations_df_copy.to_parquet(f'{path}/multirex_observations.parquet')
            if spectra:
                if path is not None:
                    final_spectra_df_copy = final_spectra_df.copy()
                    final_spectra_df_copy.columns = final_spectra_df_copy.columns.astype(str)
                    final_spectra_df_copy.to_parquet(f'{path}/multirex_spectra.parquet')
                return {"spectra": final_spectra_df, "observations": all_observations_df}
            else:
                return all_observations_df
        else:
            if path is not None:
                final_spectra_df_copy = final_spectra_df.copy()
                final_spectra_df_copy.columns = final_spectra_df_copy.columns.astype(str)
                final_spectra_df_copy.to_parquet(f'{path}/multirex_spectra.parquet')
            return final_spectra_df

        

    def clone_frozen(self):
        """
        Creates a new System instance with the current state, without reshuffling.
        
        Returns:
            System: A clone of the current System with the same current parameter values.
        """
        # Clone the atmosphere, if present
        cloned_atmosphere = None
        if self.planet.atmosphere is not None:
            cloned_atmosphere = Atmosphere(
                seed=self.planet.atmosphere.seed,
                temperature=self.planet.atmosphere.get_params()["temperature"],
                base_pressure=self.planet.atmosphere.get_params()["base_pressure"],
                top_pressure=self.planet.atmosphere.get_params()["top_pressure"],
                composition=self.planet.atmosphere.get_params()["composition"],
                fill_gas=self.planet.atmosphere.fill_gas
            )
        # Clone the planet using its original parameters
        cloned_planet = Planet(
            seed=self.planet.seed,
            radius=self.planet.get_params()["p_radius"],
            mass=self.planet.get_params()["p_mass"],
            atmosphere=cloned_atmosphere
        )
        # Clone the star using its original parameters
                # Clone the star using its original parameters
        cloned_star = Star(
            seed=self.star.seed,
            temperature=self.star.get_params()["s temperature"],
            radius=self.star.get_params()["s radius"],
            mass=self.star.get_params()["s mass"],
        )
        if getattr(self.star, "phoenix", False):
            cloned_star.phoenix = True
            cloned_star.phoenix_path = self.star.phoenix_path
        return System(cloned_planet, cloned_star, seed=self._seed, sma=self._sma)

    def explore_parameter_space(self, wn_grid, parameter_space, snr=10, labels=None,
                                header=False, n_observations=1, spectra=True,
                                observations=True, path=None, n_jobs=1):
        """
        Explore a parameter space by systematically varying parameters across specified
        ranges.

        This method allows for structured parameter space exploration by generating
        spectra for all combinations of parameter values specified in the
        parameter_space dictionary.

        Args:
            wn_grid (array): Wave number grid.
            parameter_space (dict): Dictionary specifying the parameter space to explore.
                Each key should be a parameter path (e.g., 'planet.atmosphere.temperature')
                and each value should be one of:
                    - A single value
                    - A list of values
                    - A dict with keys 'min', 'max', 'n', and optionally 'distribution'
                    ('linear' or 'log')
            snr (float, optional): Signal-to-noise ratio. Defaults to 10.
            labels (list, optional): Labels for atmospheric composition. Example:
                [["CO2", "CH4"], "CH4"]. Defaults to None.
            header (bool, optional): Whether to include header information in the saved
                files. Defaults to False.
            n_observations (int, optional): Number of observations to generate.
                Defaults to 1.
            spectra (bool, optional): Whether to save the spectra. Defaults to True.
            observations (bool, optional): Whether to save the observations. Defaults to True.
            path (str, optional): Path to save the files. If not provided, files are not saved.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1,
                meaning sequential execution. Use -1 to utilize all available cores.

        Returns:
            dict: Dictionary containing 'spectra' and/or 'observations' DataFrames
                depending on the arguments.
                - spectra (DataFrame): Spectra of the parameter space exploration.
                - observations (DataFrame): Observations of the parameter space exploration.

        Examples:
            >>> system = System(planet, star, sma=1.0)
            >>> parameter_space = {
            ...     'planet.atmosphere.temperature': {'min': 200, 'max': 400, 'n': 3},
            ...     'planet.atmosphere.composition.CH4': {
            ...         'min': -10, 'max': -1, 'n': 10, 'distribution': 'linear'
            ...     }
            ... }
            >>> wn_grid = Physics.wavenumber_grid(1.0, 10.0, 1000)
            >>> results = system.explore_parameter_space(wn_grid, parameter_space, snr=10)
        """
        # Validate the transmission model
        if self._transmission is None:
            self.make_tm()

        if not any([spectra, observations]):
            raise ValueError("At least one of 'spectra' or 'observations' must be True.")

        # Process parameter space to generate all parameter combinations
        param_values = {}
        for param_path, param_spec in parameter_space.items():
            param_values[param_path] = generate_parameter_space_values(param_spec)
        param_names = list(param_values.keys())
        param_value_lists = [param_values[name] for name in param_names]
        all_combinations = list(itertools.product(*param_value_lists))

        def process_combination(combination):
            """
            Process a single combination of parameter values.

            This function clones the current system using the clone method,
            sets the parameters based on the given combination, generates the transmission
            model and spectrum, and returns the header and the spectrum DataFrame.

            Args:
                combination (tuple): A tuple containing one value per parameter.

            Returns:
                tuple: A tuple containing the header (dict) and the spectrum DataFrame.
            """
            # En lugar de deepcopy, se usa el método clone para crear una nueva instancia.
            system_copy = self.clone_frozen()
            for i, param_path in enumerate(param_names):
                param_value = combination[i]
                path_parts = param_path.split('.')
                current_obj = system_copy
                for j in range(len(path_parts) - 1):
                    if path_parts[j] == 'planet':
                        current_obj = current_obj.planet
                    elif path_parts[j] == 'star':
                        current_obj = current_obj.star
                    elif path_parts[j] == 'atmosphere':
                        current_obj = current_obj.atmosphere
                    elif path_parts[j] == 'composition':
                        # Caso especial: actualizar directamente el diccionario de composición.
                        gas_name = path_parts[j + 1]
                        current_obj.composition[gas_name] = param_value
                        break
                else:
                    # Si no es el caso de composición, se establece el atributo.
                    attr_name = f"_{path_parts[-1]}"
                    setattr(current_obj, attr_name, param_value)
            system_copy.make_tm()
            bin_wn, bin_rprs = system_copy.generate_spectrum(wn_grid)
            columns = list(10000 / np.array(bin_wn))
            spec_df = pd.DataFrame(bin_rprs.reshape(1, -1), columns=columns)
            current_header = system_copy.get_params() if header else {}
            if labels is not None:
                valid_labels = []
                for label in labels:
                    if isinstance(label, str) and label in system_copy.transmission.chemistry.gases:
                        valid_labels.append(label)
                    elif isinstance(label, list):
                        valid_sublabels = [
                            sublabel for sublabel in label
                            if sublabel in system_copy.transmission.chemistry.gases
                        ]
                        if valid_sublabels:
                            valid_labels.append(valid_sublabels)
                current_header["label"] = valid_labels if valid_labels else []
            return current_header, spec_df


        # Process all combinations either sequentially or in parallel
        if n_jobs == 1:
            results = [process_combination(comb) for comb in all_combinations]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_combination)(comb) for comb in all_combinations
            )

        # Separate headers and spectra from the results
        header_list = [res[0] for res in results]
        spectra_list = [res[1] for res in results]
        all_spectra_df = pd.concat(spectra_list, axis=0, ignore_index=True)
        all_header_df = pd.DataFrame(header_list)

        # Concatenate the headers and spectra, and assign special attributes
        final_spectra_df = pd.concat([all_header_df, all_spectra_df], axis=1)
        warnings.filterwarnings("ignore")
        final_spectra_df.data = final_spectra_df.iloc[:, -all_spectra_df.shape[1]:]
        final_spectra_df.params = final_spectra_df.iloc[:, :all_header_df.shape[1]]
        warnings.filterwarnings("default")

        # Generate observations if requested
        if observations:
            print(f"Generating observations for {len(all_combinations)} spectra...")
            all_observations_df = generate_df_SNR_noise(final_spectra_df,
                                                        n_observations, snr)
            if path is not None:
                # Save the observations if a path is provided
                all_observations_df_copy = all_observations_df.copy()
                all_observations_df_copy.columns = (
                    all_observations_df_copy.columns.astype(str)
                )
                all_observations_df_copy.to_parquet(
                    f'{path}/multirex_parameter_space_observations.parquet'
                )
            if spectra:
                if path is not None:
                    final_spectra_df_copy = final_spectra_df.copy()
                    final_spectra_df_copy.columns = final_spectra_df_copy.columns.astype(str)
                    final_spectra_df_copy.to_parquet(
                        f'{path}/multirex_parameter_space_spectra.parquet'
                    )
                return {"spectra": final_spectra_df,
                        "observations": all_observations_df}
            else:
                return all_observations_df
        else:
            if path is not None:
                final_spectra_df_copy = final_spectra_df.copy()
                final_spectra_df_copy.columns = final_spectra_df_copy.columns.astype(str)
                final_spectra_df_copy.to_parquet(
                    f'{path}/multirex_parameter_space_spectra.parquet'
                )
            return final_spectra_df


    def __str__(self):

        composition_str = ""
        for gas, mix_ratio in self.planet.atmosphere.composition.items():
            composition_str += f"{gas}: {1e6*10**mix_ratio:.2g} ppm "

        str = rf"""System:
Star: {self.star.temperature:.1f} K, {self.star.radius:.2f} $R_\odot$, {self.star.mass:.2f} $M_\odot$
Planet: {self.planet.radius:.2f} $R_\oplus$, {self.planet.mass:.2f} $M_\oplus$
Semimajor axis: {self.sma:.2f} au
Atmosphere: {self.planet.atmosphere.temperature:.1f} K, {self.planet.atmosphere.base_pressure:.0f} Pa - {self.planet.atmosphere.top_pressure:.0f} Pa, {self.planet.atmosphere.fill_gas} fill gas
Composition: {composition_str}"""
        return str


    def __getstate__(self):
        """
        Return the state of the object for pickling.

        Returns:
            dict: The state dictionary of the System object, excluding non-picklable attributes.
        """
        state = self.__dict__.copy()
        # Exclude the transmission model since it is generated dynamically.
        if '_transmission' in state:
            del state['_transmission']
        return state


    def __setstate__(self, state):
        """
        Restore the state of the object from the unpickled state.

        Args:
            state (dict): The state dictionary to restore.
        """
        self.__dict__.update(state)