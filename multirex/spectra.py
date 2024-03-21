import json
import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from tqdm import tqdm

import taurex.log
from taurex.binning import FluxBinner, SimpleBinner
from taurex.cache import OpacityCache, CIACache
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.contributions import AbsorptionContribution, RayleighContribution
from taurex.model import TransmissionModel
from taurex.planet import Planet as tauP
from taurex.stellar import PhoenixStar, BlackbodyStar
from taurex.temperature import Isothermal

import multirex.utils as Util

taurex.log.disableLogging()

# Predefine the opacity path with the data included in the package
OpacityCache().clear_cache()
xsec_path = os.path.join(os.path.dirname(__file__), 'data')
OpacityCache().set_opacity_path(xsec_path)


def generate_value(value):
    """
    Generates a value if it is a single value or a random value if it is a range,
    returns None if the value is None.
    """
    if value is None:
        return None
    elif (isinstance(value, tuple) and
          len(value) == 2):        
        return np.random.uniform(value[0], value[1])
    else:
        return value
    

def generate_df_noise(df_spectra, df_params, n_repeat, SNR, seed=None):
    """
    Generates a new DataFrame by applying Gaussian noise in a
    vectorized manner to the spectra, and then concatenates this
    result with another DataFrame containing other columns of information.

    Parameters:
    - df_spectra: DataFrame with the spectra values.
    - df_params: DataFrame with parameters associated with each spectrum.
    - n_repeat: How many times each spectrum is replicated.
    - SNR: Signal-to-noise ratio for each observation.
    - seed: Seed for the random number generator (optional).

    Returns:
    - New DataFrame with noisy observations and the other columns 
        of information.
    """
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
    
    # apply gaussian noise vectorized
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

    # insert SNR replicated dataframe to the spectra dataframe
    df_spectra_replicated.insert(0, 'SNR', SNR)
    # insert noise replicated dataframe to the spectra dataframe
    df_spectra_replicated.insert(1, 'noise', noise_replicated.flatten())
    
    # Concatenar los DataFrames
    df_final = pd.concat(
        [df_other_columns_replicated.reset_index(drop=True),
         df_spectra_replicated.reset_index(drop=True)],
         axis=1
         )

    return df_final


def wavenumber_grid(wl_min, wl_max, resolution):
    """
    Generate a wave number grid from a wavelength range and resolution.
    
    Args:
    - wl_min (float): Minimum wavelength in microns.
    - wl_max (float): Maximum wavelength in microns.
    - resolution (int): Resolution of the wave number grid.    
    
    Returns:
    - np.array: Wave number grid in cm^-1.
    """

    return np.sort(10000 / np.logspace(np.log10(wl_min),
                                       np.log10(wl_max),
                                       resolution))

class Planet:
    """
    Represents a planet with specified properties and an optional atmosphere.

    Attributes:
    - seed (int): Random seed for reproducibility.
    - radius (float or tuple): Radius of the planet in Earth radii (single value or range).
    - mass (float or tuple): Mass of the planet in Earth masses (single value or range).
    - atmosphere (Atmosphere): An Atmosphere object.
    """

    def __init__(self, seed=None, radius=None, mass=None, atmosphere=None):
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
        """
        Gets the current parameters of the planet and its atmosphere.
        
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

    def move_universe(self, atmosphere=False):
        """
        Regenerates the planet's attributes using the original values and optionally updates the atmosphere, excluding albedo.
        """
        self._seed = self._original_params.get("seed", int(time.time()))
        np.random.seed(self._seed)
        self.set_radius(self._original_params["radius"])
        self.set_mass(self._original_params["mass"])
        if atmosphere and self._atmosphere:
            self._atmosphere.move_universe()


class Atmosphere:
    """
    Represents a plane parallel atmosphere with specified properties and composition. 

    Atributes:
    - seed (int): Random seed for reproducibility.
    - temperature (float or tuple): Temperature of the atmosphere (single value or range).
    - base_pressure (float or tuple): Base pressure of the atmosphere in Pa (single value or range).
    - top_pressure (float or tuple): Top pressure of the atmosphere in Pa (single value or range).
    - composition (dict): Composition of the atmosphere with gases and mix ratios in log10 values. (eg.{"H2O":-3, "CO2":[-2,-1]})
    - fill_gas (str): Gas or list of gases used as filler in the atmosphere composition.
    """
    def __init__(self, seed=None, temperature=None, 
                 base_pressure=None, top_pressure=None, 
                 composition=None, fill_gas=None):        
        self._original_params = dict(
            seed = seed,
            temperature = temperature,
            base_pressure = base_pressure,
            top_pressure = top_pressure,
            composition=  composition if composition is not None else dict(),
            fill_gas = fill_gas
        )

        self._seed = seed if seed is not None else int(time.time())
        np.random.seed(self._seed)
        
        self._temperature = generate_value(temperature)
        self._base_pressure = generate_value(base_pressure)
        self._top_pressure = generate_value(top_pressure)
        self._fill_gas = fill_gas
        if composition is not None:
            self.set_composition(composition)
        else:
            self._composition = dict()
            
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
        if (self._top_pressure is not None
            and value < self._top_pressure):
            raise ValueError("Base pressure must be greater than top pressure.")
        
        self._base_pressure = generate_value(value)
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
        if (self._base_pressure is not None
            and value > self._base_pressure):
            raise ValueError("Top pressure must be smaller than base pressure.")
                
        self._top_pressure = generate_value(value)
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
        if gas in self._composition:
            old_value = self._composition[gas]
            print((
                f"{gas} already exists in the composition. "
                f"Its old value was {old_value}. "
                f"It will be updated to {mix_ratio}."
                ))
            
        self._composition[gas] = generate_value(mix_ratio)
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
        Validates that the sum of gas mix ratios in the atmosphere composition, given in log10, does not exceed 1.
        """
        total_mix_ratio = sum(10**mix for mix in self._composition.values())
        
        if (total_mix_ratio > 1 or
            total_mix_ratio < 0):
            raise ValueError((f"The sum of mix ratios must be between 0 and 1."
                              f" Actual value: {total_mix_ratio}"))

    def get_params(self):
        """
        Returns the current parameters of the atmosphere.
        """
        return dict(
            temperature = self._temperature,
            base_pressure = self._base_pressure,
            top_pressure = self._top_pressure,
            composition = self._composition,
            fill_gas = self._fill_gas,
            seed = self._seed
        )

    def move_universe(self):
        """
        Regenerates the atmosphere based on original values or range of values.
        """
        self._seed = self._original_params.get("seed", int(time.time()))
        np.random.seed(self._seed)
        self.set_temperature(self._original_params["temperature"])
        self.set_base_pressure(self._original_params["base_pressure"])
        self.set_top_pressure(self._original_params["top_pressure"])
        self.set_composition(self._original_params.get("composition", {}))
        self.set_fill_gas(self._original_params["fill_gas"])
        
    def validate(self):
        """
        Validates the atmosphere's essential properties are defined, allowing for an undefined composition if fill_gas is present.
        """
        essential_attrs = [
            '_temperature', '_base_pressure', 
            '_top_pressure', '_fill_gas'
            ]        
        missing_attrs = [
            attr for attr in essential_attrs 
            if getattr(self, attr) is None
            ]
        if missing_attrs:
            print("Atmosphere Missing attributes:",
                  [attr[1:] for attr in missing_attrs])
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


class Star:
    """
    Represents a star with specified properties.

    Attributes:
        seed (int): Random seed for reproducibility.
        temperature (float or tuple): Temperature of the star in Kelvin, can be a single value or a range.
        radius (float or tuple): Radius of the star in solar radii, can be a single value or a range.
        mass (float or tuple): Mass of the star in solar masses, can be a single value or a range.
        phoenix_path (str): Path to the Phoenix model files. This parameter automates the management
            of Phoenix model files. Providing a path that lacks a 'Phoenix' folder prompts the automatic
            download of necessary model files into a newly created 'Phoenix' folder at the specified path.
            An empty string ("") uses the current working directory for this purpose. This feature removes
            the need for manual file handling by the user.
    """
    def __init__(self, seed=None, temperature=None,
                 radius=None, mass=None,phoenix_path=None):
        
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
        """
        Retrieves the current parameters of the star.
        Returns:
            dict: A dictionary containing the star's parameters.
        """
        return {
            "s temperature": self._temperature,
            "s radius": self._radius,
            "s mass": self._mass,
            "s seed": self._seed
        }

    def move_universe(self):
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
    
class System:
    """
    System class representing a system consisting of a planet and a star, with the planet 
    orbiting the star.

    Inputs:
    - planet (Planet): A Planet object.
    - star (Star): A Star object.
    - sma (float or tuple): Semi-major axis of the planet's orbit in AU (single value or range).
    - seed (int): Random seed for reproducibility.
    """

    def __init__(self, planet, star,seed=None, sma=None):
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
        """
        Get the current parameters of the system.
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

    def move_universe(self):
        """
        Regenerates the system's attributes using the original values.
        """
        self._seed = self._original_params.get("seed",
                                               int(time.time()))
                 
        np.random.seed(self._seed)
        self.set_sma(self.original_params["sma"])
        self.planet.move_universe(atmosphere=True)
        self.star.move_universe()

    def make_tm(self):
        """
        Generate a transmission model for the system.
        
        It is a necessary step to generate a transmission model
        before generating a spectrum, and if you make a change in the system 
        you need to generate a new transmission model.
        """  
        #check if the system is valid
        if not self.validate():
            print("System is not valid. A transmission model cannot be generated.")
            return
                
        from astropy.constants import M_jup, M_earth, R_jup, R_earth
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
        tauchem=TaurexChemistry(fill_gases=self.planet.atmosphere.fill_gas)
        for gas, mix_ratio in self.planet.atmosphere.composition.items():
            tauchem.addGas(ConstantGas(molecule_name=gas,
                                        mix_ratio=10**mix_ratio))
        
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
        
        #load the zscale in km
        self._zscale= self.transmission.altitude_profile*1e-3
        
    @property
    def transmission(self):
        """ Get the transmission model of the system."""
        return self._transmission
    
    
    def generate_spectrum(self, wn_grid):
        """
        Generate a spectrum based on a wave number grid.

        Parameters:
        - wn_grid (array): Wave number grid.

        Returns:
        - bin_wn (array): Wave number grid.
        - bin_rprs (array): Fluxes in rp^2/rs^2.
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
        generate a differentiated spectrum contribution based on a wave number grid.
        
        Parameters:
        wn_grid (array): Wave number grid.
        
        Returns:
        bin_wn (array): Wave number grid.
        bin_rprs (dict): Fluxes in rp^2/rs^2 per contribution and molecule.
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
       
    def generate_observations(self, wn_grid, snr, n_observations):
        """
        Generate observations with noise based on a wave number grid and save them optionally in a 
        specified format.

        Parameters:
        - wn_grid (array): Wave number grid, defining the wavelengths at which the 
        observations are made.
        - snr (float): Signal-to-noise ratio, used to determine the level of noise
        added to the observations.
        - n_observations (int): Number of noisy observations to generate.    

        Returns:
        DataFrame: Observations with added noise. Includes:
            - Columns labeled with the wavelengths (from `wn_grid`) containing the fluxes 
                in rp^2/rs^2 with added noise.
            - 'SNR' column indicating the signal-to-noise ratio used for each observation.
            - 'noise' column showing the noise level added to the fluxes.
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
        observations = generate_df_noise(spec_df,pd.DataFrame(), snr, n_observations)  
        
        return observations

    # plots 
    def plot_spectrum(self,  wn_grid, showfig=False):
        """
        Plot the spectrum.
        
        Parameters:
        - wn_grid (array): Wave number grid (in cm-1).
        - showfig (bool): Whether to show the plot (optional).
        
        Returns:
        - fig (matplotlib.figure): Figure of the plot.
        - ax (matplotlib.axes): Axes of the plot.

        """                     
        spectrum=self.generate_spectrum(wn_grid)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(10000/spectrum[0], spectrum[1])
        ax.set_xlabel("Wavelength [μm]")
        ax.set_ylabel("$(r_p/r_s)^2$")
        
        #set y low limit 
        base_rprs= self.transmission.planet.fullRadius**2/self.transmission.star.radius**2
        ax.axhline(y=base_rprs, alpha=0)
        
        ## asign the zscale to spectrum[1]
        self._zscale= self.transmission.altitude_profile*1e-3


        #add other y axis in the right
        ax2 = ax.twinx()
        ax2.axhline(y=max(self._zscale), alpha=0)
        ax2.axhline(y=0, alpha=0)
        #ax2.plot(10000/spectrum[0],self._zscale*np.ones_like(spectrum[0]), alpha=0)
        ax2.set_ylabel("Altitude [km]")
        ax2.tick_params(axis='y')
                           
        if showfig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax

    ## plot contributions
    def plot_contributions(self, wn_grid, showfig=False, showspectrum=True):
        """
        Plot the spectrum for each contribution and molecule.
        
        Parameters:
        - wn_grid (array): Wave number grid (in cm-1).
        - showfig (bool): Whether to show the plot (optional).
        - showspectrum (bool): Whether to show the total spectrum (optional).        
        
        Returns:
        - fig (matplotlib.figure): Figure of the plot.
        - ax (matplotlib.axes): Axes of the plot.
        
        """
   
        spectrum=self.generate_contributions(wn_grid)
                   
        fig, ax = plt.subplots(figsize=(10, 5))
    
        for aporte in spectrum[1]:
            for mol in spectrum[1][aporte]:
                ax.plot(10000/spectrum[0],
                        spectrum[1][aporte][mol],
                        label=aporte+": "+mol,
                        )
                
        ax.set_xlabel("Wavelength [μm]")
        ax.set_ylabel("$(r_p/r_s^2)^2$")
        #set y low limit 
        base_rprs= self.transmission.planet.fullRadius**2/self.transmission.star.radius**2
        ax.axhline(y=base_rprs, alpha=0)
        #add other y axis in the right with the zscale
        ax2 = ax.twinx()
        ax2.axhline(y=max(self._zscale), alpha=0)
        ax2.axhline(y=0, alpha=0)
        #ax2.plot(10000/spectrum[0],self._zscale, alpha=0)
        ax2.set_ylabel("Altitude [km]")
        ax2.tick_params(axis='y')
        
        if showspectrum:
            ax.plot(10000/spectrum[0], 
                    self.generate_spectrum(wn_grid)[1],
                    label="Total Spectrum",
                    color="black",
                    alpha=0.5,
                    ls="--",
                    )
                
        ax.legend()
        
        if showfig:
            plt.show()
        else:
            plt.close(fig)
  
        return fig, ax

    def explore_multiverse(self, wn_grid, snr, n_universes, labels=None, header=False,
                           n_observations=1, spectra=True, observations=True, path=None):
        """
        Explore the multiverse, generate spectra and observations, and optionally save them in Parquet format.

        Parameters:
        - wn_grid (array): Wave number grid.
        - snr (float): Signal-to-noise ratio.
        - n_universes (int): Number of universes to explore.
            One planet per universe with properties drawn from the priors.
        - labels (list(list)): Labels for atmospheric composition (optional)[["CO2,"CH4"],"CH4"].
        - header (bool): Whether to save the header in the saved files (optional).
        - n_observations (int): Number of observations to generate (optional), default 1.
        - spectra (bool): Whether to save the spectra (optional), default True.
        - observations (bool): Whether to save the observations (optional), default False.
        - path (str, optional): Path to save the files. If not provided, the files are not saved.
        
        Returns:
        - dict: Dictionary containing 'spectra' and/or 'observations' DataFrames depending on arguments.
                - spectra (DataFrame): Spectra of the universes.
                - observations (DataFrame): Observations of the universes.
        """

        # Validate the transmission model
        if self._transmission is None:
            raise ValueError("A transmission model has not been generated.")
        
        if not any([spectra, observations]):
            raise ValueError("At least one of 'spectra' or 'observations' must be True.")
        
        # Initialize a list to store all spectra generated
        spectra_list = []
        header_list = []

        for i in tqdm(range(n_universes), desc="Exploring universes"):
            self.make_tm()
            #generate the spectrum dataframe
            bin_wn,bin_rprs=self.generate_spectrum(wn_grid)
            columns = list(10000 / np.array(bin_wn))            
            bin_rprs_reshaped = bin_rprs.reshape(1, -1)
            spec_df = pd.DataFrame(bin_rprs_reshaped, columns=columns)

            current_spec_df = spec_df
            
            # prepare the header            
            current_header= dict()
            # If header true, create a dataframe with system parameters
            if header:
                params_dict = self.get_params() 
                current_header = params_dict

            # Add the labels to the header as a list in a new column of the DataFrame
            if labels is not None:
                # Validate the labels to add to the list
                valid_labels = []
                for label in labels:
                    if (isinstance(label, str) 
                            and label in self.transmission.chemistry.gases):
                        valid_labels.append(label)
                    elif isinstance(label, list):
                        valid_sublabels = [sublabel for sublabel 
                                           in label if sublabel
                                           in self.transmission.chemistry.gases]
                        if valid_sublabels:
                            valid_labels.append(valid_sublabels)
                if valid_labels:
                    current_header["label"] = valid_labels
                else:
                    current_header["label"] = []
                    
            ## add the header to the list
            header_list.append(current_header)
            ## add the spectrum to the list
            spectra_list.append(current_spec_df)
            
            # Move universe
            self.move_universe()

        ## concatenate the list of spectra
        all_spectra_df = pd.concat(spectra_list, axis=0,
                                   ignore_index=True)             
        ## concatenate the list of headers
        all_header_df = pd.DataFrame(header_list)
                        
        # generate observations
        if observations:
            print(f"Generating observations for {n_universes} spectra...")
            all_observations_df = generate_df_noise(all_spectra_df,
                                                    all_header_df,n_observations,
                                                    snr)
            ## save the observations
            if path is not None:
                ## copy the dataframe
                all_observations_df_copy=all_observations_df.copy()
                ## transform the columns to string
                all_observations_df_copy.columns=all_observations_df_copy.columns.astype(str)
                all_observations_df_copy.to_parquet(f'{path}_observations.parquet')
            if spectra:
                all_spectra_df=pd.concat([all_header_df.reset_index(drop=True),all_spectra_df.reset_index(drop=True)], axis=1)
                ## save the spectra
                if path is not None:
                    ## copy the dataframe
                    all_spectra_df_copy=all_spectra_df.copy()
                    ## transform the columns to string
                    all_spectra_df_copy.columns=all_spectra_df_copy.columns.astype(str)
                    all_spectra_df_copy.to_parquet(f'{path}_spectra.parquet')
                return dict(
                    spectra=all_spectra_df,
                    observations=all_observations_df
                )
            else:
                return dict(observations=all_observations_df)
        
        else:            
            all_spectra_df=pd.concat([all_header_df.reset_index(drop=True),\
                                      all_spectra_df.reset_index(drop=True)],
                                     axis=1)
            ## save the spectra
            if path is not None:
                ## copy the dataframe
                all_spectra_df_copy=all_spectra_df.copy()
                ## transform the columns to string
                all_spectra_df_copy.columns=all_spectra_df_copy.columns.astype(str)
                all_spectra_df_copy.to_parquet(f'{path}_spectra.parquet')
            return dict(spectra=all_spectra_df)