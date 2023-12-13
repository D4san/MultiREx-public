import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import tqdm
#Taurex imports
from taurex.stellar import PhoenixStar, BlackbodyStar
from taurex.planet import Planet as tauP
from taurex.temperature import Isothermal
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.cache import OpacityCache, CIACache
from taurex.model import TransmissionModel
from taurex.contributions import AbsorptionContribution, RayleighContribution
from taurex.binning import FluxBinner, SimpleBinner  # Importa las bibliotecas de binner
from multirex.__randinstrument import RndSNRInstrument
import taurex.log
taurex.log.disableLogging()
from taurex.cache import OpacityCache,CIACache
OpacityCache().clear_cache()
xsec_path=os.path.join(os.path.dirname(__file__),'data')
OpacityCache().set_opacity_path(xsec_path)
import time

'''
IMPLEMENTAR el acceso directo a la carpeta de opacidades
OpacityCache().set_opacity_path(xsec_path)
'''

def generate_random_value(value_range, seed=None):
    """
    Generate a random value within the specified range.
    """

    return np.random.uniform(value_range[0], value_range[1])

def load_spec(path):
    """
    Read an spectrum from a file.
    Its assumed that the file has a header with the parameters of the system
    and the spectrum in the following lines.
    And returns a dataframe with the information in header
      and the fluxes (rp^2/rs^2) per wavelegth in μm
    """
    # read the data
    data=np.loadtxt(path, comments="#", delimiter=" ", unpack=False)
    #convert data to dataframe
    # set columns names to data[0]
    df2 = pd.DataFrame( columns=[10000/data[0]])
    # set lines to data[1]
    df2.loc[0] = data[1]

    # read the header
    with open(path) as f:
        header = json.loads(f.readline()[1:])

    df1=pd.DataFrame([header])

    #concatenate header and fluxes
    df=pd.concat([df1,df2], axis=1)
    return df

def wavenumber_grid(wl_min, wl_max, resolution):
    """
    Generate a wave number grid from a wavelength range and resolution.
    """

    return np.sort(10000 / np.logspace(np.log10(wl_min), np.log10(wl_max), resolution))

class Planet:
    """
    Planet class representing a celestial body.

    Inputs:
    - seed (int): Random seed for reproducibility.
    - albedo (float or tuple): Albedo of the planet (single value or range).
    - radius (float or tuple): Radius of the planet (single value or range). In Earth radius.
    - mass (float or tuple): Mass of the planet (single value or range). In Earth masses.
    - atmosphere (Atmosphere): An Atmosphere object.
    """

    def __init__(self, seed=None, 
                 albedo=None, radius=None, 
                 mass=None, atmosphere=None):
        
        self.original_values = {"seed": seed}
        if seed is not None:
            self.seed=seed
        else:   
            self.seed= int(time.time())
        
        np.random.seed(self.seed)
        

        self.albedo = None
        self.radius = None
        self.mass = None
        self.atmosphere = None
        
        
        if albedo is not None:
            self.set_albedo(albedo)
        if radius is not None:
            self.set_radius(radius)
        if mass is not None:
            self.set_mass(mass)
        if atmosphere is not None:
            self.set_atmosphere(atmosphere)

    def set_albedo(self, value):
        """
        float or tuple: Albedo of the planet (single value or range).
        """
        self.original_values["albedo"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.albedo = generate_random_value(value, self.seed)
        else:
            self.albedo = value

    def set_radius(self, value):
        """ 
        float or tuple: Radius of the planet (single value or range). In Earth radius.
        """
        self.original_values["radius"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.radius = generate_random_value(value, self.seed)
        else:
            self.radius = value

    def set_mass(self, value):
        """
        float or tuple: Mass of the planet (single value or range). In Earth masses.
        """
        self.original_values["mass"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.mass = generate_random_value(value, self.seed)
        else:
            self.mass = value

    def set_seed(self, value):
        self.original_values["seed"] = value
        self.seed = value
    

        
    def set_atmosphere(self, value):
        self.atmosphere = value
        
    def get_atmosphere(self):
        return self.atmosphere.get_params()
    
    
    def get_params(self):
    
        """
        Get the current parameters of the planet and atmosphere.
        """  
        params={
            "p albedo": self.albedo,
            "p radius": self.radius,
            "p mass": self.mass,
            "p seed": self.seed
        }
        if self.atmosphere is not None:
   
            params.update(
                {("atm "+i): self.atmosphere.get_params()[i] for i in self.atmosphere.get_params()}
            )
            #remove coposition and add as individual parameters
            params.pop("atm composition")
            params.update(
                {("atm "+i): self.atmosphere.get_params()["composition"][i] for i in self.atmosphere.get_params()["composition"]}
            )
        return params
            
    def move_universe(self,atmosphere=False):
        if self.original_values["seed"] is not None:
            self.seed=self.original_values["seed"]
        else:
            self.seed=int(time.time())
            
        np.random.seed(self.seed)

        # regenerate the planet
        self.set_albedo(self.original_values["albedo"])
        self.set_radius(self.original_values["radius"])
        self.set_mass(self.original_values["mass"])
        if atmosphere:
            self.atmosphere.move_universe()

class Atmosphere:
    """
    Atmosphere class representing a planet's atmospheric properties.

    Inputs:
    - seed (int): Random seed for reproducibility.
    - temperature (float or tuple): Temperature of the atmosphere (single value or range).
    - base_pressure (float or tuple): Base pressure of the atmosphere (single value or range). In Pa
    - top_pressure (float or tuple): Top pressure of the atmosphere (single value or range). In Pa
    - composition (dict): Composition of the atmosphere with gases and mix ratios.
    - fill_gas (str): Gas used as filler in the atmosphere composition.
    """

    def __init__(self, seed=None, temperature=None, 
                 base_pressure=None, top_pressure=None, 
                 composition=None, fill_gas=None):
        
        self.original_values = {"seed": seed}
        if seed is not None:
            self.seed=seed
        else:
            self.seed=int(time.time())
            
        np.random.seed(self.seed)

        self.temperature = None
        self.base_pressure = None
        self.top_pressure = None
        self.composition = {}
        self.fill_gas = fill_gas
  

        if temperature is not None:
            self.set_temperature(temperature)
        if base_pressure is not None or top_pressure is not None:
            self.set_pressure(base_pressure, top_pressure)
        if composition is not None:
            self.set_composition(composition, fill_gas)

    def validate_composition(self):
        total_mix_ratio = sum(self.composition.values())
        if total_mix_ratio < 0 or total_mix_ratio > 1:
            raise ValueError("The sum of mix ratios must be between 0 and 1.")

    def set_temperature(self, value):
        """
        float or tuple: Temperature of the atmosphere (single value or range).
        """

        self.original_values["temperature"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.temperature = generate_random_value(value, self.seed)
        else:
            self.temperature = value

    def set_pressure(self, base=None, top=None):
        
        """
        base (float or tuple): Base pressure of the atmosphere (single value or range). In Pa
        top (float or tuple): Top pressure of the atmosphere (single value or range). In Pa
        """

        if base is not None:
            self.original_values["base_pressure"] = base
            if isinstance(base, tuple) and len(base) == 2:
                self.base_pressure = generate_random_value(base, self.seed)
            else:
                self.base_pressure = base

        if top is not None:
            self.original_values["top_pressure"] = top
            if isinstance(top, tuple) and len(top) == 2:
                self.top_pressure = generate_random_value(top, self.seed)
            else:
                self.top_pressure = top

    def set_seed(self, value):
        self.original_values["seed"] = value
        self.seed = value
    

    def validate_gas_mix(self, gas_mix):
        total_mix_ratio = sum(10**mix for mix in gas_mix.values())
        
        if total_mix_ratio > 1:
            raise ValueError(f"The sum of mix ratios must be between 0 and 1.")

    def set_composition(self, gases, fill_gas=None):
        """
        gases (dict): Composition of the atmosphere with gases and mix ratios. (eg. {"H2O": 0.5, "CO2": 0.5})
            The structure of the dictionary is as follows:
            - gas key (str): Gas name key (e.g. "H2O").
            - gas mix_ratio (float or tuple): Value of the mix ratio (single value or range).
        
        fill_gas(str): Gas used as filler in the atmosphere composition. (optional)
        """
        if fill_gas is not None:
            self.original_values["fill_gas"] = fill_gas
            self.fill_gas = fill_gas

        self.original_values["composition"] = gases

        for gas, mix_ratio in gases.items():
        
            if isinstance(mix_ratio, tuple) and len(mix_ratio) == 2:
                value = generate_random_value(mix_ratio, self.seed)
            
            elif isinstance(mix_ratio, tuple) and len(mix_ratio) > 2:
                raise ValueError("Mix ratio must be a single value or a range.")

            else:
                value = mix_ratio

            if  value > 1:
                raise ValueError(f"Mix ratio must be between 0 and 1., actualy gas {gas} has value {value}")

            self.composition[gas] = value

        self.validate_gas_mix(self.composition)
        
    def set_fill_gas(self, gas):
        """
        gas (str): Gas used as filler in the atmosphere composition.
        """
        self.original_values["fill_gas"] = gas
        self.fill_gas = gas
        

    def add_gas(self, gas, mix_ratio):
        """
        gas (str): Gas name key (e.g. "H2O").
        mix_ratio (float or tuple): Value of the mix ratio (single value or range).
        """

        if gas in self.composition:
            raise ValueError(f"{gas} already exists in the composition. Use set_composition to modify it.")
        if isinstance(mix_ratio, tuple) and len(mix_ratio) == 2:
            value = generate_random_value(mix_ratio, self.seed)
        else:
            value = mix_ratio
        if  value > 1:
            raise ValueError("Mix ratio must be between 0 and 1.")
        self.composition[gas] = value
        self.original_values["composition"].update({gas: mix_ratio})
        self.validate_gas_mix(self.composition)

    def remove_gas(self, gas):
        """
        gas (str): Gas name key (e.g. "H2O").
        """

        if gas not in self.composition:
            raise ValueError(f"{gas} does not exist in the composition.")
        del self.composition[gas]
        #remove from original values
        self.original_values["composition"].pop(gas)
        self.validate_gas_mix(self.composition)

    def get_params(self):
        """
        Get the current parameters of the atmosphere.
        """
        params = {
            "temperature": self.temperature,
            "base_pressure": self.base_pressure,
            "top_pressure": self.top_pressure,
            "composition": self.composition,
            "fill_gas": self.fill_gas,
            "seed": self.seed
        }
        return params
        
    def move_universe(self):
        if self.original_values["seed"] is not None:
            self.seed=self.original_values["seed"]
        else:
            self.seed=int(time.time())
            
        np.random.seed(self.seed)
        
        # regenerate the atmosphere
        self.set_temperature(self.original_values["temperature"])
        self.set_pressure(self.original_values["base_pressure"]
                          , self.original_values["top_pressure"])
        self.set_composition(self.original_values["composition"], self.original_values["fill_gas"])
        
       

class Star:
    """
    Star class representing a celestial star.

    Inputs:
    - seed (int): Random seed for reproducibility.
    - temperature (float or tuple): Temperature of the star (single value or range). In Kelvin.
    - radius (float or tuple): Radius of the star (single value or range). In solar radii.
    - mass (float or tuple): Mass of the star (single value or range). In solar masses.
    """

    def __init__(self, seed=None, temperature=None,
                 radius=None, mass=None):
        self.original_params = {"seed": seed}
        if seed is not None:
            self.seed=seed
        else:
            self.seed=int(time.time())
            
            
        np.random.seed(self.seed)
        
        
        self.temperature = None
        self.radius = None
        self.mass = None
        self.magnitudeK = 10
        self.metallicity = 1
        

        if temperature is not None:
            self.set_temperature(temperature)
        if radius is not None:
            self.set_radius(radius)
        if mass is not None:
            self.set_mass(mass)

    def set_temperature(self, value):   
        """
        float or tuple: Temperature of the star (single value or range). In Kelvin.
        """

        self.original_params["temperature"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.temperature = generate_random_value(value, self.seed)
        else:
            self.temperature = value

    def set_radius(self, value):
        """
        float or tuple: Radius of the star (single value or range). In solar radius.
        """

        self.original_params["radius"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.radius = generate_random_value(value, self.seed)
        else:
            self.radius = value

    def set_mass(self, value):
        """
        float or tuple: Mass of the star (single value or range). In solar masses.
        """

        self.original_params["mass"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.mass = generate_random_value(value, self.seed)
        else:
            self.mass = value

    def set_seed(self, value):
        self.original_params["seed"] = value
        self.seed = value

    def get_params(self):
        """
        Get the current parameters of the star.
        """
        params = {
            "s temperature": self.temperature,
            "s radius": self.radius,
            "s mass": self.mass,
            "s magnitudeK": self.magnitudeK,
            "s metallicity": self.metallicity,
            "s seed": self.seed
        }

        return params

        
    def move_universe(self):
        if self.original_params["seed"] is not None:
            self.seed=self.original_params["seed"]
        else:
            self.seed=int(time.time())
            
        np.random.seed(self.seed)
        # regenerate the star
        self.set_temperature(self.original_params["temperature"])
        self.set_radius(self.original_params["radius"])
        self.set_mass(self.original_params["mass"])

class System:
    """
    System class representing a celestial system consisting of a planet and a star.

    Inputs:
    - planet (Planet): A Planet object.
    - star (Star): A Star object.
    - distance_parsecs (float or tuple): Distance from the system to the observer in parsecs (single value or range).
    - planet_distance (float or tuple): Distance from the planet to the star in AU (single value or range).
    - orbital_period (float or tuple): Orbital period of the planet  in days (single value or range).
    - transit_time (float or tuple): Transit time of the planet in seconds (single value or range).
    """

    def __init__(self, planet, star,seed=None, distance_parsecs=None, planet_distance=None, orbital_period=None, transit_time=None):
        
        self.original_params = {"seed": seed}
        if seed is not None:
            self.seed=seed
        
        else:
            self.seed=int(time.time())
            
        np.random.seed(self.seed)

        self.planet = planet
        self.star = star
        self.distance_parsecs = None
        self.planet_distance = None
        self.orbital_period = None
        self.transit_time = None
        
        if distance_parsecs is not None:
            self.set_distance_parsecs(distance_parsecs)
        if planet_distance is not None:
            self.set_planet_distance(planet_distance)
        if orbital_period is not None:
            self.set_orbital_period(orbital_period)
        if transit_time is not None:
            self.set_transit_time(transit_time)
        self.transmission=None

        ## earth radius and solar radius from astropy
        self.earth_radius=6378136e-3 #km
        self.solar_radius=695508000e-3 #km

        
        

    def set_distance_parsecs(self, value):
        """
        float or tuple: Distance from the system to the observer in parsecs (single value or range).
        """

        self.original_params["distance_parsecs"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.distance_parsecs =\
                generate_random_value(value, self.seed)
        else:
            self.distance_parsecs = value

    def set_planet_distance(self, value):
        """
        float or tuple: Distance from the planet to the star in AU (single value or range).
        """

        self.original_params["planet_distance"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.planet_distance = \
                generate_random_value(value, self.seed)
        else:
            self.planet_distance = value

    def set_orbital_period(self, value):
        """
        float or tuple: Orbital period of the planet  in days (single value or range).
        """

        self.original_params["orbital_period"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.orbital_period = \
                generate_random_value(value, self.seed)
        else:
            self.orbital_period = value

    def set_transit_time(self, value):
        """
          float or tuple: Transit time of the planet in seconds (single value or range).
        """
        self.original_params["transit_time"] = value
        if isinstance(value, tuple) and len(value) == 2:
            self.transit_time = generate_random_value(value, self.seed)
        else:
            self.transit_time = value

    def set_seed(self, value):
        self.original_params["seed"] = value
        self.seed = value

    def get_params(self):
        """
        Get the current parameters of the system.
        """
        params = {
            "distance_parsecs": self.distance_parsecs,
            "planet_distance": self.planet_distance,
            "orbital_period": self.orbital_period,
            "transit_time": self.transit_time,
            "sys seed": self.seed
        }

        params.update(
            {(i): self.planet.get_params()[i] for i in self.planet.get_params()}
        )
        params.update(
            {(i): self.star.get_params()[i] for i in self.star.get_params()}
        )

        return params

    def move_universe(self,atmosphere=True):
        # regenerate the system    
        if self.original_params["seed"] is not None:
            self.seed=self.original_params["seed"]
        else:
            self.seed=int(time.time())
            
        np.random.seed(self.seed)
        
        self.set_distance_parsecs(self.original_params["distance_parsecs"])
        self.set_planet_distance(self.original_params["planet_distance"])
        self.set_orbital_period(self.original_params["orbital_period"])
        self.set_transit_time(self.original_params["transit_time"])
        self.planet.move_universe(atmosphere=atmosphere)
        self.star.move_universe()

    def make_transmission_model(self):
        """
        Generate a transmission model for the system.
        """
        
        """ 
        IMPLEMENT ERRORS
        """
        
        #generate Taurex planet
        mJ=317.8 # masa de júpiter en [masas terrestres]
        rJ=10.97 # radio de júpiter en [radios terrestres]
        tauplanet=tauP(planet_distance=self.planet_distance,
                    planet_mass=self.planet.mass/mJ,
                    planet_radius=self.planet.radius/rJ,
                    albedo=self.planet.albedo,
                    orbital_period=self.orbital_period,
                    transit_time=self.transit_time)
        
        
        #generate Taurex star
        taustar=BlackbodyStar(temperature=self.star.temperature,
                            radius=self.star.radius,
                            magnitudeK=self.star.magnitudeK,
                            metallicity=self.star.metallicity,
                            distance=self.distance_parsecs)
        
        
        #generate Taurex temperature
        tautemperature=Isothermal(T=self.planet.atmosphere.temperature)
        
        ## gemerate Taurex chemistry
        
        tauchem=TaurexChemistry(fill_gases=self.planet.atmosphere.fill_gas)
        for gas, mix_ratio in self.planet.atmosphere.composition.items():
            tauchem.addGas(ConstantGas(molecule_name=gas,
                                        mix_ratio=10**mix_ratio))
        
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
        
        self.transmission=tm
        

    def generate_spectrum(self, wn_grid):
        """
        Generate a spectrum based on a wave number grid.

        Args:
        - wave_numbers (array): Wave number grid.

        Output:
        - bin_wn (array): Wave number grid.
        - bin_rprs (array): Fluxes in rp^2/rs^2.
        - bin_zscale (array): Vertical scale in KM.
        """
        # Create a binner
        bn = FluxBinner(wngrid=wn_grid)
        bin_wn, bin_rprs, _, _ = bn.bin_model(\
            self.transmission.model(wngrid=wn_grid))
        
        #vertical scale

        bin_zscale=np.sqrt(bin_rprs)*self.star.radius*self.solar_radius-self.planet.radius*self.earth_radius
        
        
        return bin_wn, bin_rprs, bin_zscale
    
    def generate_full_spectrum(self, wn_grid):
        """
        generate a diferentiated spectrum contribution based on a wave number grid.
        
        bin_wn (array): Wave number grid.
        bin_rprs (dict): Fluxes in rp^2/rs^2 per contribution and molecule.
        """
        bn = FluxBinner(wngrid=wn_grid)
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
       
    def generate_observations(self, wn_grid, snr, num_observations, path=None, header=False):
        """
        Generate observations with noise based on a wave number grid.

        Args:
        - wave_numbers (array): Wave number grid.
        - snr (float): Signal-to-noise ratio.
        - num_observations (int): Number of observations to generate.
        - path (str): Path to save observations (optional).

        Returns:
        observations (array): Observations with noise.
            [0] = wave number grid
            [1] = fluxes in rp^2/rs^2
            [2] = noise in rp^2/rs^2
        """
        bn=FluxBinner(wngrid=wn_grid)
        inst = RndSNRInstrument(snr, bn)  # Instrumento con una SNR específica
        
        # Generate observations
        observations = []
        for i in range(num_observations):
            observations.append(inst.model_noise(self.transmission)[:-1])
            if path is not None:
                #convert i to string of N digits
                i_str=str(i).zfill(len(str(num_observations-1)))
                if header:
                    header_str = json.dumps(self.get_params())
                    np.savetxt(path+f"observation_{i_str}.txt", observations[i],
                           header=header_str, comments="#" )
                else:
                    np.savetxt(path+f"observation_{i_str}.txt", observations[i])
                
        return observations

    ## plots 
    ## plot spectrum
    def plot_spectrum(self,  wn_grid, title=None):
        """
        Plot the spectrum.
        
        Args:
        - wn_grid (array): Wave number grid (in cm-1).
        - title (str): Title of the plot (optional).
        
        Returns:
        - ax (matplotlib.axes): Axes of the plot.
        - fig (matplotlib.figure): Figure of the plot.

        """
       
        spectrum=self.generate_spectrum(wn_grid)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(10000/spectrum[0], spectrum[1])
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("($r_p^2$/$r_s^2$)")
        ax.set_title(title)
        ax.grid()
        plt.show()
        
        return ax, fig
    

    ## plot full spectrum
    def plot_full_spectrum(self, wn_grid, title=None):
        """
        Plot the spectrum for each contribution and molecule.
        
        Args:
        - wn_grid (array): Wave number grid (in cm-1).
        - title (str): Title of the plot (optional).
        
        Returns:
        - ax (matplotlib.axes): Axes of the plot.
        - fig (matplotlib.figure): Figure of the plot.
        
        """
   
        spectrum=self.generate_full_spectrum(wn_grid)
        fig, ax = plt.subplots(figsize=(10, 5))
        for aporte in spectrum[1]:
            for mol in spectrum[1][aporte]:
                ax.plot(10000/spectrum[0], spectrum[1][aporte][mol], label=aporte+": "+mol)
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("$r_p^2$/$r_s^2$")
        ax.set_title(title)
        ax.grid()
        ax.legend()
        plt.show()
        
        return ax, fig
                    


    def explore_multiverse(self, wn_grid, snr, path, n_iter, 
                           labels=None, header=False, n_observations=1, 
                           spectrum=True, observations=False):
        """
        
        Explore the multiverse and save spectra and observations.

        Args:
        - wave_numbers (array): Wave number grid.
        - snr (float): Signal-to-noise ratio.
        - path (str): Path to save results.
        - labels (list(list)): Labels for atmospheric composition (optional).
        - header (bool): Whether to save the header in the saved files (optional).
        - n_observations (int): Number of observations to generate (optional), default 1.
        - spectrum (bool): Whether to save the spectrum (optional), default True.
        - observations (bool): Whether to save the observations (optional), default False.
    
        Output:
        i is the number of universe and j is the number of observation
        - observations (optional) (.txt): Observations with noise
            path/observations/observation_{i}-{j}.txt
        - spectrum (optional) (.txt): Spectrum.
            path/spectrum/spectrum_{i}.txt 
            
        """
        
        header_str = False
        if spectrum == False  and observations == False:
            raise ValueError("At least one of spectrum, full_spectrum \
                or observations must be True.")
        if spectrum:
            spectrum_path=path+"/multirex/spectrums/"
            if not os.path.exists(spectrum_path):
                os.makedirs(spectrum_path)
        if observations:
            observations_path=path+"/multirex/observations/"
            if not os.path.exists(observations_path):
                os.makedirs(observations_path)
        
        for i in tqdm.tqdm(range(n_iter), desc="Number of universes explored"):
            
            i_str=str(i).zfill(len(str(n_iter-1)))
            self.make_transmission_model()
            
            if header:
                header_base= self.get_params()
                
            else:
                header_base={}
                
            if labels is not None:
                header_base["label"]=[]
                for label in labels:
                    if all(gas in self.transmission.chemistry.gases for gas in label):
                        # add label to header
                        header_base["label"].append(label)
                if len(header_base["label"])==0:
                    header_base["label"]=0
            if len(header_base)==0:
                header_str=False
            else:
                header_str = json.dumps(header_base)
            
            
            if spectrum:                
                spec=self.generate_spectrum(wn_grid)    
                spec=np.array(spec)          
                if header_str:
                    np.savetxt(spectrum_path+f"spec_uni-{i_str}.txt", 
                           spec, header=header_str, comments="#" )
                else:
                    np.savetxt(spectrum_path+f"spec_uni-{i_str}.txt", 
                           spec)
                     
            if observations:
                o_spec=self.generate_observations(wn_grid, snr, n_observations)
                for j in tqdm.tqdm(range(n_observations), desc="Number of observations realized"):
                    j_str=str(j).zfill(len(str(n_observations-1)))
                    
                    if header_str:
                        np.savetxt(observations_path+f"uni-{i_str}_obs-{j_str}.txt", 
                           o_spec[j], header=header_str, comments="#" )
                    else:
                        np.savetxt(observations_path+f"uni-{i_str}_obs-{j_str}.txt", 
                           o_spec[j])
                        
            #change universe
            self.move_universe()
                    
                
               
        
  