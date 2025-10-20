#########################################
#  __  __      _ _   _ ___ ___          #
# |  \/  |_  _| | |_(_) _ \ __|_ __     #
# | |\/| | || | |  _| |   / _|\ \ /     #
# |_|  |_|\_,_|_|\__|_|_|_\___/_\_\     #
# Planetary spectra generator           #
#########################################

"""
MultiREx Utilities Module

This module provides utility functions for the MultiREx library, primarily focused on
downloading and managing external data files needed for spectrum generation.

The main functions in this module are:
    - get_stellar_phoenix: Downloads Phoenix stellar spectra models
    - get_gases: Downloads opacity database for atmospheric gases
    - list_gases: Lists available gases in the opacity database
"""

import numpy as np
import gdown
import os
import zipfile
import pathlib
import itertools
from taurex.cache import OpacityCache, CIACache

def get_stellar_phoenix(path=""):
    """Download the Phoenix stellar spectra from the Google Drive link and
    extract the content to the specified path.
    
    This function automates the download and extraction of Phoenix stellar model files,
    which are used for more accurate stellar spectrum modeling compared to blackbody models.
    If the Phoenix directory already exists at the specified path, no download occurs.
    
    Args:
        path (str, optional): 
            Directory path where the Phoenix folder will be created
            and model files will be downloaded. If empty string, uses current directory.
            Defaults to "".
    
    Returns:
        str: Path to the Phoenix directory containing the stellar model files.
    
    Note:
        This function requires an internet connection for the initial download.
        The Phoenix models are approximately 2GB in size.
    """


    phoenix_path = os.path.join(path, 'Phoenix')
    # ZIP file URL
    url = 'https://drive.google.com/uc?id=1fgKjDu9H26y5WMwRZaMCuSpHhx8zc0pR'
    # Local ZIP file name
    zip_path = os.path.join(path, 'Phoenix.zip')

    # Check if the Phoenix directory already exists
    if not os.path.exists(phoenix_path):
        
        if path == "":
            print("The path where the Phoenix stellar spectra will be downloaded is : ",
              "current directory")
        else:
            print("The path where the Phoenix stellar spectra will be downloaded is: ",
              path)
        
        # Download the ZIP file
        gdown.download(url, zip_path, quiet=False)

        # Unzip the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)

        # Delete the ZIP file after extraction
        os.remove(zip_path)
    else:
        print("The directory to Phoenix already exists in the specified path: ",
              path if path != "" else "current directory")
    return phoenix_path

def get_gases(path=""):   
    """Download the opacity database from the Google Drive link and 
    extract the content to the specified path.
    
    This function automates the download and extraction of opacity data files for
    atmospheric gases, which are required for spectrum generation. The opacity data
    is used by TauREx to calculate the absorption of light by different gases in
    the atmosphere. If the opacity database already exists at the specified path,
    no download occurs.
    
    Args:
        path (str, optional): Directory path where the opacity database will be
        downloaded and extracted. If empty string, uses current directory.
        Defaults to "".
    
    Note:
        This function requires an internet connection for the initial download.
        After downloading, the opacity path is automatically set in the TauREx
        OpacityCache for immediate use.
        
        The opacity database is approximately 3GB in size.
    """
     # 1) If no path is provided, use the current directory
    if path == "":
        path = os.getcwd()
    os.makedirs(path, exist_ok=True)

    molecule_path = os.path.join(path, 'opacidades-todas')
    if not os.path.exists(molecule_path):
        url = 'https://drive.google.com/uc?id=1z7R0hD1IBuYo-nnl7dpE_Ls2337a0uv6'
        zip_path = os.path.join(path, "opacidades-todas.zip")

        print("Downloading the opacity database to:", path)
        gdown.download(url, zip_path, quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        os.remove(zip_path)
    else:
        print("The opacity database already exists at:", molecule_path)

    # 2) Update the TauREx cache
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(molecule_path)
    
def get_CIAs(pairs=None, atmosphere=None, path="", session=None):
    """Download and configure CIA data for TauREx.

    - pairs: list of CIA pairs, e.g. ['H2-H2','H2-He'] or [('H2','H2'),('H2','He')]
    - atmosphere: Atmosphere-like object; if provided and pairs is None, infer from
      `atmosphere.cia` (if set) or from `composition` and `fill_gas`.
    - path: destination directory for CIA files; if empty, uses env `TAUREX_CIA_PATH`
      or defaults to `./data/cia` relative to current working directory.
    - session: optional requests.Session for authenticated downloads.

    Returns:
        (saved_files, normalized_pairs): list of saved file paths (str) and list of
        normalized pair strings like ['H2-H2','H2-He'].
    """
    # Resolve destination directory
    if path == "":
        env_path = os.environ.get("TAUREX_CIA_PATH", "")
        path = env_path if env_path else os.path.join(os.getcwd(), "data", "cia")
    os.makedirs(path, exist_ok=True)
    cia_dir = pathlib.Path(path).resolve()

    # Helper normalizers/parsers
    def _normalize_species(s):
        return s.replace(" ", "").upper()

    def _parse_pair_str(spec_str):
        parts = spec_str.replace(" ", "").upper().replace("_","-").split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid CIA pair spec '{spec_str}', expected format 'A-B'")
        a,b = parts
        return tuple(sorted((a,b)))

    def _normalize_pairs(pairs_input):
        normalized = []
        for p in pairs_input:
            if isinstance(p, str):
                normalized.append(_parse_pair_str(p))
            elif isinstance(p, (tuple, list)) and len(p) == 2:
                a,b = p
                normalized.append(tuple(sorted((_normalize_species(a), _normalize_species(b)))))
            else:
                raise ValueError("Each CIA pair must be a string 'A-B' or 2-item tuple/list")
        return sorted(set(normalized))

    # Minimal mapping (extend with more pairs as needed)
    PAIR_TO_FILE = {
        ("H2","H2"):   "H2-H2_2011.cia",
        ("H2","HE"):   "H2-He_2011.cia",
        ("H2","CH4"):  "H2-CH4_eq_2011.cia",
        ("N2","N2"):   "N2-N2_2021.cia",
        ("O2","O2"):   "O2-O2_2024.cia",
        ("O2","N2"):   "O2-N2_2024.cia",
        ("N2","H2"):   "N2-H2_2024.cia",
        ("CO2","CO2"): "CO2-CO2_2024.cia",
        ("CO2","H2"):  "CO2-H2_2024.cia",
        ("CO2","HE"):  "CO2-He_2018.cia",
        ("CH4","HE"):  "CH4-He_2018.cia",
    }

    def _infer_pairs_from_atmosphere(atm):
        # If atmosphere has explicit CIA pairs, use them
        if hasattr(atm, "cia") and atm.cia:
            return _normalize_pairs(atm.cia)
        comp = getattr(atm, "composition", {}) or {}
        fill = getattr(atm, "fill_gas", []) or []
        if isinstance(fill, str):
            fill = [fill]
        active = set(_normalize_species(g) for g in list(comp.keys()) + list(fill))
        whitelist = {"H2","HE","N2","O2","CO2","CH4"}
        species = sorted(active & whitelist)
        pairs = set()
        for a in species:
            for b in species:
                key = tuple(sorted((a,b)))
                if key in PAIR_TO_FILE:
                    pairs.add(key)
        return sorted(pairs)

    # Decide target pairs
    if pairs is not None:
        normalized_pairs = _normalize_pairs(pairs)
    elif atmosphere is not None:
        normalized_pairs = _infer_pairs_from_atmosphere(atmosphere)
    else:
        raise ValueError("Provide 'pairs' (list) or 'atmosphere' to determine CIA pairs")

    # Local import of requests to avoid hard dep when not used
    if session is None:
        try:
            import requests
        except ImportError:
            raise ImportError("'requests' is required to download CIA files. Install requests or pass a Session.")
        s = requests.Session()
    else:
        s = session

    CIA_BASE = "https://hitran.org/data/CIA"
    saved_files = []
    for pair in normalized_pairs:
        if pair not in PAIR_TO_FILE:
            print(f"Skipping CIA pair {pair}: no filename mapping available")
            continue
        filename = PAIR_TO_FILE[pair]
        dest = cia_dir / filename
        if dest.exists():
            saved_files.append(dest)
            continue
        url = f"{CIA_BASE}/{filename}"
        try:
            r = s.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            saved_files.append(dest)
        except Exception as e:
            print(f"Failed to download {filename} ({pair}): {e}")

    # Configure TauREx CIA cache
    cache = CIACache()
    if hasattr(cache, 'clear_cache'):
        cache.clear_cache()
    if hasattr(cache, 'set_cia_path'):
        cache.set_cia_path(str(cia_dir))
    # Pretty-case species names for returned pair strings
    def _pretty(s):
        return "He" if s == "HE" else s
    return [str(p) for p in saved_files], [f"{_pretty(a)}-{_pretty(b)}" for a,b in normalized_pairs]

def list_gases():
    """List all available gases in the opacity database.
    
    This function prints the names of all atmospheric gases available in the
    current opacity database. These gases can be used in the composition of
    an Atmosphere object.
    
    Returns:
        None: The list of available gases is printed to the console.
        
    Note:
        You must first download the opacity database using get_gases() before
        this function will show the complete list of available gases.
        
    Example:
        >>> import multirex.utils as Util
        >>> Util.get_gases()  # Download the opacity database
        >>> Util.list_gases()  # List available gases
    """
    print("Available gases in the database:")
    print(list(OpacityCache().find_list_of_molecules()))
