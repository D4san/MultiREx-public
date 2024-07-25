#########################################
#  __  __      _ _   _ ___ ___          #
# |  \/  |_  _| | |_(_) _ \ __|_ __     #
# | |\/| | || | |  _| |   / _|\ \ /     #
# |_|  |_|\_,_|_|\__|_|_|_\___/_\_\     #
# Planetary spectra generator           #
#########################################

import numpy as np
import gdown
import os
import zipfile
from taurex.cache import OpacityCache

def get_stellar_phoenix(path=""):
    """Download the Phoenix stellar spectra from the Google Drive link and
    extract the content to the specified path."""

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
    """
    
    # Define the directory path where the content will be extracted
    molecule_path = os.path.join(path,'opacidades-todas')
    if os.path.exists(molecule_path):
        print("The directory to the opacity database already exists in the specified path: ",
              path if path != "" else "current directory")
    else:
        # URL of the ZIP file to download
        url = 'https://drive.google.com/uc?id=1z7R0hD1IBuYo-nnl7dpE_Ls2337a0uv6'
        # Local ZIP file name
        zip_path = path+"/opacidades-todas.zip"

        # Check if the directory already exists
        if not os.path.exists(molecule_path):
            
            if path == "":
                print("The path where the opacity database will be downloaded is : ",
                "current directory")
            else:
                print("The path where the opacity database will be downloaded is: ",
                path)
            
            # Download the ZIP file
            gdown.download(url, zip_path, quiet=False)

            # Unzip the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path)

            # Delete the ZIP file after extraction
            os.remove(zip_path)
        else:
            print("The directory to the opacity database already exists in the specified path: ",
                path if path != "" else "current directory")
            
    OpacityCache().clear_cache()
    xsec_path=molecule_path
    OpacityCache().set_opacity_path(xsec_path)
    
def list_gases():
    print("Available gases in the database:")
    print(list(OpacityCache().find_list_of_molecules()))
