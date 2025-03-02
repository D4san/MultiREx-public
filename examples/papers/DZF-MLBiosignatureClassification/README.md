# MultiREx
## Planetary transmission spectra generator

## Description

This folder contains the notebooks and data necessary to replicate the results and plots of the scientific article titled "Machine-assisted classification of potential biosignatures in Earth-like exoplanets using low signal-to-noise ratio transmission spectra." Here, you will find all the procedures, scripts, and data files used in the numerical experiments presented in the study.

## Requirements

The following Python libraries are required to run the scripts and notebooks in this folder:

- multirex
- pandexo and its dependencies
- scikit-learn
- plotly
- POSEIDON

## Structure

The folder is organized as follows:

- `spec_data/`: Contains the transmission spectra as CSV files. Each file contains spectra for a specific combination of molecules.
- `spec_earth/`: Contains the notebook required for the realistic case analysis, the high-resolution transmission spectra of Earth used in the study, the transmission spectra adapted to TRAPPIST-1e, and the observed transmission spectra simulated by Pandexo, differentiated by the number of transits.
- `stellar_contamination`: Contains the epsilon factor for the stellar contamination. Each file are computed for a specific combitation of spot and facula fractions.
- Main folder: Contains the notebooks used to generate the results and plots of the study. The notebooks are organized as follows:
  - `01_pandexo_spec_analysis.ipynb`: This notebook contains the analysis of the transmission for use in NIRSpec MIRI. The first part generates the `waves.txt` file, representing the range of wavelengths used in the study. The second part analyzes the relationship between the number of transits and the SNR.
  - `02_spec_data.ipynb`: This notebook generates the transmission spectra for the molecules used in the study.
  - `02_stellar_contamination_epsilon.ipynb`: This notebook generates the epsilon factors for the stellar contamination.
  - `03_AE.ipynb`: This notebook is used to train the common AutoEncoder (AE).
  - `03_AE_CH4.ipynb`: This notebook is used to train the specialized AutoEncoder for CH4.
  - `03_AE_O3.ipynb`: This notebook is used to train the specialized AutoEncoder for O3.
  - `03_AE_H2O.ipynb`: This notebook is used to train the specialized AutoEncoder for H2O.
  - `04_CH4_RF.ipynb`: This notebook is used to train the specialized Random Forest for CH4.
  - `04_O3_RF.ipynb`: This notebook is used to train the specialized Random Forest for O3.
  - `04_H2O_RF.ipynb`: This notebook is used to train the specialized Random Forest for H2O.
  - `05_BC_RF.ipynb`: This notebook contains the training of the Binary Classification (BC) Random Forest and the analysis and plots performed in its section of the paper.
  - `05_MC_RF.ipynb`: This notebook contains the training of the Multilabel Classification (MC) Random Forest and the analysis and plots performed in its section of the paper.
  - `05_SC_RF.ipynb`: This notebook contains the Specialized Classification (SC) Random Forest, which is the result of the ensemble of the specialized RFs, and includes the analysis and plots performed in its section of the paper.
  - `05_realistic.ipynb`: This notebook contains the analysis of the realistic case, using the data contained in the `spec_earth/` folder.
