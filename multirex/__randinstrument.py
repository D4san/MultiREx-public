#########################################
#  __  __      _ _   _ ___ ___          #
# |  \/  |_  _| | |_(_) _ \ __|_ __     #
# | |\/| | || | |  _| |   / _|\ \ /     #
# |_|  |_|\_,_|_|\__|_|_|_\___/_\_\     #
# Planetary spectra generator           #
#########################################

from taurex.instruments import Instrument
import numpy as np
import math

class RndSNRInstrument(Instrument):
    """Simple instrument model that, for a given
    wavelength-independant, signal-to-noise ratio,
    compute resulting noise from it.

    Parameters
    ----------

    SNR: float
        Signal-to-noise ratio

    binner: :class:`~taurex.binning.binner.Binner`, optional
        Optional resampler to generate a new spectral
        grid.


    """
    def __init__(self, SNR=10, binner=None):
        super().__init__()

        self._binner = binner
        self._SNR = SNR

    def model_noise(self, model, model_res=None, num_observations=1):

        if model_res is None:
            model_res = model.model()

        binner = self._binner
        if binner is None:
            binner = model.defaultBinner()

        wngrid, spectrum, error, grid_width = self._binner.bin_model(model_res)

        signal = spectrum.max() - spectrum.min()

        noise = np.ones(spectrum.shape)*signal/self._SNR

        specnoisi= spectrum+np.random.normal(0,\
            noise/ math.sqrt(num_observations), noise.shape)
        return wngrid, specnoisi, \
            noise / math.sqrt(num_observations)

    @classmethod
    def input_keywords(cls):
        return ['snr', 'SNR', 'signal-noise-ratio']
