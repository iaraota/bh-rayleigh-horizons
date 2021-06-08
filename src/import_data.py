import glob
import json
from pathlib import Path

from scipy import interpolate, integrate
import numpy as np

import constants as cte


class ImportDetector:
    """Import detector self.noise power spectral density.
    """

    def __init__(self, detector: str):
        """Import detector self.noise power spectral density.
        """
        self.import_detector(detector)

    def import_detector(self, detector: str, detector_folder=str(Path(__file__).parent.absolute()) + '/../../detectors'):
        """Import detector self.noise power specrtal density

        Parameters
        ----------
        detector : str
            Detector name 'LIGO', 'LISA', 'CE' = 'CE2silicon', 'CE2silica' or 'ET'
        detector_folder : str
            FOlder containing detectors files

        Raises
        ------
        ValueError
            If chosen detector is not {'LIGO', 'LISA', 'CE' = 'CE2silicon', 'CE2silica' or 'ET'}
            raises value error.

        """
        # choose detector
        self.noise = {}
        i_freq = 0
        i_psd = 1
        if detector == "LIGO":
            file_name = "aLIGODesign.txt"
            self.noise["label"] = "LIGO - Design sensitivity"
        elif detector == "LISA":
            file_name = "LISA_Strain_Sensitivity_range.txt"
            self.noise["label"] = "LISA sensitivity"
        elif detector == "ET":
            i_psd = 3
            file_name = "ET/ETDSensitivityCurve.txt"
            self.noise["label"] = "ET_D sum sensitivity"
        elif detector == "CE" or detector == "CE2silicon":
            file_name = "CE/CE2silicon.txt"
            self.noise["label"] = "CE silicon sensitivity"
        elif detector == "CE2silica":
            file_name = "CE/CE2silica.txt"
            self.noise["label"] = "CE silica sensitivity"
        else:
            raise ValueError(
                "Wrong detector option! Choose \"LIGO\", \"LISA\", \"CE\" = \"CE2silicon\", \"CE2silica\" or \"ET\"")

        # import detector psd
        file_path = detector_folder + '/' + file_name
        self.noise_file = np.genfromtxt(file_path)
        self.noise["freq"], self.noise["psd"] = self.noise_file[:,
                                                                i_freq], self.noise_file[:, i_psd]

        # make self.noise arrays immutable arrays
        self.noise["freq"].flags.writeable = False
        self.noise["psd"].flags.writeable = False

        # interpolate psd
        self.itp = interpolate.interp1d(
            self.noise["freq"], self.noise["psd"], "cubic")


class ImportQNMParameters:
    """Import fitted QNM parameters from SXS simulations.
    """

    def __init__(self, q_mass, simulation_folder=str(Path(__file__).parent.absolute()) + '/../../simulations'):
        """Initiate class

        Parameters
        ----------
        q_mass : float, string
            Binary black hole mass ratio. q >= 1.
        """
        self.import_simulation_qnm_parameters(q_mass, simulation_folder)

    def import_simulation_qnm_parameters(self, q_mass, simulation_folder: str):
        """Import QNM fits from simulations contained in
        the folder '{simulation_folder}{q_mass}*'

        Parameters
        ----------
        q_mass : float, string
            Binary black hole mass ratio. q >= 1.
        simulation_folder : str
            Folder containing simulation fitted files.
        """

        # find simulation folder for the chosen mass ratio
        q_mass = str(q_mass)
        if float(q_mass) < 10:
            q_mass = '0' + q_mass
        simulation_folder = glob.glob(f'{simulation_folder}/{q_mass}*')[0]

        # import fitted parameters
        with open(simulation_folder + '/data/qnm_pars/ratios.json') as file:
            self.ratios = json.load(file)

        with open(simulation_folder + '/data/qnm_pars/amplitudes.json') as file:
            self.amplitudes = json.load(file)

        with open(simulation_folder + '/data/qnm_pars/phases.json') as file:
            self.phases = json.load(file)

        with open(simulation_folder + '/data/qnm_pars/omegas.json') as file:
            self.omegas = json.load(file)

        with open(simulation_folder + '/data/qnm_pars/dphi.json') as file:
            self.dphases = json.load(file)

        for (key, value) in self.amplitudes.items():
            self.amplitudes[key] = abs(self.amplitudes[key])

        self.omegas["(2,2,1) I"] = self.omegas["(2,2,1)"]
        self.omegas["(2,2,1) II"] = self.omegas["(2,2,1)"]

        # final mass
        with open(simulation_folder + '/data/qnm_pars/bh_pars.json') as file:
            self.bh_pars = json.load(file)


def luminosity_distance(redshift: float):
    """
    Compute luminosity distance as function of the redshift

    Parameters
    ----------
        redshift: scalar
            Cosmological redshift value

    Returns
    -------
        scalar: Returns luminosity distance relative to given redshift
    """

    # cosmological constants
    # values from https://arxiv.org/pdf/1807.06209.pdf
    h = 0.6796
    H_0 = h * 100 * 1e+3  # Huble constant m s**-1 Mpc**-1
    clight = 2.99792458e8  # speed of lightm s**-1
    Dist_H = clight / H_0  # Huble distance

    Omega_M = 0.315
    Omega_Λ = 1 - Omega_M
    Omega_K = 0.0

    def Ez(z): return 1 / np.sqrt(Omega_M * (1 + z)
                                  ** 3 + Omega_K * (1 + z)**2 + Omega_Λ)
    Dist_C = Dist_H * integrate.quad(Ez, 0, redshift)[0]
    Dist_L = (1 + redshift) * Dist_C
    """#= If Omega_K was not 0
    if Omega_K > 0
        Dist_M = Dist_H*sinh(sqrt(Omega_K)*Dist_C/Dist_H)/sqrt(Omega_K)
    elif Omega_K == 0.0
        Dist_M = Dist_C
    elif Omega_K < 0
        Dist_M = Dist_H*np.sin(sqrt(Omega_K)*Dist_C/Dist_H)/sqrt(Omega_K)
    Dist_A = Dist_M/(1+redshift)
    Dist_L = (1+redshift)*Dist_M
    """
    return Dist_L


def convert_units(Mass_QNM: float, redshift: float, mass_f=1):
    """ Compute factors that converts times and frequencies of the QNM
        according to the BH mass and redshift and waveform amplitude factor

    Parameters
    ----------
    Mass_QNM : scalar
        Final black hole mass in source frame
    redshift : scalar
        Redshift of the source
    mass_f : scalar, optional
        Factor of the final black hole mass relative to the total mass
        of the binary (it is given by the BBH simulation), by default 1

    Returns
    -------
    array_like
        Returns time and amplitude conversion factors
    """

    # Source parameters
    M_final = (1 + redshift) * Mass_QNM  # QNM mass in detector frame
    M_total = M_final / mass_f  # Binary total mass (m1+m2)
    d_L = luminosity_distance(redshift)
    time_unit = (M_total) * cte.tSun
    strain_unit = ((M_final) * cte.tSun) / \
        (d_L * cte.Dist)

    return time_unit, strain_unit
