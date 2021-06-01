import glob
import json

from scipy import interpolate
import numpy as np

class ImportDetector:
    """Import detector self.noise power spectral density.
    """

    def __init__(self):
        """Import detector self.noise power spectral density.
        """


    def import_detector(self, detector:str, detector_folder='../detectors'):
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
            raise ValueError("Wrong detector option! Choose \"LIGO\", \"LISA\", \"CE\" = \"CE2silicon\", \"CE2silica\" or \"ET\"")

        # import detector psd
        file_path = detector_folder+'/'+file_name
        self.noise_file = np.genfromtxt(file_path)
        self.noise["freq"], self.noise["psd"] = self.noise_file[:,i_freq], self.noise_file[:,i_psd]

        # make self.noise arrays immutable arrays
        self.noise["freq"].flags.writeable = False
        self.noise["psd"].flags.writeable = False

        # interpolate psd
        self.itp = interpolate.interp1d(self.noise["freq"], self.noise["psd"], "cubic")


class ImportQNMParameters:
    """Import fitted QNM parameters from SXS simulations.
    """

    def __init__(self, q_mass, simulation_folder = '../simulations'):
        """Initiate class

        Parameters
        ----------
        q_mass : float, string
            Binary black hole mass ratio. q >= 1.
        """
        self.import_simulation_qnm_parameters(q_mass)


    def import_simulation_qnm_parameters(self, q_mass, simulation_folder:str):
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
            q_mass = '0'+q_mass
        simulation_folder = glob.glob(f'{simulation_folder}/{q_mass}*')

        # import fitted parameters
        with open(simulation_folder+'/data/qnm_pars/ratios.json') as file:
            self.ratios = json.load(file)

        with open(simulation_folder+'/data/qnm_pars/amplitudes.json') as file:
            self.amplitudes = json.load(file)

        with open(simulation_folder+'/data/qnm_pars/phases.json') as file:
            self.phases = json.load(file)

        with open(simulation_folder+'/data/qnm_pars/omegas.json') as file:
            self.omegas = json.load(file)

        with open(simulation_folder+'/data/qnm_pars/dphi.json') as file:
            self.dphases = json.load(file)


        for (key, value) in amplitudes:
            self.amplitudes[key] = abs(self.amplitudes[key])


        self.omegas["(2,2,1) I"]  = self.omegas["(2,2,1)"]
        self.omegas["(2,2,1) II"]  = self.omegas["(2,2,1)"]

        # final mass
        with open(simulation_folder+'/data/qnm_pars/bh_pars.json') as file:
            self.bh_pars = json.load(file)
