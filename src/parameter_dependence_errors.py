from pathlib import Path
from multiprocessing import Pool
from itertools import combinations

import numpy as np
from tqdm import tqdm  # progress bar

# import module of functions to compute the Fisher Matrix of QNMs
import fisher_matrix_elements_two_modes as fme
from import_data import convert_units, ImportQNMParameters, ImportDetector
from compute_snr import compute_SRN


def run_all_detectors_two_modes():
    mode_0 = '(2,2,0)'
    mode_1 = '(2,2,1) II'
    final_mass = 100
    redshift = 0.1
    qs = [1.5, 10]
    for mass_ratio in qs:
        compute_error(final_mass, redshift, mode_0, mode_1, mass_ratio)

def compute_error(final_mass, redshift, mode_0, mode_1, mass_ratio):

    antenna_plus = np.sqrt(1 / 5 / 4 / np.pi)
    antenna_cross = antenna_plus

    compute_errors_amplitude_ratio(
        final_mass, redshift, antenna_plus, antenna_cross, mode_0, mode_1, 'LIGO', mass_ratio)


def compute_errors_amplitude_ratio(final_mass: float, redshift: float, antenna_plus: float, antenna_cross: float, mode_0: str, mode_1: str, detector: str, mass_ratio: float):
    """Compute and save the QNMs parameters errors using Fisher Matrix formalism.
    The singal is assumed to be a sum of two QNMs.

    Parameters
    ----------
    final_mass : float
        Mass of the remmnant black hole
    redshift : float
        Cosmological redshift of the source
    antenna_plus : float
        Antenna pattern response for the plus polarization, include
        also the spheroidal harmonic value.
    antenna_cross : float
        Antenna pattern response for the cross polarization, include
        also the spheroidal harmonic value.
    mode_0 : str
        First QNM considered in the signal. Must be {'(2,2,0)',
        (2,2,1) I', '(2,2,1) II', (3,3,0), (4,4,0), (2,1,0)}.
    mode_1 : str
        Second QNM considered in the signal. Must be {'(2,2,0)',
        (2,2,1) I', '(2,2,1) II', (3,3,0), (4,4,0), (2,1,0)}. Must
        be different from mode_0.
    detector : str
        Gravitational wave detector name. Must be {'LIGO', 'LISA',
        'CE' = 'CE2silicon', 'CE2silica', 'ET'}.
    mass_ratio : float
        Binary black hole mass ratio. mass_ratio >= 1. This is used to
        determine the QNM parameters.
    """
    noise = ImportDetector(detector)
    qnm_pars = ImportQNMParameters(mass_ratio)
    time_unit, strain_unit = convert_units(
        final_mass, redshift, qnm_pars.bh_pars['remnant_mass'])

    # Compute QNM frequency and damping time according to the source
    freq, tau = {}, {}
    for (mode, omega) in qnm_pars.omegas.items():
        freq[mode] = omega['omega_r'] / 2 / np.pi / time_unit
        tau[mode] = time_unit / omega['omega_i']

    delta_tau = abs(tau[mode_0] - tau[mode_1])

    # Create qnm parameters dictionary

    qnm_parameters = {
        'freq_array': noise.noise['freq'],
        'A': qnm_pars.amplitudes[mode_0],
        'phi_0': qnm_pars.phases[mode_0],
        'f_0': freq[mode_0],
        'tau_0': tau[mode_0],
        'R': qnm_pars.amplitudes[mode_1] / qnm_pars.amplitudes[mode_0],
        'phi_1': qnm_pars.phases[mode_1],
        'f_1': freq[mode_1],
        'tau_1': tau[mode_1],
    }

    # find src folder path
    src_path = str(Path(__file__).parent.absolute())

    # create data folder if it doesn't exist
    data_path = src_path + '/../data/parameters_dependence_errors'
    Path(data_path).mkdir(parents=True, exist_ok=True)

    file_rayleigh_criterion = f'{data_path}/{detector}_q_{mass_ratio}_amplitude_and_ratio.dat'

    values = [(
        final_mass,
        redshift,
        freq,
        tau,
        mode_0,
        mode_1,
        strain_unit,
        antenna_plus,
        antenna_cross,
        {
            'freq_array': noise.noise['freq'],
            'A': A_factor * qnm_pars.amplitudes[mode_0],
            'phi_0': qnm_pars.phases[mode_0],
            'f_0': freq[mode_0],
            'tau_0': tau[mode_0],
            'R': R_factor * qnm_pars.amplitudes[mode_1] / qnm_pars.amplitudes[mode_0],
            'phi_1': qnm_pars.phases[mode_1],
            'f_1': freq[mode_1],
            'tau_1': tau[mode_1],
        },
        noise.noise,
        file_rayleigh_criterion)
        for A_factor in np.linspace(0.5, 2, num=30) for R_factor in np.linspace(0.5, 2, num=30)
    ]

    cores = 6
    with Pool(processes=cores) as pool:
        res = pool.starmap(save_errors_A_R, tqdm(values, total=len(values)))

    # for A_factor in np.linspace(0.1, 10, num=100):
    #     qnm_parameters['A'] = qnm_pars.amplitudes[mode_0] * A_factor
    #     for R_factor in tqdm(np.linspace(0.1, 10, num=100)):
    #         qnm_parameters['R'] = R_factor * \
    #             qnm_pars.amplitudes[mode_1] / qnm_pars.amplitudes[mode_0],
    #         # Compute Fisher Matrix errors
    #         sigma = compute_two_modes_fisher_matrix(
    #             strain_unit, antenna_plus, antenna_cross, qnm_parameters, noise.noise)

    #         # save error
    #         file_rayleigh_criterion = f'{data_path}/{detector}_q_{mass_ratio}_amplitude_and_ratio.dat'
    #         if not Path(file_rayleigh_criterion).is_file():
    #             with open(file_rayleigh_criterion, 'w') as file:
    #                 file.write(f'#(0)mass(1)redshift(2)mode_0(3)mode_1')
    #                 file.write(f'(4)freq_0(5)sigma_freq_0')
    #                 file.write(f'(6)freq_1(7)sigma_freq_1')
    #                 file.write(f'(8)tau_0(9)sigma_tau_0')
    #                 file.write(f'(10)tau_1(11)sigma_tau_1')
    #                 file.write(f'(6)amplitude(7)ratio\n')
    #         with open(file_rayleigh_criterion, 'a') as file:
    #             file.write(
    #                 f'{final_mass}\t{redshift}\t"{mode_0}"\t"{mode_1}"\t')
    #             file.write(f"{freq[mode_0]}\t{sigma['f_0']}\t")
    #             file.write(f"{freq[mode_1]}\t{sigma['f_1']}\t")
    #             file.write(f"{tau[mode_0]}\t{sigma['tau_0']}\t")
    #             file.write(f"{tau[mode_1]}\t{sigma['tau_1']}\t")
    #             file.write(f"{qnm_parameters['A']}\t{qnm_parameters['R']}\n")

    # return 0


def save_errors_A_R(final_mass: float, redshift: float, freq, tau, mode_0, mode_1, strain_unit: float, antenna_plus: float, antenna_cross: float, qnm_parameters: dict, noise: dict, file_rayleigh_criterion):
    # Compute Fisher Matrix errors
    sigma = compute_two_modes_fisher_matrix(
        strain_unit, antenna_plus, antenna_cross, qnm_parameters, noise)

    # save error
    if not Path(file_rayleigh_criterion).is_file():
        with open(file_rayleigh_criterion, 'w') as file:
            file.write(f'#(0)mass(1)redshift(2)mode_0(3)mode_1')
            file.write(f'(4)freq_0(5)sigma_freq_0')
            file.write(f'(6)freq_1(7)sigma_freq_1')
            file.write(f'(8)tau_0(9)sigma_tau_0')
            file.write(f'(10)tau_1(11)sigma_tau_1')
            file.write(f'(6)amplitude(7)ratio\n')
    with open(file_rayleigh_criterion, 'a') as file:
        file.write(
            f'{final_mass}\t{redshift}\t"{mode_0}"\t"{mode_1}"\t')
        file.write(f"{freq[mode_0]}\t{sigma['f_0']}\t")
        file.write(f"{freq[mode_1]}\t{sigma['f_1']}\t")
        file.write(f"{tau[mode_0]}\t{sigma['tau_0']}\t")
        file.write(f"{tau[mode_1]}\t{sigma['tau_1']}\t")
        file.write(f"{qnm_parameters['A']}\t{qnm_parameters['R']}\n")

    return 0


def compute_two_modes_fisher_matrix(global_amplitude: float, antenna_plus: float, antenna_cross: float, qnm_pars: dict, noise: dict):
    """Compute the QNM parameter errors of a signal containing
    two QNMs using Fisher Matrix formalism.

    Parameters
    ----------
    global_amplitude : float
        Global physical amplitude of the signal. M/D_L
    antenna_plus : float
        Antenna pattern response for the plus polarization, include
        also the spheroidal harmonic value.
    antenna_cross : float
        Antenna pattern response for the coss polarization, include
        also the spheroidal harmonic value.
    qnm_pars : dict
        Dictionary with the QNM parameters of the both modes:
            'A': global amplitude
            'phi_0': 1st mode phase
            'f_0': 1st mode frequency
            'tau_0': 1st mode decay time
            'R': amplitude ratio
            'phi_1': 2nd mode phase
            'f_1': 2nd mode frequency
            'tau_1': 2nd mode decay time
    noise : dict
        Dictionary containing:
            'freq': noise frequency sampling
            'psd': square root of the noise spectral density
                (it will be squared in the Fisher Matrix computation)

    Returns
    -------
    dict
        Returns sigma containing the estimated errors for both modes
        parameters
            'A': global amplitude
            'phi_0': 1st mode phase
            'f_0': 1st mode frequency
            'tau_0': 1st mode decay time
            'R': amplitude ratio
            'phi_1': 2nd mode phase
            'f_1': 2nd mode frequency
            'tau_1': 2nd mode decay time
    """
    # convert noise to numpy arrays
    noise['freq'] = np.asanyarray(noise['freq'])
    noise['psd'] = np.asanyarray(noise['psd'])
    qnm_pars['freq_array'] = noise['freq']
    # create a 8x8 matrix filled with zeros
    fisher_matrix = np.zeros([8, 8])

    # sort the parameters to compute the Fisher Matrix
    parameters_list = [
        'A',
        'phi_0',
        'f_0',
        'tau_0',
        'R',
        'phi_1',
        'f_1',
        'tau_1',
    ]

    for i in range(0, 8):
        for j in range(0, 8):
            fisher_matrix[i, j] = fme.inner_product(noise['freq'],
                                                    global_amplitude * (
                antenna_plus *
                (fme.choose_polarization_and_parameter(
                    'real', parameters_list[i])(qnm_pars))
                + antenna_cross * (fme.choose_polarization_and_parameter('imag', parameters_list[i])(qnm_pars))),
                global_amplitude * (
                    antenna_plus *
                (fme.choose_polarization_and_parameter(
                    'real', parameters_list[j])(qnm_pars))
                    + antenna_cross * (fme.choose_polarization_and_parameter('imag', parameters_list[j])(qnm_pars))),
                noise['psd']
            )

    if np.linalg.det(fisher_matrix) == 0:
        correlation_matrix = np.zeros([8, 8])
        correlation_matrix.fill(np.inf)
    else:
        correlation_matrix = np.linalg.inv(fisher_matrix)

    sigma = {
        'A':    np.sqrt(np.abs(correlation_matrix[0, 0])),
        'phi_0':  np.sqrt(np.abs(correlation_matrix[1, 1])),
        'f_0':    np.sqrt(np.abs(correlation_matrix[2, 2])),
        'tau_0':  np.sqrt(np.abs(correlation_matrix[3, 3])),
        'R':    np.sqrt(np.abs(correlation_matrix[4, 4])),
        'phi_1':  np.sqrt(np.abs(correlation_matrix[5, 5])),
        'f_1':    np.sqrt(np.abs(correlation_matrix[6, 6])),
        'tau_1':  np.sqrt(np.abs(correlation_matrix[7, 7])),
    }

    return sigma


if __name__ == '__main__':
    run_all_detectors_two_modes()
