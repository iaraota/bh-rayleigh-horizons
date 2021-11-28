from pathlib import Path
from multiprocessing import Pool
from itertools import combinations

import numpy as np
from tqdm import tqdm  # progress bar

# import module of functions to compute the Fisher Matrix of QNMs
import fisher_matrix_elements_two_modes as fme
from import_data import convert_units, ImportQNMParameters, ImportDetector
from compute_snr import compute_SRN, compute_SRN_2modes


def run_all_detectors_two_modes():
    detectors = ['LIGO', 'LISA', 'CE', 'ET']
    modes = ['(2,2,0)', '(2,2,1) II', '(3,3,0)', '(4,4,0)', '(2,1,0)']
    qs = [1.5, 10]
    for q in qs:
        for detector in detectors:
            for comb in list(combinations(modes, 2)):
                print(q, detector, comb)
                compute_rayleigh_criterion_all_masses_and_redshifts(
                    comb[0], comb[1], q, detector)


def run_all_detectors_two_modes_fundamental():
    detectors = ['LISA']#['LIGO', 'LISA', 'CE', 'ET']
    modes = ['(2,2,1) II']#, '(3,3,0)', '(4,4,0)', '(2,1,0)']
    qs = [1.5]#, 10]
    for q in qs:
        for detector in detectors:
            for mode in modes:
                comb = ['(2,2,0)'] + [mode]
                print(q, detector, comb)
                compute_rayleigh_criterion_all_masses_and_redshifts(
                    comb[0], comb[1], q, detector)


def compute_rayleigh_criterion_all_masses_and_redshifts(mode_0: str, mode_1: str, mass_ratio: float, detector: str, cores=6):
    """Compute and save the QNMs parameters errors using Fisher Matrix formalism
    for all mass and redshift in the detector range. The singal is assumed to be
    a sum of two QNMs.

    Parameters
    ----------
    mode_0 : str
        First QNM considered in the signal. Must be {'(2,2,0)',
        (2,2,1) I', '(2,2,1) II', (3,3,0), (4,4,0), (2,1,0)}. Must
        be different from mode_0.
    mode_1 : str
        Second QNM considered in the signal. Must be {'(2,2,0)',
        (2,2,1) I', '(2,2,1) II', (3,3,0), (4,4,0), (2,1,0)}. Must
        be different from mode_0.
    mass_ratio : float
        Binary black hole mass ratio. mass_ratio >= 1. This is used to
        determine the QNM parameters.
    detector : str
        Gravitational wave detector name. Must be {'LIGO', 'LISA',
        'CE' = 'CE2silicon', 'CE2silica', 'ET'}.
    cores : int, optional
        Number of cores used for parallel computation. Default: 6.
    """
    N_points = 100  # number of points per chunck
    masses = {
        'LIGO': np.logspace(start=1, stop=4, num=int(N_points * 3), base=10),
        'CE':   np.logspace(start=1, stop=5, num=int(N_points * 4), base=10),
        'ET':   np.logspace(start=1, stop=5, num=int(N_points * 4), base=10),
        'LISA': np.logspace(start=4, stop=9, num=int(N_points * 5.0), base=10),
    }
    redshifts = {
        'LIGO': np.logspace(start=-4.05, stop=0, num=int(N_points * 4), base=10),
        'CE':   np.logspace(start=-2.05, stop=1.05, num=int(N_points * 3), base=10),
        'ET':   np.logspace(start=-2.05, stop=1.05, num=int(N_points * 3), base=10),
        'LISA': np.logspace(start=-2.05, stop=2.05, num=int(N_points * 5), base=10),
    }
    # angle average of antenna patter response times angle average of spheroidal harmonics
    antenna_plus = {
        'LIGO': np.sqrt(1 / 5 / 4 / np.pi),
        'CE':   np.sqrt(1 / 5 / 4 / np.pi),
        'ET':   np.sqrt(1 / 5 / 4 / np.pi) * 3 / 2,
        'LISA': np.sqrt(1 / 4 / np.pi),
    }
    antenna_cross = antenna_plus
    values = [(
        mass,
        redshift,
        antenna_plus[detector],
        antenna_cross[detector],
        mode_0,
        mode_1,
        detector,
        mass_ratio,
    )
        for mass in masses[detector] for redshift in redshifts[detector]
    ]

    with Pool(processes=cores) as pool:
        res = pool.starmap(compute_two_modes_rayleigh_criterion,
                           tqdm(values, total=len(values)))


def compute_two_modes_rayleigh_criterion(final_mass: float, redshift: float, antenna_plus: float, antenna_cross: float, mode_0: str, mode_1: str, detector: str, mass_ratio: float):
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

    # Compute difference between modes parameters
    delta_freq = abs(freq[mode_0] - freq[mode_1])
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

    # Create qnm parameters dictionary for SNR
    qnm_pars_snr = {}
    for mode in [mode_0, mode_1]:
        qnm_pars_snr[mode] = {
            'freq_array': noise.noise['freq'],
            'A_lmn': qnm_pars.amplitudes[mode],
            'phi_lmn': qnm_pars.phases[mode],
            'f_lmn': freq[mode],
            'tau_lmn': tau[mode],
        }

    # Compute Fisher Matrix errors
    sigma = compute_two_modes_fisher_matrix(
        strain_unit, antenna_plus, antenna_cross, qnm_parameters, noise.noise)
    snr_1 = compute_SRN(strain_unit, antenna_plus,
                        antenna_cross, qnm_pars_snr[mode_0], noise.noise)
    snr_2 = compute_SRN(strain_unit, antenna_plus,
                        antenna_cross, qnm_pars_snr[mode_1], noise.noise)
    snr_both = compute_SRN_2modes(strain_unit, antenna_plus, antenna_cross,
                                  qnm_pars_snr[mode_0], qnm_pars_snr[mode_1], noise.noise)

    # find src folder path
    src_path = str(Path(__file__).parent.absolute())

    # create data folder if it doesn't exist
    data_path = src_path + '/../data'
    Path(data_path).mkdir(parents=True, exist_ok=True)
    Path(data_path + '/all_errors').mkdir(parents=True, exist_ok=True)
    Path(data_path + '/rayleigh_criterion').mkdir(parents=True, exist_ok=True)

    # save everything
    file_all_errors = f'{data_path}/all_errors/{detector}_q_{mass_ratio}_all_errors.dat'
    if not Path(file_all_errors).is_file():
        with open(file_all_errors, 'w') as file:
            file.write('#(0)mass(1)redshift(2)mode_0(3)mode_1')
            file.write('(4)freq_mode_0(5)freq_mode_1(6)tau_mode_0(7)tau_mode_0')
            file.write('(8)SNR_mode_0(9)SNR_mode_1')
            file.write('(10)error_A(11)error_phi_mode_0')
            file.write('(12)error_f_mode_0(13)error_tau_mode_0')
            file.write('(14)error_R(15)error_phi_mode_1')
            file.write('(16)error_f_mode_1(17)error_tau_mode_1')
            file.write('(18)SNR_ringdown\n')
    with open(file_all_errors, 'a') as file:
        file.write(f'{final_mass}\t{redshift}\t"{mode_0}"\t"{mode_1}"\t')
        file.write(f'{freq[mode_0]}\t{freq[mode_1]}\t')
        file.write(f'{tau[mode_0]}\t{tau[mode_1]}\t')
        file.write(f'{snr_1}\t{snr_2}\t')
        file.write(f"{sigma['A']}\t{sigma['phi_0']}\t")
        file.write(f"{sigma['f_0']}\t{sigma['tau_0']}\t")
        file.write(f"{sigma['R']}\t{sigma['phi_1']}\t")
        file.write(f"{sigma['f_1']}\t{sigma['tau_1']}\t")
        file.write(f"{snr_both}\n")

    # save rayleigh criterion
    file_rayleigh_criterion = f'{data_path}/rayleigh_criterion/{detector}_q_{mass_ratio}_rayleigh_criterion.dat'
    if not Path(file_rayleigh_criterion).is_file():
        with open(file_rayleigh_criterion, 'w') as file:
            file.write(f'#(0)mass(1)redshift(2)mode_0(3)mode_1')
            file.write(f'(4)delta_freq(5)sigma_freq_mode_0(6)sigma_freq_mode_1')
            file.write(f'(7)delta_tau(8)sigma_tau_mode_0(9)sigma_tau_mode_1')
            file.write(f'(6)SNR_mode_0(7)SNRmode_1(8)SNR_ringdown\n')
    with open(file_rayleigh_criterion, 'a') as file:
        file.write(f'{final_mass}\t{redshift}\t"{mode_0}"\t"{mode_1}"\t')
        file.write(f"{delta_freq}\t")
        file.write(f"{sigma['f_0']}\t{sigma['f_1']}\t")
        file.write(f'{delta_tau}\t')
        file.write(f"{sigma['tau_0']}\t{sigma['tau_1']}\t")
        file.write(f'{snr_1}\t{snr_2}\t{snr_both}\n')

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
    run_all_detectors_two_modes_fundamental()
