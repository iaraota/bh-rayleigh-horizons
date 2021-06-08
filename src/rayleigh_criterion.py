from pathlib import Path
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm  # progress bar

# import module of functions to compute the Fisher Matrix of QNMs
import fisher_matrix_elements as fme
from import_data import convert_units, ImportQNMParameters, ImportDetector
from compute_snr import compute_SRN


def compute_rayleigh_criterion_all_masses_and_redshifts(mode_1: str, mode_2: str, mass_ratio: float, detector: str, cores=6):
    """Compute and save the QNMs parameters errors using Fisher Matrix formalism
    for all mass and redshift in the detector range. The singal is assumed to be
    a sum of two QNMs.

    Parameters
    ----------
    mode_1 : str
        First QNM considered in the signal. Must be {'(2,2,0)',
        (2,2,1) I', '(2,2,1) II', (3,3,0), (4,4,0), (2,1,0)}. Must
        be different from mode_1.
    mode_2 : str
        Second QNM considered in the signal. Must be {'(2,2,0)',
        (2,2,1) I', '(2,2,1) II', (3,3,0), (4,4,0), (2,1,0)}. Must
        be different from mode_1.
    mass_ratio : float
        Binary black hole mass ratio. mass_ratio >= 1. This is used to
        determine the QNM parameters.
    detector : str
        Gravitational wave detector name. Must be {'LIGO', 'LISA',
        'CE' = 'CE2silicon', 'CE2silica', 'ET'}.
    cores : int, optional
        Number of cores used for parallel computation. Default: 6.
    """
    N_points = 30  # number of points per chunck
    masses = {
        'LIGO': np.logspace(start=1, stop=4, num=int(N_points * 3), base=10),
        'CE':   np.logspace(start=1, stop=5, num=int(N_points * 4), base=10),
        'ET':   np.logspace(start=1, stop=5, num=int(N_points * 4), base=10),
        'LISA': np.logspace(start=4, stop=9, num=int(N_points * 5.0), base=10),
    }
    redshifts = {
        'LIGO': np.logspace(start=-2, stop=0, num=int(N_points * 2), base=10),
        'CE':   np.logspace(start=-2, stop=1, num=int(N_points * 3), base=10),
        'ET':   np.logspace(start=-2, stop=1, num=int(N_points * 3), base=10),
        'LISA': np.logspace(start=-2, stop=1, num=int(N_points * 5), base=10),
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
        mode_1,
        mode_2,
        detector,
        mass_ratio,
    )
        for mass in masses[detector] for redshift in redshifts[detector]
    ]

    with Pool(processes=cores) as pool:
        res = pool.starmap(compute_two_modes_rayleigh_criterion,
                           tqdm(values, total=len(values)))


def compute_two_modes_rayleigh_criterion(final_mass: float, redshift: float, antenna_plus: float, antenna_cross: float, mode_1: str, mode_2: str, detector: str, mass_ratio: float):
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
    mode_1 : str
        First QNM considered in the signal. Must be {'(2,2,0)',
        (2,2,1) I', '(2,2,1) II', (3,3,0), (4,4,0), (2,1,0)}.
    mode_2 : str
        Second QNM considered in the signal. Must be {'(2,2,0)',
        (2,2,1) I', '(2,2,1) II', (3,3,0), (4,4,0), (2,1,0)}. Must
        be different from mode_1.
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
    delta_freq = abs(freq[mode_1] - freq[mode_2])
    delta_tau = abs(tau[mode_1] - tau[mode_2])

    # Create qnm parameters dictionary
    qnm_parameters = {}
    for mode in [mode_1, mode_2]:
        qnm_parameters[mode] = {
            'freq_array': noise.noise['freq'],
            'A_lmn': qnm_pars.amplitudes[mode],
            'phi_lmn': qnm_pars.phases[mode],
            'f_lmn': freq[mode],
            'tau_lmn': tau[mode],
        }

    # Compute Fisher Matrix errors
    sigma_1, sigma_2 = compute_two_modes_fisher_matrix(
        strain_unit, antenna_plus, antenna_cross, qnm_parameters[mode_1], qnm_parameters[mode_2], noise.noise)
    snr_1 = compute_SRN(strain_unit, antenna_plus,
                        antenna_cross, qnm_parameters[mode_1], noise.noise)
    snr_2 = compute_SRN(strain_unit, antenna_plus,
                        antenna_cross, qnm_parameters[mode_2], noise.noise)

    # find src folder path
    src_path = str(Path(__file__).parent.absolute())

    # create data folder if it doesn't exist
    data_path = src_path + '/../data'
    Path(data_path).mkdir(parents=True, exist_ok=True)
    Path(data_path + '/all_errors').mkdir(parents=True, exist_ok=True)
    Path(data_path + '/rayleigh_criterion').mkdir(parents=True, exist_ok=True)

    # save everything
    file_all_errors = f'{data_path}/all_errors/{detector}_q_{mass_ratio}_mode1_{mode_1[1]+mode_1[3]+mode_1[5]}_mode2_{mode_2[1]+mode_2[3]+mode_2[5]}_all_errors.dat'
    if not Path(file_all_errors).is_file():
        with open(file_all_errors, 'w') as file:
            file.write(
                f'#(0)mass(1)redshift(2)freq-{mode_1}(3)freq-{mode_2}(4)tau-{mode_1}(5)tau-{mode_2}')
            file.write(f'(6)SNR-{mode_1}(7)SNR-{mode_2}')
            file.write(
                f'(8)error_A-{mode_1}(9)error_phi-{mode_1}(10)error_f-{mode_1}(11)error_tau-{mode_1}')
            file.write(
                f'(12)error_A-{mode_2}(13)error_phi-{mode_2}(14)error_f-{mode_2}(15)error_tau-{mode_2}\n')
    with open(file_all_errors, 'a') as file:
        file.write(
            f'{final_mass}\t{redshift}\t{freq[mode_1]}\t{freq[mode_2]}\t{tau[mode_1]}\t{tau[mode_2]}\t')
        file.write(f'{snr_1}\t{snr_2}\t')
        file.write(
            f"{sigma_1['A_lmn']}\t{sigma_1['phi_lmn']}\t{sigma_1['f_lmn']}\t{sigma_1['tau_lmn']}\t")
        file.write(
            f"{sigma_2['A_lmn']}\t{sigma_2['phi_lmn']}\t{sigma_2['f_lmn']}\t{sigma_2['tau_lmn']}\n")

    # save rayleigh criterion
    file_rayleigh_criterion = f'{data_path}/rayleigh_criterion/{detector}_q_{mass_ratio}_mode1_{mode_1[1]+mode_1[3]+mode_1[5]}_mode2_{mode_2[1]+mode_2[3]+mode_2[5]}_rayleigh_criterion.dat'
    if not Path(file_rayleigh_criterion).is_file():
        with open(file_rayleigh_criterion, 'w') as file:
            file.write(f'#(0)mass(1)redshift(2)delta_freq(3)max(sigma_freq)')
            file.write(
                f'(4)delta_tau(5)max(sigma_tau)(6)SNR-{mode_1}(7)SNR-{mode_2}\n')
    with open(file_rayleigh_criterion, 'a') as file:
        file.write(f'{final_mass}\t{redshift}\t')
        file.write(f"{delta_freq}\t")
        file.write(f"{max(sigma_1['f_lmn'], sigma_2['f_lmn'])}\t")
        file.write(f'{delta_tau}\t')
        file.write(f"{max(sigma_1['tau_lmn'], sigma_2['tau_lmn'])}\t")
        file.write(f'{snr_1}\t{snr_2}\n')

    return 0


def compute_two_modes_fisher_matrix(global_amplitude: float, antenna_plus: float, antenna_cross: float, qnm_pars_1: dict, qnm_pars_2: dict, noise: dict):
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
    qnm_pars_1 : dict
        Dictionary with the QNM parameters of the first mode:
            'A_lmn': amplitude
            'phi_lmn': phase
            'f_lmn': frequency
            'tau_lmn': decay time
    qnm_pars_2 : dict
        Dictionary with the QNM parameters of the second mode:
            'A_lmn': amplitude
            'phi_lmn': phase
            'f_lmn': frequency
            'tau_lmn': decay time
    noise : dict
        Dictionary containing:
            'freq': noise frequency sampling
            'psd': square root of the noise spectral density
                (it will be squared in the Fisher Matrix computation)

    Returns
    -------
    dict, dict
        Returns sigma_1, sigma_2 containing the estimated errors for modes 1 and 2,
        respectively, of the same parameters
            'A_lmn': amplitude
            'phi_lmn': phase
            'f_lmn': frequency
            'tau_lmn': decay time
    """
    # convert noise to numpy arrays
    noise['freq'] = np.asanyarray(noise['freq'])
    noise['psd'] = np.asanyarray(noise['psd'])

    # create a 8x8 matrix filled with zeros
    fisher_matrix = np.zeros([8, 8])

    # sort the parameters to compute the Fisher Matrix
    parameters_list = [
        'A_lmn',
        'phi_lmn',
        'f_lmn',
        'tau_lmn',
        'A_lmn',
        'phi_lmn',
        'f_lmn',
        'tau_lmn',
    ]

    # create fisher matrix elements functions argument dictionaries
    qnm_parameters_dict = {}
    for i in range(0, 4):
        qnm_parameters_dict[i] = {
            'freq_array': noise['freq'],
            'A_lmn': qnm_pars_1['A_lmn'],
            'phi_lmn': qnm_pars_1['phi_lmn'],
            'f_lmn': qnm_pars_1['f_lmn'],
            'tau_lmn': qnm_pars_1['tau_lmn'],
        }
    for i in range(4, 8):
        qnm_parameters_dict[i] = {
            'freq_array': noise['freq'],
            'A_lmn': qnm_pars_2['A_lmn'],
            'phi_lmn': qnm_pars_2['phi_lmn'],
            'f_lmn': qnm_pars_2['f_lmn'],
            'tau_lmn': qnm_pars_2['tau_lmn'],
        }

    for i in range(0, 8):
        for j in range(0, 8):
            fisher_matrix[i, j] = fme.inner_product(noise['freq'],
                                                    global_amplitude * (
                antenna_plus *
                (fme.choose_polarization_and_parameter(
                    'real', parameters_list[i])(qnm_parameters_dict[i]))
                + antenna_cross * (fme.choose_polarization_and_parameter('imag', parameters_list[i])(qnm_parameters_dict[i]))),
                global_amplitude * (
                    antenna_plus *
                (fme.choose_polarization_and_parameter(
                    'real', parameters_list[j])(qnm_parameters_dict[i]))
                    + antenna_cross * (fme.choose_polarization_and_parameter('imag', parameters_list[j])(qnm_parameters_dict[j]))),
                noise['psd']
            )

    if np.linalg.det(fisher_matrix) == 0:
        correlation_matrix = np.zeros([8, 8])
        correlation_matrix.fill(np.inf)
    else:
        correlation_matrix = np.linalg.inv(fisher_matrix)

    sigma_1 = {
        'A_lmn':    np.sqrt(np.abs(correlation_matrix[0, 0])),
        'phi_lmn':  np.sqrt(np.abs(correlation_matrix[1, 1])),
        'f_lmn':    np.sqrt(np.abs(correlation_matrix[2, 2])),
        'tau_lmn':  np.sqrt(np.abs(correlation_matrix[3, 3])),
    }

    sigma_2 = {
        'A_lmn':    np.sqrt(np.abs(correlation_matrix[4, 4])),
        'phi_lmn':  np.sqrt(np.abs(correlation_matrix[5, 5])),
        'f_lmn':    np.sqrt(np.abs(correlation_matrix[6, 6])),
        'tau_lmn':  np.sqrt(np.abs(correlation_matrix[7, 7])),
    }

    return sigma_1, sigma_2
