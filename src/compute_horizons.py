import os
import pandas as pd
import numpy as np


def import_data(detector: str, mass_ratio, par='rayleigh_criterion'):
    """Import data computed from scr/rayleigh_criterion_two_modes.py

    Parameters
    ----------
    detector : str
        Gravitational wave detector name. Must be {'LIGO', 'LISA',
        'CE' = 'CE2silicon', 'CE2silica', 'ET'}.
    mass_ratio : TYPE
        Binary black hole mass ratio. mass_ratio >= 1. This is used to
        determine the QNM parameters.

    Returns
    -------
    Pandas DataFrame
        Return DataFrame with the data
    """
    columns = ('mass',
               'redshift',
               'mode_0',
               'mode_1',
               'delta_freq',
               'sigma_freq_0',
               'sigma_freq_1',
               'delta_tau',
               'sigma_tau_0',
               'sigma_tau_1',
               'snr_0',
               'snr_1',
               )

    df = pd.read_csv(
        f'../data/rayleigh_criterion/{detector}_q_{mass_ratio}_{par}.dat', delimiter="\t", comment="#", names=columns)

    return df


def import_data_all_errors(detector: str, mass_ratio, par='tau'):
    """Import data computed from scr/rayleigh_criterion_two_modes.py

    Parameters
    ----------
    detector : str
        Gravitational wave detector name. Must be {'LIGO', 'LISA',
        'CE' = 'CE2silicon', 'CE2silica', 'ET'}.
    mass_ratio : TYPE
        Binary black hole mass ratio. mass_ratio >= 1. This is used to
        determine the QNM parameters.

    Returns
    -------
    Pandas DataFrame
        Return DataFrame with the data
    """

    df = pd.read_csv(
        f'../data/all_errors/{detector}_q_{mass_ratio}_{par}.csv')

    return df


def compute_criterion(data_frame):
    """Compute Rayleigh Criterion and SNR > 8 criterion.

    Parameters
    ----------
    data_frame : Pandas DataFrame
        DataFrame imported using 'import_data' function.
    """
    # compute rayleigh criterion
    data_frame['ray_freq'] = data_frame.apply(
        lambda row: np.sign(row.delta_freq - max(row.sigma_freq_0, row.sigma_freq_1)) + 1, axis=1)

    data_frame['ray_tau'] = data_frame.apply(
        lambda row: np.sign(row.delta_tau - max(row.sigma_tau_0, row.sigma_tau_1)) + 1, axis=1)

    data_frame['rayleigh'] = data_frame.apply(
        lambda row: np.sign(row.ray_freq and row.ray_tau), axis=1)

    data_frame['rayleigh_one'] = data_frame.apply(
        lambda row: np.sign(row.ray_freq or row.ray_tau), axis=1)

    # compute SNR > 8
    data_frame['snr_c_0'] = data_frame.apply(
        lambda row: np.sign(row.snr_0 - 8) + 1, axis=1)
    data_frame['snr_c_1'] = data_frame.apply(
        lambda row: np.sign(row.snr_1 - 8) + 1, axis=1)
    data_frame['snr'] = data_frame.apply(
        lambda row: np.sign(row.snr_c_0 and row.snr_c_1), axis=1)

    # both conditions
    data_frame['both'] = data_frame.apply(
        lambda row: np.sign(row.rayleigh and row.snr), axis=1)

    # critical conditions (one condition)
    data_frame['critical'] = data_frame.apply(
        lambda row: np.sign(row.rayleigh_one and row.snr), axis=1)


def compute_criterion_all_errors(data_frame):
    """Compute Rayleigh Criterion and SNR > 8 criterion.

    Parameters
    ----------
    data_frame : Pandas DataFrame
        DataFrame imported using 'import_data' function.
    """
    # compute rayleigh criterion
    data_frame['ray_freq'] = data_frame.apply(
        lambda row: np.sign(np.abs(row.freq_mode_0 - row.freq_mode_1)
                            - max(row.error_f_mode_0, row.error_f_mode_1)) + 1, axis=1)

    data_frame['ray_tau'] = data_frame.apply(
        lambda row: np.sign(np.abs(row.tau_mode_0 - row.tau_mode_1)
                            - max(row.error_tau_mode_0, row.error_tau_mode_1)) + 1, axis=1)

    data_frame['rayleigh'] = data_frame.apply(
        lambda row: np.sign(row.ray_freq and row.ray_tau), axis=1)

    data_frame['rayleigh_one'] = data_frame.apply(
        lambda row: np.sign(row.ray_freq or row.ray_tau), axis=1)
    ##!!!!!!!!!!!!!!!
    #####ATENCAO SE O SNR FOR CORRIGIDO TIRAR A RAIZ QUADRADA
    ##!!!!!!!!!!!!!!!!!!!!!!!!!!
    # compute SNR > 8
    data_frame['snr_c_0'] = data_frame.apply(
        lambda row: np.sign(np.sqrt(row.SNR_mode_0) - 8) + 1, axis=1)
    data_frame['snr_c_1'] = data_frame.apply(
        lambda row: np.sign(np.sqrt(row.SNR_mode_1) - 8) + 1, axis=1)
    data_frame['snr'] = data_frame.apply(
        lambda row: np.sign(row.snr_c_0 and row.snr_c_1), axis=1)

    # both conditions
    data_frame['both'] = data_frame.apply(
        lambda row: np.sign(row.rayleigh and row.snr), axis=1)

    # critical conditions (one condition)
    data_frame['critical'] = data_frame.apply(
        lambda row: np.sign(row.rayleigh_one and row.snr), axis=1)

    # Distinguishability condition (covariance) (eq; 66 of https://arxiv.org/pdf/2107.05609.pdf)
    data_frame['covariance'] = data_frame.apply(
        lambda row: np.sign(
            ((row.freq_mode_0 - row.freq_mode_1)**2) / (row.error_f_mode_0**2 + row.error_f_mode_1**2) +
            ((row.tau_mode_0 - row.tau_mode_1)**2) /
            (row.error_tau_mode_0**2 + row.error_tau_mode_1**2) - 1
        ) + 1, axis=1)

    data_frame['covariance_snr'] = data_frame.apply(
        lambda row: np.sign(row.covariance and row.snr), axis=1)


def find_horizon_contour(data_frame, modes: tuple, criterion='both'):
    """Compute horizon contour

    Parameters
    ----------
    data_frame : Pandas DataFrame
        Description
    modes : tuple
        2d tuple containig mode_0 and mode_1

    Returns
    -------
    list, list
        Returns masses and redshifts at the horizon arrays
    """
    pair = (data_frame.mode_0 == modes[0]) & (
        data_frame.mode_1 == modes[1]) & (data_frame[criterion] == 1)
    df_pair = data_frame[pair]
    X = df_pair.mass
    Y = df_pair.redshift
    Z = df_pair[criterion]
    masses = sorted(set(X))
    redshifts = [max(Y[X == mass]) for mass in masses]
    if len(redshifts) > 0:
        redshifts[0] = min(Y[X == masses[0]])
        redshifts[-1] = min(Y[X == masses[-1]])

    return masses, redshifts


def two_modes_horizons(data_frame, save_df, detector, mass_ratio, criterion='both'):
    modes = ['(2,2,1) II', '(3,3,0)', '(4,4,0)', '(2,1,0)']
    two_modes = [('(2,2,0)', mode) for mode in modes]

    for comb in two_modes:
        extra = {}
        extra['masses'],  extra['redshifts'] = find_horizon_contour(
            data_frame, comb, criterion)
        if len(extra['masses']) == 0:
            extra['masses'],  extra['redshifts'] = [np.nan] * 2, [np.nan] * 2
        extra['modes'] = [f'{comb}'] * len(extra['masses'])
        extra['detector'] = [detector] * len(extra['masses'])
        extra['mass_ratio'] = [mass_ratio] * len(extra['masses'])
        save_df = save_df.append(pd.DataFrame(extra))

    return save_df


def save_horizons():
    detectors = ["LIGO", "ET", "CE", "LISA"]
    mass_ratios = [1.5, 10]

    data_frames = {}

    horizons = pd.DataFrame()
    for detector in detectors:
        for mass_ratio in mass_ratios:
            data_frames[detector] = import_data(detector, mass_ratio)
            compute_criterion(data_frames[detector])
            horizons = two_modes_horizons(
                data_frames[detector], horizons, detector, mass_ratio)

    file_name = f'../data/two_modes_horizons.dat'
    horizons.to_csv(file_name, sep='\t', index=False)


if __name__ == '__main__':
    detector = "LIGO"
    mass_ratios = [1.5, 10]

    data_frames = {}

    horizons = pd.DataFrame()
    par = 'tau'
    # par = 'Qfactor'
    for mass_ratio in mass_ratios:
        data_frame = import_data_all_errors(detector, mass_ratio, par)
        compute_criterion_all_errors(data_frame)
        horizons = two_modes_horizons(
            data_frame, horizons, detector, mass_ratio, 'critical')

    file_name = f'../data/critical_horizons.csv'
    horizons.to_csv(file_name, index=False)
