"""Compute the SNR of QNMs

from page 8 of https://arxiv.org/pdf/gr-qc/0512160.pdf:

The Fourier transform of the waveform can be computed using the elementary relation
$$
\int_{-\infty}^{+\infty} e^{i \omega t}\left(e^{\pm i \omega_{\ell m n}t - |t|/\tau_{\ell m n}} \right)dt = \frac{2/\tau_{\ell m n}}{1/\tau_{\ell m n}^2 + (\omega \pm \omega_{\ell m n})^2} \equiv 2 b_\pm
$$


Divide by factor $2$ to account for the doubling prescription.
$$
\tilde{h}_+ = \Re(\tilde{h}) = \frac{A_{\ell m n}}{2}\left[e^{-i\phi_{\ell m n}}b_+ + e^{i\phi_{\ell m n}}b_- \right]
$$
$$
\tilde{h}_\times = \Im(\tilde{h}) = \frac{i A_{\ell m n}}{2}\left[-e^{-i\phi_{\ell m n}}b_+ + e^{i\phi_{\ell m n}}b_- \right]
$$

For frequency in Herz: $\omega \rightarrow 2\pi f$
"""
import numpy as np

# import noise weighted inner product funcion
from fisher_matrix_elements import inner_product


def compute_SRN(global_amplitude: float, antenna_plus: float, antenna_cross: float, qnm_pars: dict, noise: dict):
    # convert noise to numpy arrays
    noise['freq'] = np.asanyarray(noise['freq'])
    noise['psd'] = np.asanyarray(noise['psd'])

    return np.sqrt(inner_product(noise['freq'],
                                 global_amplitude *
                                 (antenna_plus * h_real(qnm_pars) +
                                  antenna_cross * h_imag(qnm_pars)),
                                 global_amplitude *
                                 (antenna_plus * h_real(qnm_pars) +
                                  antenna_cross * h_imag(qnm_pars)),
                                 noise['psd']
                                 ))


def compute_SRN_2modes(global_amplitude: float, antenna_plus: float, antenna_cross: float, qnm_pars_0: dict, qnm_pars_1: dict, noise: dict):
    # convert noise to numpy arrays
    noise['freq'] = np.asanyarray(noise['freq'])
    noise['psd'] = np.asanyarray(noise['psd'])

    return np.sqrt(inner_product(noise['freq'],
                                 global_amplitude *
                                 (antenna_plus * (h_real(qnm_pars_0) + h_real(qnm_pars_1)) +
                                  antenna_cross * (h_imag(qnm_pars_0) + h_imag(qnm_pars_1))),
                                 global_amplitude *
                                 (antenna_plus * (h_real(qnm_pars_0) + h_real(qnm_pars_1)) +
                                  antenna_cross * (h_imag(qnm_pars_0) + h_imag(qnm_pars_1))),
                                 noise['psd']
                                 ))

# define b_\pm


def b_p(freq_array: list, f_lmn: float, tau_lmn: float):
    """Fourier transform of the only the part that contains
    a time dependence of a QNM. b_plus defined in the docstring
    above.

    Parameters
    ----------
    freq_array : list
        Sample frequencies corresponding to QNM waveform h
    f_lmn : float
        QNM frequency
    tau_lmn : float
        QNM decay time

    Returns
    -------
    list
        fourier transform of the time part of a QNM
    """
    return tau_lmn / (1 + ((2 * np.pi * freq_array + 2 * np.pi * f_lmn) * tau_lmn)**2)


def b_m(freq_array: list, f_lmn: float, tau_lmn: float):
    """Fourier transform of the only the complex conjugate of the
    part that contains a time dependence of a QNM.
    b_minus defined in the docstring above.

    Parameters
    ----------
    freq_array : list
        Sample frequencies corresponding to QNM waveform h
    f_lmn : float
        QNM frequency
    tau_lmn : float
        QNM decay time

    Returns
    -------
    list
        fourier transform of the complex conjugate of time part of a QNM
    """
    return tau_lmn / (1 + ((2 * np.pi * freq_array - 2 * np.pi * f_lmn) * tau_lmn)**2)

# define the Fourier transform of each polirization


def h_real(parameter: dict):
    """Fourier transform of the plus polarization (real part) of
    a QNM

    Parameters
    ----------
    A_lmn : float
        QNM mode amplitude
    phi_lmn : float
        QNM mode phase
    f_lmn : float
        QNM mode frequency os oscilation
    tau_lmn : float
        QNM mode decay time


    Returns
    -------
    list
        Fourier transform of the plus polarization (real part) of
        a QNM
    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return (A_lmn / 2) * (
        np.exp(-1j * phi_lmn) * b_p(freq_array, f_lmn, tau_lmn)
        + np.exp(1j * phi_lmn) * b_m(freq_array, f_lmn, tau_lmn)
    )


def h_imag(parameter: dict):
    """Fourier transform of the cross polarization (imaginary part)
    of a QNM

    Parameters
    ----------
    A_lmn : float
        QNM mode amplitude
    phi_lmn : float
        QNM mode phase
    f_lmn : float
        QNM mode frequency os oscilation
    tau_lmn : float
        QNM mode decay time


    Returns
    -------
    list
        Fourier transform of the plus polarization (imaginary part)
        of a QNM
    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return 1j * (A_lmn / 2) * (
        - np.exp(-1j * phi_lmn) * b_p(freq_array, f_lmn, tau_lmn)
        + np.exp(1j * phi_lmn) * b_m(freq_array, f_lmn, tau_lmn)
    )
