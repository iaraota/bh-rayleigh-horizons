"""Functions to compute the elements of the Fisher
Matrix considering the QNM parameters, see equation
(4.2) of E. Berti, V. Cardoso, and C. M. Will,
Phys. Rev. D 73, 064030 (2006),
https://arxiv.org/pdf/gr-qc/0512160.pdf

the Fourier transforms consider the Flanagan and Hughes
prescription (see E. E. Flanagan and S. A. Hughes, PRD
57, 4535 (1998)  https://arxiv.org/pdf/gr-qc/9701039.pdf
or eq. (3.5) of E. Berti, V. Cardoso, and C. M. Will,
Phys. Rev. D 73, 064030 (2006),
https://arxiv.org/pdf/gr-qc/0512160.pdf)

The partial derivatives were computed using Sympy,
see 'notebooks/derivatives_fourier_transform_FH.ipynb'.
"""

import numpy as np


def inner_product(freq_array: list, h1: list, h2: list, Sh: list):
    """Compute gravitational wave noise-weigthed inner product

    Parameters
    ----------
    freq_array : list
        Sample frequencies corresponding to h1, h2 and Sh for integration
    h1 : list
        first argument
    h2 : list
        second argument
    Sh : list
        Square root of noise Power Spectral Density [1/sqrt(Hz)]

    Returns
    -------
    scalar: Returns inner product between h1 and h2 weigthed by detector noise Sh.
    """
    freq_array, h1, h2, Sh = np.asanyarray(freq_array), np.asanyarray(
        h1), np.asanyarray(h2), np.asanyarray(Sh)
    return 4 * np.real(np.trapz((h1 * np.conj(h2)) / Sh**2, x=freq_array))

def choose_polarization_and_parameter(polarization:str, parameter:str):
    """Choose the Fisher Matrix elemente function of a given
    waveform polarization and a QNM parameter

    Parameters
    ----------
    polarization : str
        waveform polarization, must be 'real' or 'imag'
    parameter : str
        QNM parameter, must be 'A', 'phi', 'f', 'tau'

    Returns
    -------
    function
        Returns function relative to the chosen polarization and parameter
    """
    returned_function = {
        'real': {
            'A_lmn': ft_dh_dA_real,
            'phi_lmn': ft_dh_dphi_imag,
            'f_lmn': ft_dh_df_real,
            'tau_lmn': ft_dh_dtau_real,
            },
        'imag':{
            'A_lmn': ft_dh_dA_imag,
            'phi_lmn': ft_dh_dphi_imag,
            'f_lmn': ft_dh_df_imag,
            'tau_lmn': ft_dh_dtau_imag,
            }
        }
    return returned_function[polarization][parameter]


def ft_dh_dA_real(parameter:dict):
    """Compute the derivative relative to the amplitude
    of the Fourier Transform of the real part of the QNM,
    h = A*exp(-abs(t)/tau)*cos(2*pi*f - phi). The abs(t)
    is relative to the Flanagan and Hughes prescription
    (see E. E. Flanagan and S. A. Hughes, Phys. Rev. D
    57, 4535 (1998)  https://arxiv.org/pdf/gr-qc/9701039.pdf
    or eq. (3.5) of E. Berti, V. Cardoso, and C. M. Will,
    Phys. Rev. D 73, 064030 (2006),
    https://arxiv.org/pdf/gr-qc/0512160.pdf)

    Parameters
    ----------
    parameter : dict
        A dictionary containing the keys:
                freq_array : list
                    Sample frequencies corresponding to QNM waveform h
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
        Returns the derivative relative to the frequency
        of the Fourier Transform of the real part of the QNM

    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return tau_lmn * np.exp(-1j * phi_lmn) / (2 * (tau_lmn**2 * (2 * np.pi * freq_array + 2 * np.pi * f_lmn)**2 + 1)) + tau_lmn * np.exp(1j * phi_lmn) / (2 * (tau_lmn**2 * (2 * np.pi * freq_array - 2 * np.pi * f_lmn)**2 + 1))

def ft_dh_dA_imag(parameter:dict):
    """Compute the derivative relative to the amplitude
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*exp(-abs(t)/tau)*sin(2*pi*f - phi).
    The abs(t) is relative to the Flanagan and Hughes
    prescription (see E. E. Flanagan and S. A. Hughes,
    Phys. Rev. D 57, 4535 (1998)
    https://arxiv.org/pdf/gr-qc/9701039.pdf
    or eq. (3.5) of E. Berti, V. Cardoso, and C. M. Will,
    Phys. Rev. D 73, 064030 (2006),
    https://arxiv.org/pdf/gr-qc/0512160.pdf)

    Parameters
    ----------
    parameter : dict
        A dictionary containing the keys:
                freq_array : list
                    Sample frequencies corresponding to QNM waveform h
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
        Returns the derivative relative to the amplitude of the
        Fourier Transform of the imaginary part of the QNM

    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return 0.5j*(-tau_lmn*np.exp(-1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2 + 1) + tau_lmn*np.exp(1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2 + 1))

def ft_dh_dphi_real(parameter:dict):
    """Compute the derivative relative to the phase of
    the Fourier Transform of the real part of the QNM,
    h = A*exp(-abs(t)/tau)*cos(2*pi*f - phi). The abs(t)
    is relative to the Flanagan and Hughes prescription
    (see E. E. Flanagan and S. A. Hughes, Phys. Rev. D
    57, 4535 (1998)  https://arxiv.org/pdf/gr-qc/9701039.pdf
    or eq. (3.5) of E. Berti, V. Cardoso, and C. M. Will,
    Phys. Rev. D 73, 064030 (2006),
    https://arxiv.org/pdf/gr-qc/0512160.pdf)

    Parameters
    ----------
    parameter : dict
        A dictionary containing the keys:
                freq_array : list
                    Sample frequencies corresponding to QNM waveform h
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
        Returns the derivative relative to the phase of
        the Fourier Transform of the real part of the QNM

    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return A_lmn*(-1j*tau_lmn*np.exp(-1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2 + 1) + 1j*tau_lmn*np.exp(1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2 + 1))/2

def ft_dh_dphi_imag(parameter:dict):
    """Compute the derivative relative to the phase
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*exp(-abs(t)/tau)*sin(2*pi*f - phi).
    The abs(t) is relative to the Flanagan and Hughes
    prescription (see E. E. Flanagan and S. A. Hughes,
    Phys. Rev. D 57, 4535 (1998)
    https://arxiv.org/pdf/gr-qc/9701039.pdf
    or eq. (3.5) of E. Berti, V. Cardoso, and C. M. Will,
    Phys. Rev. D 73, 064030 (2006),
    https://arxiv.org/pdf/gr-qc/0512160.pdf)

    Parameters
    ----------
    parameter : dict
        A dictionary containing the keys:
                freq_array : list
                    Sample frequencies corresponding to QNM waveform h
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
        Returns the derivative relative to the phase of the
        Fourier Transform of the imaginary part of the QNM

    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return 0.5j*A_lmn*(1j*tau_lmn*np.exp(-1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2 + 1) + 1j*tau_lmn*np.exp(1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2 + 1))

def ft_dh_df_real(parameter:dict):
    """Compute the derivative relative to the frequency
    of the Fourier Transform of the real part of the QNM,
    h = A*exp(-abs(t)/tau)*cos(2*pi*f - phi). The abs(t)
    is relative to the Flanagan and Hughes prescription
    (see E. E. Flanagan and S. A. Hughes, Phys. Rev. D
    57, 4535 (1998)  https://arxiv.org/pdf/gr-qc/9701039.pdf
    or eq. (3.5) of E. Berti, V. Cardoso, and C. M. Will,
    Phys. Rev. D 73, 064030 (2006),
    https://arxiv.org/pdf/gr-qc/0512160.pdf)

    Parameters
    ----------
    parameter : dict
        A dictionary containing the keys:
                freq_array : list
                    Sample frequencies corresponding to QNM waveform h
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
        Returns the derivative relative to the frequency
        of the Fourier Transform of the real part of the QNM

    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return A_lmn*(4*np.pi*tau_lmn**3*(2*np.pi*freq_array - 2*np.pi*f_lmn)*np.exp(1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2 + 1)**2 - 4*np.pi*tau_lmn**3*(2*np.pi*freq_array + 2*np.pi*f_lmn)*np.exp(-1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2 + 1)**2)/2

def ft_dh_df_imag(parameter:dict):
    """Compute the derivative relative to the frequency
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*exp(-abs(t)/tau)*sin(2*pi*f - phi).
    The abs(t) is relative to the Flanagan and Hughes
    prescription (see E. E. Flanagan and S. A. Hughes,
    Phys. Rev. D 57, 4535 (1998)
    https://arxiv.org/pdf/gr-qc/9701039.pdf
    or eq. (3.5) of E. Berti, V. Cardoso, and C. M. Will,
    Phys. Rev. D 73, 064030 (2006),
    https://arxiv.org/pdf/gr-qc/0512160.pdf)

    Parameters
    ----------
    parameter : dict
        A dictionary containing the keys:
                freq_array : list
                    Sample frequencies corresponding to QNM waveform h
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
        Returns the derivative relative to the frequency of the
        Fourier Transform of the imaginary part of the QNM

    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return 0.5j*A_lmn*(4*np.pi*tau_lmn**3*(2*np.pi*freq_array - 2*np.pi*f_lmn)*np.exp(1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2 + 1)**2 + 4*np.pi*tau_lmn**3*(2*np.pi*freq_array + 2*np.pi*f_lmn)*np.exp(-1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2 + 1)**2)

def ft_dh_dtau_real(parameter:dict):
    """Compute the derivative relative to the decay time
    of the Fourier Transform of the real part of the QNM,
    h = A*exp(-abs(t)/tau)*cos(2*pi*f - phi). The abs(t)
    is relative to the Flanagan and Hughes prescription
    (see E. E. Flanagan and S. A. Hughes, Phys. Rev. D
    57, 4535 (1998)  https://arxiv.org/pdf/gr-qc/9701039.pdf
    or eq. (3.5) of E. Berti, V. Cardoso, and C. M. Will,
    Phys. Rev. D 73, 064030 (2006),
    https://arxiv.org/pdf/gr-qc/0512160.pdf)

    Parameters
    ----------
    parameter : dict
        A dictionary containing the keys:
                freq_array : list
                    Sample frequencies corresponding to QNM waveform h
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
        Returns the derivative relative to the decay time
        of the Fourier Transform of the real part of the QNM

    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return A_lmn*(-2*tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2*np.exp(1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2 + 1)**2 - 2*tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2*np.exp(-1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2 + 1)**2 + np.exp(-1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2 + 1) + np.exp(1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2 + 1))/2

def ft_dh_dtau_imag(parameter:dict):
    """Compute the derivative relative to the decay time
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*exp(-abs(t)/tau)*sin(2*pi*f - phi).
    The abs(t) is relative to the Flanagan and Hughes
    prescription (see E. E. Flanagan and S. A. Hughes,
    Phys. Rev. D 57, 4535 (1998)
    https://arxiv.org/pdf/gr-qc/9701039.pdf
    or eq. (3.5) of E. Berti, V. Cardoso, and C. M. Will,
    Phys. Rev. D 73, 064030 (2006),
    https://arxiv.org/pdf/gr-qc/0512160.pdf)

    Parameters
    ----------
    parameter : dict
        A dictionary containing the keys:
                freq_array : list
                    Sample frequencies corresponding to QNM waveform h
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
        Returns the derivative relative to the decay time of
        the Fourier Transform of the imaginary part of the QNM

    """
    freq_array = np.asanyarray(parameter['freq_array'])
    A_lmn = np.asanyarray(parameter['A_lmn'])
    phi_lmn = np.asanyarray(parameter['phi_lmn'])
    f_lmn = np.asanyarray(parameter['f_lmn'])
    tau_lmn = np.asanyarray(parameter['tau_lmn'])

    return 0.5j*A_lmn*(-2*tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2*np.exp(1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2 + 1)**2 + 2*tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2*np.exp(-1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2 + 1)**2 - np.exp(-1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array + 2*np.pi*f_lmn)**2 + 1) + np.exp(1j*phi_lmn)/(tau_lmn**2*(2*np.pi*freq_array - 2*np.pi*f_lmn)**2 + 1))
