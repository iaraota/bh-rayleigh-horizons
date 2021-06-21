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
from numpy import exp, pi


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
        QNM parameter, must be {'A', 'phi_0', 'f_0', 'tau_0',
        'R', 'phi_1', 'f_1', 'tau_1'}

    Returns
    -------
    function
        Returns function relative to the chosen polarization and parameter
    """
    returned_function = {
        'real': {
            'A': ft_dh_dA_real,
            'phi_0': ft_dh_dphi0_imag,
            'f_0': ft_dh_df0_real,
            'tau_0': ft_dh_dtau0_real,
            'R': ft_dh_dR_real,
            'phi_1': ft_dh_dphi1_imag,
            'f_1': ft_dh_df1_real,
            'tau_1': ft_dh_dtau1_real,
            },
        'imag':{
            'A': ft_dh_dA_imag,
            'phi_0': ft_dh_dphi0_imag,
            'f_0': ft_dh_df0_imag,
            'tau_0': ft_dh_dtau0_imag,
            'R': ft_dh_dR_imag,
            'phi_1': ft_dh_dphi1_imag,
            'f_1': ft_dh_df1_imag,
            'tau_1': ft_dh_dtau1_imag,
            }
        }
    return returned_function[polarization][parameter]


def ft_dh_dA_real(parameter:dict):
    """Compute the derivative relative to the global amplitude
    of the Fourier Transform of the real part of the QNM,
    h = A*(exp(-abs(t)/tau_0)*cos(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*cos(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio. The abs(t)
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the global amplitude
        of the Fourier Transform of the real part of the QNM

    """

    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return Q_0*exp(-1.0j*phi_0)/(2*pi*f_0*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) + Q_0*exp(1.0j*phi_0)/(2*pi*f_0*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) + R*(Q_1*exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) + Q_1*exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)))/2

def ft_dh_dA_imag(parameter:dict):
    """Compute the derivative relative to the global amplitude
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*(exp(-abs(t)/tau_0)*sin(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*sin(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio.
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the global amplitude
        of the Fourier Transform of the imaginary part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return 0.5j*(-Q_0*exp(-1.0j*phi_0)/(pi*f_0*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) + Q_0*exp(1.0j*phi_0)/(pi*f_0*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) + R*(-Q_1*exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) + Q_1*exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1))))

def ft_dh_dphi0_real(parameter:dict):
    """Compute the derivative relative to the first mode phase
    of the Fourier Transform of the real part of the QNM,
    h = A*(exp(-abs(t)/tau_0)*cos(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*cos(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio. The abs(t)
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the first mode phase
        of the Fourier Transform of the real part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return A*(-1.0j*Q_0*exp(-1.0j*phi_0)/(pi*f_0*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) + 1.0j*Q_0*exp(1.0j*phi_0)/(pi*f_0*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)))/2

def ft_dh_dphi0_imag(parameter:dict):
    """Compute the derivative relative to the first mode phase
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*(exp(-abs(t)/tau_0)*sin(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*sin(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio.
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the first mode phase
        of the Fourier Transform of the imaginary part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return 0.5j*A*(1.0j*Q_0*exp(-1.0j*phi_0)/(pi*f_0*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) + 1.0j*Q_0*exp(1.0j*phi_0)/(pi*f_0*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)))

def ft_dh_df0_real(parameter:dict):
    """Compute the derivative relative to the 1st mode frequency
    of the Fourier Transform of the real part of the QNM,
    h = A*(exp(-abs(t)/tau_0)*cos(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*cos(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio. The abs(t)
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the 1st mode frequency
        of the Fourier Transform of the real part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return A*(Q_0*(-4*Q_0**2*(2*pi*f_0 + 2*pi*freq_array)/(pi*f_0**2) + 2*Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**3))*exp(-1.0j*phi_0)/(pi*f_0*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)**2) + Q_0*(4*Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)/(pi*f_0**2) + 2*Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**3))*exp(1.0j*phi_0)/(pi*f_0*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)**2) - Q_0*exp(-1.0j*phi_0)/(pi*f_0**2*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) - Q_0*exp(1.0j*phi_0)/(pi*f_0**2*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)))/2

def ft_dh_df0_imag(parameter:dict):
    """Compute the derivative relative to the 1st mode frequency
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*(exp(-abs(t)/tau_0)*sin(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*sin(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio.
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the 1st mode frequency
        of the Fourier Transform of the imaginary part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return 0.5j*A*(-Q_0*(-4*Q_0**2*(2*pi*f_0 + 2*pi*freq_array)/(pi*f_0**2) + 2*Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**3))*exp(-1.0j*phi_0)/(pi*f_0*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)**2) + Q_0*(4*Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)/(pi*f_0**2) + 2*Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**3))*exp(1.0j*phi_0)/(pi*f_0*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)**2) + Q_0*exp(-1.0j*phi_0)/(pi*f_0**2*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) - Q_0*exp(1.0j*phi_0)/(pi*f_0**2*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)))

def ft_dh_dtau0_real(parameter:dict):
    """Compute the derivative relative to the 1st mode decay time
    of the Fourier Transform of the real part of the QNM,
    h = A*(exp(-abs(t)/tau_0)*cos(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*cos(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio. The abs(t)
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the 1st mode decay time
        of the Fourier Transform of the real part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return A*(-2*Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2*exp(1.0j*phi_0)/(pi**3*f_0**3*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)**2) - 2*Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2*exp(-1.0j*phi_0)/(pi**3*f_0**3*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)**2) + exp(-1.0j*phi_0)/(pi*f_0*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) + exp(1.0j*phi_0)/(pi*f_0*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)))/2

def ft_dh_dtau0_imag(parameter:dict):
    """Compute the derivative relative to the 1st mode decay time
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*(exp(-abs(t)/tau_0)*sin(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*sin(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio.
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the 1st mode decay time of
        the Fourier Transform of the imaginary part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return 0.5j*A*(-2*Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2*exp(1.0j*phi_0)/(pi**3*f_0**3*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)**2) + 2*Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2*exp(-1.0j*phi_0)/(pi**3*f_0**3*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)**2) - exp(-1.0j*phi_0)/(pi*f_0*(Q_0**2*(2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)) + exp(1.0j*phi_0)/(pi*f_0*(Q_0**2*(-2*pi*f_0 + 2*pi*freq_array)**2/(pi**2*f_0**2) + 1)))


def ft_dh_dR_real(parameter:dict):
    """Compute the derivative relative to the amplitude ratio
    of the Fourier Transform of the real part of the QNM,
    h = A*(exp(-abs(t)/tau_0)*cos(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*cos(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio. The abs(t)
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the amplitude ratio
        of the Fourier Transform of the real part of the QNM

    """

    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return A*(Q_1*exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) + Q_1*exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)))/2


def ft_dh_dR_imag(parameter:dict):
    """Compute the derivative relative to the global amplitude
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*(exp(-abs(t)/tau_0)*sin(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*sin(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio.
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the global amplitude
        of the Fourier Transform of the imaginary part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return 0.5j*A*(-Q_1*exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) + Q_1*exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)))

def ft_dh_dphi1_real(parameter:dict):
    """Compute the derivative relative to the 2nd mode phase
    of the Fourier Transform of the real part of the QNM,
    h = A*(exp(-abs(t)/tau_0)*cos(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*cos(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio. The abs(t)
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the 2nd mode phase
        of the Fourier Transform of the real part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return A*R*(-1.0j*Q_1*exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) + 1.0j*Q_1*exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)))/2

def ft_dh_dphi1_imag(parameter:dict):
    """Compute the derivative relative to the 2nd mode phase
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*(exp(-abs(t)/tau_0)*sin(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*sin(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio.
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the second mode phase
        of the Fourier Transform of the imaginary part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return 0.5j*A*R*(1.0j*Q_1*exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) + 1.0j*Q_1*exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)))

def ft_dh_df1_real(parameter:dict):
    """Compute the derivative relative to the 2nd mode frequency
    of the Fourier Transform of the real part of the QNM,
    h = A*(exp(-abs(t)/tau_0)*cos(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*cos(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio. The abs(t)
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the 2nd mode frequency
        of the Fourier Transform of the real part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return A*R*(Q_1*(-4*Q_1**2*(2*pi*f_1 + 2*pi*freq_array)/(pi*f_1**2) + 2*Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**3))*exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)**2) + Q_1*(4*Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)/(pi*f_1**2) + 2*Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**3))*exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)**2) - Q_1*exp(-1.0j*phi_1)/(pi*f_1**2*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) - Q_1*exp(1.0j*phi_1)/(pi*f_1**2*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)))/2

def ft_dh_df1_imag(parameter:dict):
    """Compute the derivative relative to the 2nd mode frequency
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*(exp(-abs(t)/tau_0)*sin(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*sin(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio.
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the 2nd mode frequency
        of the Fourier Transform of the imaginary part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return 0.5j*A*R*(-Q_1*(-4*Q_1**2*(2*pi*f_1 + 2*pi*freq_array)/(pi*f_1**2) + 2*Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**3))*exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)**2) + Q_1*(4*Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)/(pi*f_1**2) + 2*Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**3))*exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)**2) + Q_1*exp(-1.0j*phi_1)/(pi*f_1**2*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) - Q_1*exp(1.0j*phi_1)/(pi*f_1**2*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)))

def ft_dh_dtau1_real(parameter:dict):
    """Compute the derivative relative to the 2nd mode decay time
    of the Fourier Transform of the real part of the QNM,
    h = A*(exp(-abs(t)/tau_0)*cos(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*cos(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio. The abs(t)
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the 2nc mode decay time
        of the Fourier Transform of the real part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return A*R*(-2*Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2*exp(1.0j*phi_1)/(pi**3*f_1**3*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)**2) - 2*Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2*exp(-1.0j*phi_1)/(pi**3*f_1**3*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)**2) + exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) + exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)))/2

def ft_dh_dtau1_imag(parameter:dict):
    """Compute the derivative relative to the 2nd mode decay time
    of the Fourier Transform of the imaginary part of
    the QNM, h = A*(exp(-abs(t)/tau_0)*sin(2*pi*f_0 - phi_0) +
    R*exp(-abs(t)/tau_1)*sin(2*pi*f_1 - phi_1)), where
    R = A_1/A is the modes amplitude ratio.
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
                A : float
                    First QNM mode amplitude
                phi_0 : float
                    First QNM mode phase
                f_0 : float
                    First QNM mode frequency os oscilation
                tau_0 : float
                    First QNM mode decay time
                R : float
                    Amplitude ratior, R = A_1/A
                phi_1 : float
                    Second QNM mode phase
                f_1 : float
                    Second QNM mode frequency os oscilation
                tau_1 : float
                    Second QNM mode decay time

    Returns
    -------
    list
        Returns the derivative relative to the 2nd mode decay time of
        the Fourier Transform of the imaginary part of the QNM

    """
    # convert everything to numpy objects
    freq_array = np.asanyarray(parameter['freq_array'])

    A = np.asanyarray(parameter['A'])
    phi_0 = np.asanyarray(parameter['phi_0'])
    f_0 = np.asanyarray(parameter['f_0'])
    tau_0 = np.asanyarray(parameter['tau_0'])
    Q_0 = tau_0*f_0*np.pi

    R = np.asanyarray(parameter['R'])
    phi_1 = np.asanyarray(parameter['phi_1'])
    f_1 = np.asanyarray(parameter['f_1'])
    tau_1 = np.asanyarray(parameter['tau_1'])
    Q_1 = tau_1*f_1*np.pi

    return 0.5j*A*R*(-2*Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2*exp(1.0j*phi_1)/(pi**3*f_1**3*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)**2) + 2*Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2*exp(-1.0j*phi_1)/(pi**3*f_1**3*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)**2) - exp(-1.0j*phi_1)/(pi*f_1*(Q_1**2*(2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)) + exp(1.0j*phi_1)/(pi*f_1*(Q_1**2*(-2*pi*f_1 + 2*pi*freq_array)**2/(pi**2*f_1**2) + 1)))

