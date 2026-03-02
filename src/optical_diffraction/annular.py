"""
Annular aperture diffraction using Babinet's principle.

An annular aperture (inner obscuration ratio ε) is computed as:
    E_annular = E_outer_disk - E_inner_disk

where the inner disk has NA_inner = ε * NA_outer.
"""

import numpy as np
from typing import Tuple
from .richards_wolf import RichardsWolfSimulator


def annular_field(
    NA: float,
    epsilon: float,
    r: np.ndarray,
    z: np.ndarray,
    wavelength: float = 0.532,
    n_medium: float = 1.0,
    input_field: str = 'gaussian',
    truncation_coeff: float = 4.0,
    polarization: str = 'x',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute annular aperture electric field using Babinet's principle.

    Parameters
    ----------
    NA : float
        Numerical aperture of the outer disk
    epsilon : float
        Obscuration ratio (0 = full disk, 1 = fully blocked)
    r : np.ndarray
        Radial coordinates in microns
    z : np.ndarray
        Axial coordinates in microns (z=0 at focus)
    wavelength : float
        Wavelength in microns (default: 0.532)
    n_medium : float
        Refractive index of medium (default: 1.0)
    input_field : str
        'uniform' or 'gaussian' (default: 'gaussian')
    truncation_coeff : float
        Truncation coefficient for Gaussian input (default: 4.0)
    polarization : str
        'x', 'y', or 'circular' (default: 'x')

    Returns
    -------
    Ex, Ey, Ez : np.ndarray (complex)
        Electric field components of the annular aperture
    """
    # Outer disk
    rw_outer = RichardsWolfSimulator(
        wavelength=wavelength,
        numerical_aperture=NA,
        n_medium=n_medium,
        polarization=polarization,
        input_field=input_field,
        truncation_coeff=truncation_coeff,
    )
    Ex_out, Ey_out, Ez_out = rw_outer.compute_field(r, z)

    if epsilon <= 0.001:
        return Ex_out, Ey_out, Ez_out

    # Inner disk: NA_inner = epsilon * NA_outer
    # Use gaussian_reference_na=NA so both disks sample the same physical beam
    rw_inner = RichardsWolfSimulator(
        wavelength=wavelength,
        numerical_aperture=NA * epsilon,
        n_medium=n_medium,
        polarization=polarization,
        input_field=input_field,
        truncation_coeff=truncation_coeff,
        gaussian_reference_na=NA,
    )
    Ex_in, Ey_in, Ez_in = rw_inner.compute_field(r, z)

    return Ex_out - Ex_in, Ey_out - Ey_in, Ez_out - Ez_in


def annular_intensity(
    NA: float,
    epsilon: float,
    r: np.ndarray,
    z: np.ndarray,
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Compute normalized total intensity for an annular aperture.

    Parameters
    ----------
    NA : float
        Outer numerical aperture
    epsilon : float
        Obscuration ratio
    r : np.ndarray
        Radial coordinates in microns
    z : np.ndarray
        Axial coordinates in microns
    normalize : bool
        Normalize to peak intensity (default: True)
    **kwargs
        Passed to annular_field

    Returns
    -------
    I : np.ndarray
        Intensity distribution
    """
    Ex, Ey, Ez = annular_field(NA, epsilon, r, z, **kwargs)
    I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    if normalize and I.max() > 0:
        I = I / I.max()
    return I
