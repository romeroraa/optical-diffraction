"""
Analytical theory for focused Gaussian beams through circular apertures.

Based on:
- Tanaka et al., 1985 (intensity distributions)
- Horvath & Bor, 2003 (truncation effects)

Ported from gbp-mc repository with refactoring for mcdo structure.
"""

import numpy as np
from scipy.special import jv
from scipy.integrate import quad
from typing import Tuple, Union


class FocusedGaussianBeamTheory:
    """
    Analytical theory for focused Gaussian beam through circular aperture.

    Implements formulas from Tanaka et al. (1985) for intensity distributions
    of truncated Gaussian beams focused by a lens.

    Parameters
    ----------
    numerical_aperture : float
        Numerical aperture (NA = n*sin(α))
    wavelength : float
        Wavelength in microns
    n_medium : float
        Refractive index of medium
    focal_length : float
        Focal length in microns
    z_focus : float
        Z-coordinate of focus (default: 0)
    truncation_coeff : float
        Truncation coefficient α (Horvath & Bor, 2003)
        - α ≥ 4: untruncated Gaussian beam
        - α < 4: truncated beam
        - α = 0: spherical beam (uniform aperture illumination)
        Default: 4 (untruncated)
    """

    def __init__(
        self,
        numerical_aperture: float,
        wavelength: float = 0.532,
        n_medium: float = 1.0,
        focal_length: float = None,
        z_focus: float = 0.0,
        truncation_coeff: float = 4.0
    ):
        self.NA = numerical_aperture
        self.wavelength = wavelength
        self.n = n_medium
        self.z_f = z_focus
        self.trunc_coeff = truncation_coeff

        # Calculate aperture from NA
        theta = np.arcsin(self.NA / self.n)

        # If focal length not provided, use default based on aperture
        if focal_length is None:
            # Default: aperture diameter = 1500 μm (from gbp-mc)
            self.aperture = 1500.0
            self.f = (self.aperture / 2) / np.tan(theta)
        else:
            self.f = focal_length
            self.aperture = 2 * self.f * np.tan(theta)

        self.z_lens = self.z_f - self.f

        # Wave number
        self.k = 2 * np.pi / self.wavelength

        # Beam waist calculation
        # w_incident: 1/e² radius of Gaussian at aperture plane
        # For Gaussian amplitude A(r) = exp(-r²/w_incident²) = exp(-α*(r/r_ap)²)
        # we need w_incident = r_aperture / sqrt(α) = (aperture/2) / sqrt(α)
        self.w_incident = (self.aperture / 2) / np.sqrt(truncation_coeff)

        # Focused beam waist using Gaussian beam propagation
        Nw = self.w_incident**2 / (self.wavelength * self.f)
        self.w0 = self.w_incident / np.sqrt(1 + (np.pi * Nw)**2)

        # Rayleigh range
        self.z_R = np.pi * self.w0**2 / self.wavelength

        # For reference
        self.airy_radius = 0.61 * wavelength / numerical_aperture

    def beam_radius(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Gaussian beam radius w(z) at position z.

        Parameters
        ----------
        z : float or array
            Axial position(s)

        Returns
        -------
        w : float or array
            Beam radius (1/e² intensity radius)
        """
        return self.w0 * np.sqrt(1 + ((z - self.z_f) / self.z_R)**2)

    def curvature_radius(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Radius of curvature R(z) of the wavefront.

        Parameters
        ----------
        z : float or array
            Axial position(s)

        Returns
        -------
        R : float or array
            Radius of curvature
        """
        return -(z - self.z_f) * (1 + (self.z_R / (z - self.z_f))**2)

    def epsilon(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Dimensionless parameter ε(z) for beam propagation.

        Parameters
        ----------
        z : float or array
            Axial position(s)

        Returns
        -------
        ε : float or array
        """
        k = 2 * np.pi / self.wavelength
        ws = self.beam_radius(z=self.z_lens)
        return 2 * (z - self.z_f) / (k * ws**2)

    def axial_intensity(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Axial intensity I(r=0, z) along optical axis.

        Reference: Tanaka et al., 1985

        IMPORTANT: Requires lens at z=0 (or adjust z coordinates accordingly)

        Parameters
        ----------
        z : float or array
            Axial position(s)

        Returns
        -------
        I : float or array
            Normalized intensity
        """
        z = np.atleast_1d(z)

        k = 2 * np.pi / self.wavelength
        ws = self.beam_radius(z=self.z_lens)
        alpha = (self.aperture / 2) / ws
        P = k * ws**2 / self.f
        Z = (z - self.z_lens) / self.f

        eps = self.epsilon(z=self.z_lens)

        s1 = (P * alpha**2 * (1 - Z) / (2 * Z)) + (alpha**2 * eps / (1 + eps**2))
        s2 = -alpha**2 / (1 + eps**2)

        f1 = P**2 * alpha**4 / (4 * Z**2 * (s1**2 + s2**2))
        f2 = 1 + np.exp(2*s2) - 2*np.exp(s2)*np.cos(s1)
        intensity = f1 * f2

        # Handle z = 0 case
        z_is_zero = np.abs(z - self.z_lens) < 1e-10
        intensity[z_is_zero] = 1.0

        return intensity if len(z) > 1 else intensity[0]

    def focal_plane_intensity(
        self,
        r: Union[float, np.ndarray],
        z: float = None
    ) -> Union[float, np.ndarray]:
        """
        Intensity distribution I(r, z) in focal region.

        Reference: Tanaka et al., 1985

        Parameters
        ----------
        r : float or array
            Radial coordinate(s) from optical axis
        z : float, optional
            Axial position. If None, uses focal plane (z=z_f)

        Returns
        -------
        I : float or array
            Normalized intensity
        """
        if z is None:
            z = self.z_f

        r = np.atleast_1d(r)

        k = 2 * np.pi / self.wavelength
        ws = self.beam_radius(z=self.z_lens)
        alpha = (self.aperture / 2) / ws
        P = k * ws**2 / self.f
        Z = (z - self.z_lens) / self.f
        R = r / ws

        eps = self.epsilon(z=self.z_lens)

        s1 = (P * alpha**2 * (1 - Z) / (2 * Z)) + (alpha**2 * eps / (1 + eps**2))
        s2 = -alpha**2 / (1 + eps**2)

        factor = P * alpha**2 / Z**2

        # Define integrands
        def f1_integrand(r0, R_val):
            return r0 * jv(0, P * alpha * R_val / Z * r0) * \
                   np.exp(-alpha**2 * r0**2 / (1 + eps**2)) * np.cos(s1 * r0**2)

        def f2_integrand(r0, R_val):
            return r0 * jv(0, P * alpha * R_val / Z * r0) * \
                   np.exp(-alpha**2 * r0**2 / (1 + eps**2)) * np.sin(s1 * r0**2)

        # Compute intensity for each r value
        intensity = np.zeros_like(r, dtype=float)

        for i, R_val in enumerate(R):
            # Integrate from 0 to 1
            firstterm, _ = quad(lambda r0: f1_integrand(r0, R_val), 0, 1, limit=100)
            secondterm, _ = quad(lambda r0: f2_integrand(r0, R_val), 0, 1, limit=100)

            intensity[i] = factor * (firstterm**2 + secondterm**2)

        # Normalize to peak
        if intensity.max() > 0:
            intensity = intensity / intensity.max()

        return intensity if len(r) > 1 else intensity[0]

    def transverse_intensity(
        self,
        r: Union[float, np.ndarray],
        z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Simple Gaussian transverse intensity profile.

        I(r, z) = exp(-2*r²/w(z)²)

        Parameters
        ----------
        r : float or array
            Radial coordinate(s)
        z : float or array
            Axial position(s)

        Returns
        -------
        I : float or array
            Normalized intensity
        """
        wz = self.beam_radius(z)
        return np.exp(-2 * r**2 / wz**2)

    def get_parameters(self) -> dict:
        """
        Get all beam parameters.

        Returns
        -------
        params : dict
            Dictionary of beam parameters
        """
        return {
            'NA': self.NA,
            'wavelength': self.wavelength,
            'n_medium': self.n,
            'focal_length': self.f,
            'aperture': self.aperture,
            'z_focus': self.z_f,
            'truncation_coeff': self.trunc_coeff,
            'w0': self.w0,
            'z_R': self.z_R,
            'w_incident': getattr(self, 'w_incident', None),
            'airy_radius': self.airy_radius,
        }
