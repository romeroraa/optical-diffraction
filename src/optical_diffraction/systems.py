"""Optical system definitions and classes with corrected wavelength handling."""

import numpy as np
from typing import Optional, Dict, Any
from .constants import LIGHT_SPEED


class OpticalSystem:
    """
    Base class for optical systems with proper wavelength handling.
    
    This class handles basic optical parameters and calculates derived properties
    like Rayleigh length, diffraction limits, and angular apertures based on
    Gaussian beam theory and Richards-Wolf diffraction theory.
    
    CRITICAL FIX: Properly handles wavelength in vacuum vs wavelength in medium
    to match Romallosa et al. (2003) equations exactly.
    
    Parameters
    ----------
    wavelength : float
        Wavelength in vacuum in meters
    numerical_aperture : float
        Numerical aperture (NA) of the focusing system
    refractive_index : float, optional
        Refractive index of the medium (default: 1.0 for air/vacuum)
    name : str, optional
        Name/description of the optical system
    
    Examples
    --------
    >>> # Create system for 750 nm laser with NA=0.8 in medium n=1.3 (Romallosa paper)
    >>> system = OpticalSystem(750e-9, 0.8, 1.3, "Romallosa reproduction")
    >>> print(f"Vacuum wavelength: {system.wavelength_vacuum*1e9:.1f} nm")
    >>> print(f"Medium wavelength: {system.wavelength*1e9:.1f} nm")
    >>> print(f"Wave number: {system.k:.2f} rad/m")
    """
    
    def __init__(
        self, 
        wavelength: float,  # This is wavelength in VACUUM
        numerical_aperture: float,
        refractive_index: float = 1.0,
        name: Optional[str] = None
    ):
        # Input validation
        if wavelength <= 0:
            raise ValueError("Wavelength must be positive")
        if numerical_aperture <= 0:
            raise ValueError("Numerical aperture must be positive")
        if numerical_aperture > refractive_index:
            raise ValueError("Numerical aperture cannot exceed refractive index")
        if refractive_index <= 0:
            raise ValueError("Refractive index must be positive")
        
        # Store primary parameters
        self.wavelength_vacuum = float(wavelength)  # Store vacuum wavelength
        self.wavelength = self.wavelength_vacuum / refractive_index  # Medium wavelength
        self.NA = float(numerical_aperture)
        self.n = float(refractive_index)
        self.name = name or f"λ={wavelength*1e9:.0f}nm, NA={numerical_aperture:.2f}, n={refractive_index:.2f}"
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
    
    def _calculate_derived_parameters(self):
        """
        Calculate derived optical parameters using MEDIUM wavelength.
        
        This is critical for matching Romallosa et al. (2003) equations where
        the wave number k = 2π/λ_medium is used throughout.
        """
        self.k = 2 * np.pi / self.wavelength  # k = 2π/λ_medium
        
        # Frequency (same in vacuum and medium)
        self.frequency = LIGHT_SPEED / self.wavelength_vacuum
        self.angular_frequency = 2 * np.pi * self.frequency
        
        # Angular aperture: sin α = NA/n (from definition of NA)
        self.angular_aperture = np.arcsin(self.NA / self.n)
        
        # Rayleigh length using medium wavelength
        # From Gaussian beam theory: zR = πw₀²/λ_medium
        # For diffraction-limited focusing: w₀ ≈ λ/(π·NA), so zR ≈ λ_medium/NA²
        self.rayleigh_length = np.pi * self.wavelength / (self.NA**2)
        
        # Diffraction limits using medium wavelength (Airy pattern theory)
        self.airy_radius = 0.61 * self.wavelength / self.NA  # First zero radius
        self.diffraction_limit = 0.51 * self.wavelength / self.NA  # FWHM ≈ 0.51λ/NA
        
        # Gaussian beam parameters using medium wavelength
        self.beam_waist_limit = self.wavelength / (np.pi * self.NA)  # Minimum w₀
        self.focal_parameter = 2 * self.rayleigh_length  # Total depth of focus = 2zR
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all optical parameters as a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing all optical parameters
        """
        return {
            'wavelength_vacuum': self.wavelength_vacuum,
            'wavelength_medium': self.wavelength,
            'numerical_aperture': self.NA,
            'refractive_index': self.n,
            'name': self.name,
            'wave_number': self.k,
            'frequency': self.frequency,
            'angular_aperture': self.angular_aperture,
            'rayleigh_length': self.rayleigh_length,
            'airy_radius': self.airy_radius,
            'diffraction_limit': self.diffraction_limit,
            'beam_waist_limit': self.beam_waist_limit,
        }
    
    def summary(self) -> str:
        """
        Generate a formatted summary of the optical system.
        
        Returns
        -------
        str
            Formatted summary string
        """
        summary = f"Optical System: {self.name}\n"
        summary += "=" * (len(summary) - 1) + "\n"
        summary += f"Vacuum Wavelength:   {self.wavelength_vacuum*1e9:.1f} nm\n"
        summary += f"Medium Wavelength:   {self.wavelength*1e9:.1f} nm\n"
        summary += f"Numerical Aperture:  {self.NA:.3f}\n"
        summary += f"Refractive Index:    {self.n:.3f}\n"
        summary += f"Angular Aperture:    {np.degrees(self.angular_aperture):.1f}°\n"
        summary += f"Wave Number (k):     {self.k:.2f} rad/m\n"
        summary += f"\nDerived Properties:\n"
        summary += f"Diffraction Limit:   {self.diffraction_limit*1e9:.1f} nm\n"
        summary += f"Airy Radius:         {self.airy_radius*1e9:.1f} nm\n"
        summary += f"Rayleigh Length:     {self.rayleigh_length*1e6:.2f} μm\n"
        summary += f"Min Beam Waist:      {self.beam_waist_limit*1e9:.1f} nm\n"
        return summary
    
    def __repr__(self) -> str:
        """Concise representation."""
        return (f"OpticalSystem(λ_vac={self.wavelength_vacuum*1e9:.1f}nm, "
                f"λ_med={self.wavelength*1e9:.1f}nm, "
                f"NA={self.NA:.2f}, n={self.n:.2f})")
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return self.summary()
    