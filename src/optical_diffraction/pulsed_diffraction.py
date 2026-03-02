"""
Pulsed wave diffraction calculations for ultrafast optics.

This module extends the CW diffraction calculations to handle ultrafast pulsed beams
with arbitrary pulse widths and spectral characteristics.

Key Features:
- Gaussian pulse envelope analysis with linear chirp support
- Spectral bandwidth integration using standard time-bandwidth methods
- Time-bandwidth product calculations  
- Support for few-cycle pulse analysis
- Proper incoherent integration of spectral components
- Parallel processing for improved performance
"""

import numpy as np
from tqdm import tqdm
from typing import Tuple, Union, Optional, Dict, Any
from scipy import integrate
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .diffraction import richards_wolf_vector_diffraction
from .constants import LIGHT_SPEED


class PulsedBeam:
    """
    Represents a pulsed optical beam with Gaussian temporal envelope.
    
    Based on the complex Gaussian pulse model with linear frequency chirp.
    The temporal envelope is: E(t) = A exp(-Γt²) exp(-jωct)
    where Γ = α - jβ is the complex envelope parameter.
    
    Parameters
    ----------
    center_wavelength : float
        Center wavelength λc in meters
    pulse_width_fwhm : float  
        Full width at half maximum pulse duration τ in seconds
    chirp_parameter : float, optional
        Linear chirp parameter β (default: 0 for unchirped pulses)
        - β = 0: Transform-limited (unchirped) pulse
        - β > 0: Positive (red-to-blue) chirp  
        - β < 0: Negative (blue-to-red) chirp
        Units: rad/s²
    time_bandwidth_product : float, optional
        Time-bandwidth product (default: 0.44 for Gaussian pulses)
        
    Notes
    -----
    The chirp parameter β describes linear frequency chirp across the pulse:
    - Real pulses often acquire chirp when propagating through dispersive media
    - Chirp affects both temporal duration and spectral characteristics
    - For β ≠ 0, the actual pulse duration will be longer than transform-limited
    """
    
    def __init__(
        self,
        center_wavelength: float,
        pulse_width_fwhm: float,
        chirp_parameter: float = 0.0,
        time_bandwidth_product: float = 0.44
    ):
        self.lambda_c = center_wavelength
        self.tau = pulse_width_fwhm
        self.beta = chirp_parameter
        self.tbp = time_bandwidth_product
        
        # Calculate derived parameters (Romallosa eq. 8-9)
        self.alpha = 2 * np.log(2) / (self.tau**2)  # From FWHM relationship
        self.gamma = self.alpha - 1j * self.beta  # Complex parameter
        
        # Center frequency and bandwidth
        self.nu_c = LIGHT_SPEED / self.lambda_c
        self.omega_c = 2 * np.pi * self.nu_c
        
        # Spectral bandwidth (Romallosa text after eq. 10)
        self.delta_nu = self.tbp / self.tau
        self.delta_omega = 2 * np.pi * self.delta_nu
        
        # Override for explicit spectral range (for literature comparison)
        self._nu_min_override = None
        self._nu_max_override = None
        
    def temporal_envelope(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate complex temporal envelope E(t).
        
        Uses the complex Gaussian form from Romallosa equation (8):
        E(t) = (2α/π)^(1/4) exp(-Γt²) exp(-jωct)
        where Γ = α - jβ includes both pulse shaping (α) and chirp (β).
        
        Parameters
        ----------
        t : np.ndarray
            Time array in seconds
            
        Returns
        -------
        np.ndarray (complex)
            Complex temporal envelope including chirp effects
            
        Notes
        -----
        The chirp parameter β causes:
        - Instantaneous frequency variation across the pulse
        - Pulse broadening beyond the transform limit
        - Phase modulation that affects spectral characteristics
        """
        amplitude = (2 * self.alpha / np.pi)**(1/4)
        envelope = np.exp(-self.gamma * t**2)  # γ = α - jβ includes chirp
        carrier = np.exp(-1j * self.omega_c * t)
        
        return amplitude * envelope * carrier
    
    def intensity_envelope(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate intensity envelope |E(t)|².
        
        From Romallosa equation (9):
        |E(t)|² = (2α/π)^(1/2) exp[-(4ln2)(t/τ)²]
        
        Parameters
        ----------
        t : np.ndarray
            Time array in seconds
            
        Returns
        -------
        np.ndarray (real)
            Intensity envelope
        """
        amplitude = (2 * self.alpha / np.pi)**(1/2)
        gaussian = np.exp(-4 * np.log(2) * (t / self.tau)**2)
        
        return amplitude * gaussian
    
    def spectral_distribution(self, nu: np.ndarray) -> np.ndarray:
        """
        Calculate power spectral density |E(ν)|².
        
        Uses the exact form from Romallosa et al. (2003):
        |E(ω)|² = (1/2π)^(1/4) (1/a)^(1/2) exp[-(ω-ωc)²/(2a)]
        
        Parameters
        ----------
        nu : np.ndarray
            Frequency array in Hz
            
        Returns
        -------
        np.ndarray (real)
            Power spectral density
        """
        # Convert frequency to angular frequency for consistency with paper
        omega = 2 * np.pi * nu
        omega_c = 2 * np.pi * self.nu_c
        
        # Use parameter 'a' from Romallosa paper
        # From Fourier transform relationship: a = 1/(4*alpha) where alpha = 2*ln(2)/tau^2
        # a = self.tau**2 / (8 * np.log(2)) 
        a = 1 / (self.tau**2 / (2 * np.log(2)))  
        
        # Romallosa equation: |E(ω)|² = (1/2π)^(1/4) (1/a)^(1/2) exp[-(ω-ωc)²/(2a)]
        amplitude = (1/(2*np.pi))**(1/4) * (1/a)**(1/2)
        gaussian = np.exp(-(omega - omega_c)**2 / (2*a))
        
        return amplitude * gaussian
    
    def get_spectral_range(self, bandwidth_factor: float = 1.0) -> Tuple[float, float]:
        """
        Get frequency range for spectral integration.
        
        For bandwidth_factor=1.0, integrates over νc ± Δν/2 which corresponds
        to the FWHM of the spectral distribution.
        
        Parameters
        ----------
        bandwidth_factor : float
            Multiple of spectral FWHM to include (default: 1.0 = FWHM)
            - 1.0: Integrates over FWHM
            - 2.0: Integrates over 2×FWHM 
            - etc.
            
        Returns
        -------
        tuple
            (nu_min, nu_max) in Hz
        """
        # Check for manual override first (for literature comparison)
        if self._nu_min_override is not None:
            return self._nu_min_override, self._nu_max_override
            
        # Use standard method: νc ± (bandwidth_factor × Δν/2)
        half_range = bandwidth_factor * self.delta_nu / 2
        nu_min = self.nu_c - half_range
        nu_max = self.nu_c + half_range
        
        # Handle extreme few-cycle cases where nu_min might be negative
        if nu_min <= 0:
            # For few-cycle pulses, use asymmetric range to avoid negative frequencies
            nu_min = self.nu_c / 100  # Small positive frequency
            total_bandwidth = bandwidth_factor * self.delta_nu
            nu_max = nu_min + total_bandwidth
            
        return nu_min, nu_max
    
    def get_spectral_range_wavelengths(self, bandwidth_factor: float = 1.0) -> Tuple[float, float]:
        """
        Get wavelength range for spectral integration.
        
        Parameters
        ----------
        bandwidth_factor : float
            Multiple of spectral FWHM to include (default: 1.0 = FWHM)
            
        Returns
        -------
        tuple
            (lambda_min, lambda_max) in meters
        """
        nu_min, nu_max = self.get_spectral_range(bandwidth_factor)
        # Convert frequency to wavelength: λ = c/ν
        # Higher frequency → shorter wavelength, so nu_max → lambda_min
        lambda_min = LIGHT_SPEED / nu_max
        lambda_max = LIGHT_SPEED / nu_min
        return lambda_min, lambda_max
    
    def get_spectral_frequencies(self, n_points: int = 1000, bandwidth_factor: float = 1.0) -> np.ndarray:
        """
        Generate frequency array for spectral calculations.
        
        Parameters
        ----------
        n_points : int
            Number of frequency points (default: 1000)
        bandwidth_factor : float
            Multiple of spectral FWHM to include (default: 1.0 = FWHM)
            
        Returns
        -------
        np.ndarray
            Frequency array in Hz
        """
        nu_min, nu_max = self.get_spectral_range(bandwidth_factor)
        return np.linspace(nu_min, nu_max, n_points)
    
    def get_spectral_wavelengths(self, n_points: int = 1000, bandwidth_factor: float = 1.0) -> np.ndarray:
        """
        Generate wavelength array for spectral calculations.
        
        Parameters
        ----------
        n_points : int
            Number of wavelength points (default: 1000)
        bandwidth_factor : float
            Multiple of spectral FWHM to include (default: 1.0 = FWHM)
            
        Returns
        -------
        np.ndarray
            Wavelength array in meters
        """
        frequencies = self.get_spectral_frequencies(n_points, bandwidth_factor)
        # Convert to wavelengths and reverse order (short to long wavelength)
        wavelengths = LIGHT_SPEED / frequencies
        return wavelengths[::-1]  # Reverse to go from short to long wavelength
    
    def set_spectral_range(self, nu_min: float, nu_max: float):
        """
        Set explicit spectral integration range.
        
        Parameters
        ----------
        nu_min, nu_max : float
            Frequency limits in Hz
        """
        self._nu_min_override = nu_min
        self._nu_max_override = nu_max
        
    def set_spectral_range_wavelengths(self, lambda_min: float, lambda_max: float):
        """
        Set spectral range using wavelength limits.
        
        Parameters
        ----------
        lambda_min, lambda_max : float
            Wavelength limits in meters
        """
        nu_max = LIGHT_SPEED / lambda_min  # Short wavelength → high frequency
        nu_min = LIGHT_SPEED / lambda_max  # Long wavelength → low frequency
        self.set_spectral_range(nu_min, nu_max)
        
    def clear_spectral_range_override(self):
        """Clear any manual spectral range override."""
        self._nu_min_override = None
        self._nu_max_override = None
    
    def coherence_length(self) -> float:
        """
        Calculate coherence length c*τ.
        
        Returns
        -------
        float
            Coherence length in meters
        """
        return LIGHT_SPEED * self.tau
    
    def is_few_cycle(self) -> bool:
        """
        Check if pulse is few-cycle (coherence length < λ/2).
        
        Returns
        -------
        bool
            True if few-cycle pulse
        """
        return self.coherence_length() < self.lambda_c / 2
    
    def summary(self) -> str:
        """Generate summary of pulse parameters."""
        coherence = self.coherence_length()
        nyquist_limit = self.lambda_c / 2
        
        # Get current spectral range using built-in methods
        lambda_min, lambda_max = self.get_spectral_range_wavelengths(bandwidth_factor=1.0)
        lambda_min_nm = lambda_min * 1e9  # nm
        lambda_max_um = lambda_max * 1e6   # μm
        
        summary = f"Pulsed Beam Parameters:\n"
        summary += f"Center wavelength: {self.lambda_c*1e9:.1f} nm\n"
        summary += f"Pulse width (FWHM): {self.tau*1e15:.1f} fs\n"
        summary += f"Spectral bandwidth: {self.delta_nu*1e-12:.2f} THz\n"
        summary += f"Integration range: {lambda_min_nm:.0f} nm to {lambda_max_um:.2f} μm\n"
        summary += f"Coherence length: {coherence*1e9:.1f} nm\n"
        summary += f"Nyquist limit (λ/2): {nyquist_limit*1e9:.1f} nm\n"
        summary += f"Few-cycle pulse: {'Yes' if self.is_few_cycle() else 'No'}\n"
        
        if self._nu_min_override is not None:
            summary += f"Using manual spectral range override\n"
        
        return summary


def _calculate_frequency_component(args):
    """
    Calculate field contribution for a single frequency component.
    """
    (nu, weight, wavelength, system_na, system_n, 
     r_coords, z_coords, polarization_angle, addition_method) = args
    
    # Reconstruct system from parameters
    from .systems import OpticalSystem
    temp_system = OpticalSystem(
        wavelength=wavelength,
        numerical_aperture=system_na,
        refractive_index=system_n
    )
    
    # Calculate CW field for this frequency
    Ex_cw, Ey_cw, Ez_cw = richards_wolf_vector_diffraction(
        temp_system, r_coords, z_coords, polarization_angle
    )
    
    if addition_method == 'coherent':
        return weight * Ex_cw, weight * Ey_cw, weight * Ez_cw
    else:  # incoherent
        intensity_x = weight * np.abs(Ex_cw)**2
        intensity_y = weight * np.abs(Ey_cw)**2
        intensity_z = weight * np.abs(Ez_cw)**2
        return intensity_x, intensity_y, intensity_z


def pulsed_richards_wolf_diffraction(
    system,  # OpticalSystem
    pulsed_beam: PulsedBeam,
    r_coords: np.ndarray,
    z_coords: np.ndarray,
    n_frequencies: int = 201,
    polarization_angle: float = 0.0,
    addition_method: str = 'incoherent',
    n_processes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate pulsed beam diffraction using spectral integration.
    
    Two methods available:
    1. 'coherent': Add field amplitudes then take intensity (like your example)
    2. 'incoherent': Add intensities directly (Romallosa equation 10)
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    pulsed_beam : PulsedBeam
        Pulsed beam parameters
    r_coords, z_coords : np.ndarray
        Spatial coordinates in meters
    n_frequencies : int
        Number of frequency components for integration (default: 201)
    polarization_angle : float
        Polarization angle in radians
    addition_method : str
        'coherent' for field addition, 'incoherent' for intensity addition
    n_processes : int, optional
        Number of parallel processes. If None, uses all CPU cores.
        Set to 1 to disable parallelization.
        
    Returns
    -------
    Ex, Ey, Ez : np.ndarray (complex)
        Electric field components
    """
    
    # Handle process count
    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_frequencies)
    elif n_processes == 1:
        # Serial processing (original behavior)
        pass
    
    # Get spectral range
    nu_min, nu_max = pulsed_beam.get_spectral_range(bandwidth_factor=3.0)
    
    # Create frequency array
    frequencies = np.linspace(nu_min, nu_max, n_frequencies)
    
    # Get spectral weights
    spectral_weights = pulsed_beam.spectral_distribution(frequencies)
    
    # Initialize arrays
    shape = np.broadcast_shapes(np.asarray(r_coords).shape, np.asarray(z_coords).shape)
    
    if addition_method == 'coherent':
        Ex_total = np.zeros(shape, dtype=complex)
        Ey_total = np.zeros(shape, dtype=complex)
        Ez_total = np.zeros(shape, dtype=complex)
    else:
        intensity_x_total = np.zeros(shape, dtype=float)
        intensity_y_total = np.zeros(shape, dtype=float)
        intensity_z_total = np.zeros(shape, dtype=float)
    
    if n_processes == 1:
        # Serial processing
        for nu, weight in tqdm(zip(frequencies, spectral_weights)):
            # Calculate wavelength for this frequency
            wavelength = LIGHT_SPEED / nu
            
            # Create temporary system with this wavelength
            temp_system = type(system)(
                wavelength=wavelength,
                numerical_aperture=system.NA,
                refractive_index=system.n,
                name=f"Temp system λ={wavelength*1e9:.1f}nm"
            )
            
            # Calculate CW field for this frequency
            Ex_cw, Ey_cw, Ez_cw = richards_wolf_vector_diffraction(
                temp_system, r_coords, z_coords, polarization_angle
            )
            
            if addition_method == 'coherent':
                # Add weighted field amplitudes (coherent addition)
                Ex_total += weight * Ex_cw
                Ey_total += weight * Ey_cw
                Ez_total += weight * Ez_cw
            else:  # incoherent addition
                # Add weighted intensities (Romallosa equation 10)
                intensity_x_total += weight * np.abs(Ex_cw)**2
                intensity_y_total += weight * np.abs(Ey_cw)**2
                intensity_z_total += weight * np.abs(Ez_cw)**2
    
    else:
        # Parallel processing
        # Prepare arguments for parallel processing
        args_list = []
        for nu, weight in zip(frequencies, spectral_weights):
            wavelength = LIGHT_SPEED / nu
            args_list.append((
                nu, weight, wavelength, system.NA, system.n,
                r_coords, z_coords, polarization_angle, addition_method
            ))
        
        # Parallel computation
        chunk_size = max(1, len(args_list) // (n_processes * 4))
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(tqdm(
                executor.map(_calculate_frequency_component, args_list, chunksize=chunk_size),
                total=len(args_list),
                desc="Computing frequency components"
            ))
        
        # Accumulate results
        for result in results:
            if addition_method == 'coherent':
                Ex_contrib, Ey_contrib, Ez_contrib = result
                Ex_total += Ex_contrib
                Ey_total += Ey_contrib
                Ez_total += Ez_contrib
            else:
                intensity_x_contrib, intensity_y_contrib, intensity_z_contrib = result
                intensity_x_total += intensity_x_contrib
                intensity_y_total += intensity_y_contrib
                intensity_z_total += intensity_z_contrib
    
    if addition_method == 'incoherent':
        # Convert back to effective field amplitudes
        # Note: Phase information is lost in intensity integration
        Ex_total = np.sqrt(intensity_x_total + 0j)  # Make complex
        Ey_total = np.sqrt(intensity_y_total + 0j) 
        Ez_total = np.sqrt(intensity_z_total + 0j)
    
    return Ex_total, Ey_total, Ez_total


def create_pulsed_axial_profile(
    system,  # OpticalSystem
    pulsed_beam: PulsedBeam,
    z_range: float = None,
    nz: int = 201,
    r_constant: float = 0.0,
    n_frequencies: int = 201,
    addition_method: str = 'incoherent',
    n_processes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate axial profile for pulsed beam.
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    pulsed_beam : PulsedBeam
        Pulsed beam parameters
    z_range : float, optional
        Axial range (±z_range). Default: 4 × Rayleigh length
    nz : int
        Number of axial points
    r_constant : float
        Constant radial position in meters (default: 0 for on-axis)
    n_frequencies : int
        Number of frequency components (default: 201)
    addition_method : str
        'coherent' for field addition, 'incoherent' for intensity addition
    n_processes : int, optional
        Number of parallel processes. If None, uses all CPU cores.
        
    Returns
    -------
    z_coords : np.ndarray
        Axial coordinates
    Ex, Ey, Ez : np.ndarray (complex)
        Electric field components
    """
    if z_range is None:
        z_range = 4 * system.rayleigh_length
        
    z_coords = np.linspace(-z_range, z_range, nz)
    r_coords = np.full_like(z_coords, r_constant)  # Constant r
    
    Ex, Ey, Ez = pulsed_richards_wolf_diffraction(
        system, pulsed_beam, r_coords, z_coords, n_frequencies, 
        addition_method=addition_method, n_processes=n_processes
    )
    
    return z_coords, Ex, Ey, Ez


def create_pulsed_transverse_profile(
    system,  # OpticalSystem
    pulsed_beam: PulsedBeam,
    r_range: float = None,
    nr: int = 201,
    z_constant: float = 0.0,
    n_frequencies: int = 201,
    addition_method: str = 'incoherent',
    n_processes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate transverse profile for pulsed beam.
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    pulsed_beam : PulsedBeam
        Pulsed beam parameters
    r_range : float, optional
        Radial range (0 to r_range). Default: 4 × Airy radius
    nr : int
        Number of radial points
    z_constant : float
        Constant axial position in meters (default: 0 for at focus)
    n_frequencies : int
        Number of frequency components (default: 201)
    addition_method : str
        'coherent' for field addition, 'incoherent' for intensity addition
    n_processes : int, optional
        Number of parallel processes. If None, uses all CPU cores.
        
    Returns
    -------
    r_coords : np.ndarray
        Radial coordinates
    Ex, Ey, Ez : np.ndarray (complex)
        Electric field components
    """
    if r_range is None:
        r_range = 4 * system.airy_radius
        
    r_coords = np.linspace(-r_range, r_range, nr)
    z_coords = np.full_like(r_coords, z_constant)  # Constant z
    
    Ex, Ey, Ez = pulsed_richards_wolf_diffraction(
        system, pulsed_beam, r_coords, z_coords, n_frequencies,
        addition_method=addition_method, n_processes=n_processes
    )
    
    return r_coords, Ex, Ey, Ez