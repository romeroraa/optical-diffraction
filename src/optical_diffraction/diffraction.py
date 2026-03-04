"""
Diffraction calculations implementing Richards-Wolf vector diffraction theory
and scalar diffraction theories for comparison.

This module implements both scalar and vector diffraction theories:
- Richards-Wolf vector diffraction (exact theory)
- Scalar circular aperture diffraction (paraxial limit)
- Gaussian beam scalar diffraction (for comparison)
- Profile creation functions

Fixed version to match Romallosa et al. (2003) paper exactly.
"""

import numpy as np
from typing import Tuple, Union, Optional, Literal
from scipy.special import jv  # Bessel functions
from scipy import integrate
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def _calculate_single_integral_point(args):
    """
    Calculate integral for a single (u, v) point.
    Used by parallel processing.
    """
    (integrand_func, u_val, v_val, alpha, integration_params) = args
    
    try:
        result, _ = integrate.quad(
            lambda theta: integrand_func(theta, u_val, v_val),
            0, alpha,
            complex_func=True,
            **integration_params
        )
        return result
    except:
        return 0.0 + 0.0j


def _I0_integrand(theta, u_val, v_val, sin_alpha, sin2_alpha):
    """I0 integrand function with robust handling."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Handle theta=0 case more carefully
    if abs(sin_theta) < 1e-15:
        return 0.0 + 0.0j
        
    apodization = np.sqrt(cos_theta)
    geometric = sin_theta * (1 + cos_theta)
    bessel_arg = v_val * sin_theta / sin_alpha
    bessel = jv(0, bessel_arg)
    phase_arg = u_val * cos_theta / sin2_alpha
    phase = np.exp(1j * phase_arg)

    return apodization * geometric * bessel * phase


def _I1_integrand(theta, u_val, v_val, sin_alpha, sin2_alpha):
    """I1 integrand function with robust handling."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if abs(sin_theta) < 1e-15:
        return 0.0 + 0.0j

    apodization = np.sqrt(cos_theta)
    geometric = sin_theta**2
    bessel_arg = v_val * sin_theta / sin_alpha
    bessel = jv(1, bessel_arg)
    phase_arg = u_val * cos_theta / sin2_alpha
    phase = np.exp(1j * phase_arg)

    return apodization * geometric * bessel * phase


def _I2_integrand(theta, u_val, v_val, sin_alpha, sin2_alpha):
    """I2 integrand function with robust handling."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if abs(sin_theta) < 1e-15:
        return 0.0 + 0.0j

    apodization = np.sqrt(cos_theta)
    geometric = sin_theta * (1 - cos_theta)
    bessel_arg = v_val * sin_theta / sin_alpha
    bessel = jv(2, bessel_arg)
        
    phase_arg = u_val * cos_theta / sin2_alpha
    phase = np.exp(1j * phase_arg)
    
    return apodization * geometric * bessel * phase


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================

def convert_to_optical_coordinates(
    system,  # OpticalSystem
    r: np.ndarray,
    z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert physical coordinates to optical coordinates u, v.
    
    From Romallosa et al. (2003) equations:
    u = (k sin²α)z = (2π/λ) * (sin²α) * z  
    v = (k sin α)r = (2π/λ) * (sin α) * r
    
    where sin α = NA/n (in the medium) and λ is wavelength in medium
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system containing wavelength and NA
    r : np.ndarray
        Radial coordinates in meters
    z : np.ndarray  
        Axial coordinates in meters
        
    Returns
    -------
    u, v : np.ndarray
        Optical coordinates
    """
    # Wave number in the medium (CRITICAL: use wavelength in medium!)
    k = system.k  # This should be 2π/λ_medium
    
    # Angular aperture: sin α = NA/n
    sin_alpha = system.NA / system.n  # This is correct
    
    # Optical coordinates from Romallosa paper
    u = k * (sin_alpha**2) * z  # Axial optical coordinate
    v = k * sin_alpha * r       # Radial optical coordinate
    
    return u, v


# =============================================================================
# RICHARDS-WOLF VECTOR DIFFRACTION INTEGRALS
# =============================================================================

def calculate_integral_I0(
    system,  # OpticalSystem
    u: np.ndarray,
    v: np.ndarray,
    n_processes: Optional[int] = None
) -> np.ndarray:
    """
    Calculate I₀ integral from Romallosa equation (5).
    
    I₀(u,v) = ∫₀^α cos^(1/2)(θ) sin(θ) (1+cos(θ)) J₀(v sin(θ)/sin(α)) 
              × exp(ju cos(θ)/sin²(α)) dθ
              
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    u, v : np.ndarray
        Optical coordinates
    n_processes : int, optional
        Number of parallel processes. If None, uses all CPU cores.
        Set to 1 to disable parallelization.
        
    Returns
    -------
    I0 : np.ndarray (complex)
        I₀ integral values
    """
    alpha = system.angular_aperture
    sin_alpha = np.sin(alpha)
    sin2_alpha = sin_alpha**2
    
    # More robust integration parameters
    integration_params = {
        'limit': 200,           # Increased from 100
        'epsabs': 1e-10,        # Slightly relaxed from 1e-12
        'epsrel': 1e-8,         # Slightly relaxed from 1e-10  
        'maxp1': 200,           # Maximum number of extrapolation points
        'full_output': False    # Don't return extra info
    }
    
    u_flat = np.atleast_1d(u).flatten()
    v_flat = np.atleast_1d(v).flatten()
    
    # Handle process count
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(u_flat))
    elif n_processes == 1:
        # Serial processing (original behavior)
        pass
    
    if n_processes == 1:
        # Serial implementation with improved error handling
        def integrand(theta, u_val, v_val):
            return _I0_integrand(theta, u_val, v_val, sin_alpha, sin2_alpha)
        
        I0_flat = np.zeros_like(u_flat, dtype=complex)
        
        for i, (u_val, v_val) in enumerate(zip(u_flat, v_flat)):
            try:
                result, _ = integrate.quad(
                    lambda theta: integrand(theta, u_val, v_val),
                    0, alpha, 
                    complex_func=True,
                    **integration_params
                )
                I0_flat[i] = result
            except Exception as e:
                # More robust error handling - only print if really needed
                I0_flat[i] = 0.0 + 0.0j
    else:
        # Parallel processing
        integrand_func = partial(
            _I0_integrand,
            sin_alpha=sin_alpha,
            sin2_alpha=sin2_alpha
        )
        
        n_processes = min(n_processes, len(u_flat))
        
        args_list = [
            (integrand_func, u_val, v_val, alpha, integration_params)
            for u_val, v_val in zip(u_flat, v_flat)
        ]
        
        # Parallel computation
        chunk_size = max(1, len(args_list) // (n_processes * 4))
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            I0_flat = list(tqdm(
                executor.map(_calculate_single_integral_point, args_list, chunksize=chunk_size),
                total=len(args_list),
                desc="Computing I0 integral"
            ))
        
        I0_flat = np.array(I0_flat, dtype=complex)
    
    return I0_flat.reshape(np.atleast_1d(u).shape)


def calculate_integral_I1(
    system,  # OpticalSystem
    u: np.ndarray,
    v: np.ndarray,
    n_processes: Optional[int] = None
) -> np.ndarray:
    """
    Calculate I₁ integral from Romallosa equation (6).
    
    I₁(u,v) = ∫₀^α cos^(1/2)(θ) sin²(θ) J₁(v sin(θ)/sin(α)) 
              × exp(ju cos(θ)/sin²(α)) dθ
              
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    u, v : np.ndarray
        Optical coordinates
    n_processes : int, optional
        Number of parallel processes. If None, uses all CPU cores.
        Set to 1 to disable parallelization.
        
    Returns
    -------
    I1 : np.ndarray (complex)
        I₁ integral values
    """
    alpha = system.angular_aperture
    sin_alpha = np.sin(alpha)
    sin2_alpha = sin_alpha**2
    
    integration_params = {
        'limit': 200,
        'epsabs': 1e-10,
        'epsrel': 1e-8,
        'maxp1': 200,
        'full_output': False
    }
    
    u_flat = np.atleast_1d(u).flatten()
    v_flat = np.atleast_1d(v).flatten()
    
    # Handle process count
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(u_flat))
    elif n_processes == 1:
        pass
    
    if n_processes == 1:
        # Serial implementation
        def integrand(theta, u_val, v_val):
            return _I1_integrand(theta, u_val, v_val, sin_alpha, sin2_alpha)
        
        I1_flat = np.zeros_like(u_flat, dtype=complex)
        
        for i, (u_val, v_val) in enumerate(zip(u_flat, v_flat)):
            try:
                result, _ = integrate.quad(
                    lambda theta: integrand(theta, u_val, v_val),
                    0, alpha,
                    complex_func=True,
                    **integration_params
                )
                I1_flat[i] = result
            except:
                I1_flat[i] = 0.0 + 0.0j
    else:
        # Parallel processing
        integrand_func = partial(
            _I1_integrand,
            sin_alpha=sin_alpha,
            sin2_alpha=sin2_alpha
        )
        
        n_processes = min(n_processes, len(u_flat))
        
        args_list = [
            (integrand_func, u_val, v_val, alpha, integration_params)
            for u_val, v_val in zip(u_flat, v_flat)
        ]
        
        # Parallel computation
        chunk_size = max(1, len(args_list) // (n_processes * 4))
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            I1_flat = list(tqdm(
                executor.map(_calculate_single_integral_point, args_list, chunksize=chunk_size),
                total=len(args_list),
                desc="Computing I1 integral"
            ))
        
        I1_flat = np.array(I1_flat, dtype=complex)
    
    return I1_flat.reshape(np.atleast_1d(u).shape)


def calculate_integral_I2(
    system,  # OpticalSystem
    u: np.ndarray,
    v: np.ndarray,
    n_processes: Optional[int] = None
) -> np.ndarray:
    """
    Calculate I₂ integral from Romallosa equation (7).
    
    I₂(u,v) = ∫₀^α cos^(1/2)(θ) sin(θ) (1-cos(θ)) J₂(v sin(θ)/sin(α)) 
              × exp(ju cos(θ)/sin²(α)) dθ
              
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    u, v : np.ndarray
        Optical coordinates
    n_processes : int, optional
        Number of parallel processes. If None, uses all CPU cores.
        Set to 1 to disable parallelization.
        
    Returns
    -------
    I2 : np.ndarray (complex)
        I₂ integral values
    """
    alpha = system.angular_aperture
    sin_alpha = np.sin(alpha)
    sin2_alpha = sin_alpha**2
    
    integration_params = {
        'limit': 200,
        'epsabs': 1e-10,
        'epsrel': 1e-8,
        'maxp1': 200,
        'full_output': False
    }
    
    u_flat = np.atleast_1d(u).flatten()
    v_flat = np.atleast_1d(v).flatten()
    
    # Handle process count
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(u_flat))
    elif n_processes == 1:
        pass
    
    if n_processes == 1:
        # Serial implementation
        def integrand(theta, u_val, v_val):
            return _I2_integrand(theta, u_val, v_val, sin_alpha, sin2_alpha)
        
        I2_flat = np.zeros_like(u_flat, dtype=complex)
        
        for i, (u_val, v_val) in enumerate(zip(u_flat, v_flat)):
            try:
                result, _ = integrate.quad(
                    lambda theta: integrand(theta, u_val, v_val),
                    0, alpha,
                    complex_func=True,
                    **integration_params
                )
                I2_flat[i] = result
            except:
                I2_flat[i] = 0.0 + 0.0j
    else:
        # Parallel processing
        integrand_func = partial(
            _I2_integrand,
            sin_alpha=sin_alpha,
            sin2_alpha=sin2_alpha
        )
        
        n_processes = min(n_processes, len(u_flat))
        
        args_list = [
            (integrand_func, u_val, v_val, alpha, integration_params)
            for u_val, v_val in zip(u_flat, v_flat)
        ]
        
        # Parallel computation
        chunk_size = max(1, len(args_list) // (n_processes * 4))
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            I2_flat = list(tqdm(
                executor.map(_calculate_single_integral_point, args_list, chunksize=chunk_size),
                total=len(args_list),
                desc="Computing I2 integral"
            ))
        
        I2_flat = np.array(I2_flat, dtype=complex)
    
    return I2_flat.reshape(np.atleast_1d(u).shape)


# =============================================================================
# FIELD COMPONENT CALCULATIONS
# =============================================================================

def calculate_field_Ex(
    I0: np.ndarray,
    I2: np.ndarray,
    phi: np.ndarray = 0.0
) -> np.ndarray:
    """
    Calculate Ex field component from Richards-Wolf theory.

    Ex(P) = -j(I₀ - I₂ cos 2φ)

    For linear x-polarization at φ=0: cos(2φ)=1, so Ex = -j(I₀ - I₂)
    """
    return -1j * (I0 - I2 * np.cos(2 * phi))


def calculate_field_Ey(
    I2: np.ndarray,
    phi: np.ndarray = 0.0
) -> np.ndarray:
    """
    Calculate Ey field component from Richards-Wolf theory.

    Ey(P) = -jI₂ sin 2φ

    For linear x-polarization at φ=0: sin(2φ)=0, so Ey = 0
    """
    return -1j * I2 * np.sin(2 * phi)


def calculate_field_Ez(
    I1: np.ndarray,
    phi: np.ndarray = 0.0
) -> np.ndarray:
    """
    Calculate Ez field component from Romallosa equation (4).
    
    Ez(P) = -2I₁ cos φ
    
    For linear x-polarization: φ = 0, so cos φ = 1
    
    Note: Factor is -2, not -2j like Ex, Ey
    """
    return -2 * I1 * np.cos(phi)  # Note: NO j factor here!


# =============================================================================
# MAIN DIFFRACTION FUNCTIONS
# =============================================================================

def richards_wolf_vector_diffraction(
    system,  # OpticalSystem
    r_coords: np.ndarray, 
    z_coords: np.ndarray,
    polarization_angle: float = 0.0,
    n_processes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Richards-Wolf vector diffraction using Romallosa equations (1-7).
    
    Implements the full Richards-Wolf diffraction theory for 
    linearly polarized input beam.
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    r_coords, z_coords : np.ndarray
        Spatial coordinates in meters
    polarization_angle : float
        Polarization angle in radians (0 = x-polarized)
    n_processes : int, optional
        Number of parallel processes. If None, uses all CPU cores.
        Set to 1 to disable parallelization.
        
    Returns
    -------
    Ex, Ey, Ez : np.ndarray (complex)
        Electric field components from Romallosa equations (2-4)
    """
    # Convert to optical coordinates (Romallosa context)
    u, v = convert_to_optical_coordinates(system, r_coords, z_coords)
    
    # Calculate the three fundamental integrals (Romallosa eqs 5-7)
    I0 = calculate_integral_I0(system, u, v, n_processes)
    I1 = calculate_integral_I1(system, u, v, n_processes)
    I2 = calculate_integral_I2(system, u, v, n_processes)
    
    # Calculate field components (Romallosa eqs 2-4)
    Ex = calculate_field_Ex(I0, I2, phi=polarization_angle)
    Ey = calculate_field_Ey(I2, phi=polarization_angle)
    Ez = calculate_field_Ez(I1, phi=polarization_angle)
    
    return Ex, Ey, Ez


def scalar_circular_aperture_diffraction(
    system,  # OpticalSystem
    r_coords: np.ndarray,
    z_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scalar diffraction through circular aperture.
    
    This is the low-NA limit of Richards-Wolf theory where vector
    effects are negligible. Uses only the I₀ integral without
    apodization factors.
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    r_coords, z_coords : np.ndarray
        Spatial coordinates in meters
        
    Returns
    -------
    Ex, Ey, Ez : np.ndarray (complex)
        Field components (Ey=Ez=0 for scalar theory)
    """
    k = system.k
    alpha = system.angular_aperture
    sin_alpha = np.sin(alpha)
    
    # Convert to optical coordinates
    u, v = convert_to_optical_coordinates(system, r_coords, z_coords)
    
    # Scalar I0 integral (no apodization, no higher-order terms)
    def scalar_I0_integrand(theta, u_val, v_val):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        if abs(sin_theta) < 1e-15:
            return 0.0 + 0.0j
            
        # Scalar version: no cos^(1/2) apodization
        geometric = sin_theta
        bessel_arg = v_val * sin_theta / sin_alpha
        
        if abs(bessel_arg) > 100:
            bessel = 0.0
        else:
            bessel = jv(0, bessel_arg)  # J₀ only
            
        phase_arg = u_val * cos_theta / (sin_alpha**2)
        phase = np.exp(1j * phase_arg)
        
        return geometric * bessel * phase
    
    # Calculate scalar integral
    u_flat = np.atleast_1d(u).flatten()
    v_flat = np.atleast_1d(v).flatten()
    I0_scalar = np.zeros_like(u_flat, dtype=complex)
    
    for i, (u_val, v_val) in enumerate(zip(u_flat, v_flat)):
        try:
            result, _ = integrate.quad(
                lambda theta: scalar_I0_integrand(theta, u_val, v_val),
                0, alpha, 
                complex_func=True,
                limit=200,
                epsabs=1e-10,
                epsrel=1e-8
            )
            I0_scalar[i] = result
        except:
            I0_scalar[i] = 0.0 + 0.0j
    
    I0_scalar = I0_scalar.reshape(np.atleast_1d(u).shape)
    
    # For scalar theory: only Ex component
    Ex = -1j * I0_scalar
    Ey = np.zeros_like(Ex, dtype=complex)
    Ez = np.zeros_like(Ex, dtype=complex)
    
    return Ex, Ey, Ez


def scalar_diffraction_focused_field(
    system,  # OpticalSystem
    r_coords: np.ndarray,
    z_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scalar diffraction for focused Gaussian beam (paraxial approximation).
    
    Based on standard Gaussian beam propagation equations.
    Returns field components where Ex carries all the field and Ey=Ez=0.
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    r_coords, z_coords : np.ndarray
        Spatial coordinates in meters
        
    Returns
    -------
    Ex, Ey, Ez : np.ndarray (complex)
        Field components (Ey=Ez=0 for scalar theory)
    """
    # Gaussian beam parameters using medium wavelength
    k = system.k  # Wave number in medium
    w0 = system.airy_radius  # Use Airy radius as beam waist approximation
    zR = system.rayleigh_length
    
    # Handle z=0 case to avoid division by zero
    z_coords_safe = np.where(np.abs(z_coords) < 1e-15, 1e-15, z_coords)
    
    # Beam parameters at position z
    w_z = w0 * np.sqrt(1 + (z_coords_safe / zR)**2)
    R_z = z_coords_safe * (1 + (zR / z_coords_safe)**2)
    gouy_phase = np.arctan(z_coords_safe / zR)
    
    # Gaussian field amplitude
    amplitude = (w0 / w_z) * np.exp(-(r_coords / w_z)**2)
    
    # Phase terms
    phase = k * z_coords + k * (r_coords**2) / (2 * R_z) - gouy_phase
    
    # Total field
    Ex = amplitude * np.exp(1j * phase)
    Ey = np.zeros_like(Ex, dtype=complex)
    Ez = np.zeros_like(Ex, dtype=complex)
    
    return Ex, Ey, Ez

# =============================================================================
# PROFILE CREATION FUNCTIONS
# =============================================================================

def create_axial_profile(
    system,  # OpticalSystem
    z_range: float = None,
    nz: int = 201,
    r_constant: float = 0.0,
    method: Literal['scalar', 'vector', 'gaussian'] = 'vector'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate axial field profile Ex, Ey, Ez(z, r=r_constant).
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    z_range : float, optional
        Axial range (±z_range). Default: 4 × Rayleigh length
    nz : int
        Number of axial points
    r_constant : float
        Constant radial position in meters (default: 0 for on-axis)
    method : str
        'vector': Richards-Wolf vector diffraction
        'scalar': Scalar circular aperture 
        'gaussian': Gaussian beam propagation
        
    Returns
    -------
    z_coords : np.ndarray
        Axial coordinates
    Ex, Ey, Ez : np.ndarray (complex)
        Electric field components
    """
    if z_range is None:
        z_range = 4 * system.rayleigh_length
        
    # From negative to positive bounds
    z_coords = np.linspace(-z_range, z_range, nz)
    r_coords = np.full_like(z_coords, r_constant)  # Constant r value
    
    if method == 'gaussian':
        Ex, Ey, Ez = scalar_diffraction_focused_field(system, r_coords, z_coords)
    elif method == 'scalar':
        Ex, Ey, Ez = scalar_circular_aperture_diffraction(system, r_coords, z_coords)
    else:  # vector
        Ex, Ey, Ez = richards_wolf_vector_diffraction(system, r_coords, z_coords)
        
    return z_coords, Ex, Ey, Ez


def create_transverse_profile(
    system,  # OpticalSystem
    r_range: float = None,
    nr: int = 201,
    z_constant: float = 0.0,
    method: Literal['scalar', 'vector', 'gaussian'] = 'vector'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate transverse field profile Ex, Ey, Ez(r, z=z_constant).
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    r_range : float, optional
        Radial range (0 to r_range). Default: 4 × Airy radius
    nr : int
        Number of radial points
    z_constant : float
        Constant axial position in meters (default: 0 for at focus)
    method : str
        Diffraction method to use
        
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
    z_coords = np.full_like(r_coords, z_constant)  # Constant z value
    
    if method == 'gaussian':
        Ex, Ey, Ez = scalar_diffraction_focused_field(system, r_coords, z_coords)
    elif method == 'scalar':
        Ex, Ey, Ez = scalar_circular_aperture_diffraction(system, r_coords, z_coords)
    else:  # vector
        Ex, Ey, Ez = richards_wolf_vector_diffraction(system, r_coords, z_coords)
        
    return r_coords, Ex, Ey, Ez


def create_coordinate_grids(
    system,  # OpticalSystem
    r_range: float = None,
    z_range: float = None,
    nr: int = 51,
    nz: int = 51
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create coordinate grids for 2D field calculations.
    
    Parameters
    ----------
    system : OpticalSystem
        Optical system parameters
    r_range : float, optional
        Radial range (0 to r_range). Default: 4 × Airy radius
    z_range : float, optional
        Axial range (±z_range). Default: 4 × Rayleigh length
    nr : int
        Number of radial points
    nz : int
        Number of axial points
        
    Returns
    -------
    r_coords, z_coords : np.ndarray
        1D coordinate arrays
    R, Z : np.ndarray
        2D coordinate grids
    """
    if r_range is None:
        r_range = 4 * system.airy_radius
    if z_range is None:
        z_range = 4 * system.rayleigh_length
        
    r_coords = np.linspace(-r_range, r_range, nr)
    z_coords = np.linspace(-z_range, z_range, nz)
    R, Z = np.meshgrid(r_coords, z_coords)
    
    return r_coords, z_coords, R, Z


def calculate_radial_longitudinal_cross_section(
    system,  # OpticalSystem
    r_range: float = None,
    z_range: float = None,
    nr: int = 51,
    nz: int = 51,
    method: Literal['scalar', 'vector', 'gaussian'] = 'vector'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate radial-longitudinal cross-section field distribution.
    
    Returns
    -------
    r_coords, z_coords : np.ndarray
        1D coordinate arrays
    R, Z : np.ndarray
        2D coordinate grids
    Ex, Ey, Ez : np.ndarray (complex)
        2D electric field components
    """
    r_coords, z_coords, R, Z = create_coordinate_grids(
        system, r_range, z_range, nr, nz
    )
    
    if method == 'gaussian':
        Ex, Ey, Ez = scalar_diffraction_focused_field(system, R, Z)
    elif method == 'scalar':
        Ex, Ey, Ez = scalar_circular_aperture_diffraction(system, R, Z)
    else:  # vector
        Ex, Ey, Ez = richards_wolf_vector_diffraction(system, R, Z)
        
    return r_coords, z_coords, R, Z, Ex, Ey, Ez