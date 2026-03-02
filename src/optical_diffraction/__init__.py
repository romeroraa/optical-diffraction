"""
Optical Diffraction Package

A package for analyzing focused optical beams using Richards-Wolf
vector diffraction theory and scalar diffraction theories.

Research progression:
1. Richards-Wolf vector diffraction — CW and pulsed beams (validated)
2. Gaussian input beams — Tanaka et al. analytical theory
3. Annular apertures — Babinet's principle with Gaussian inputs
4. Combined analysis — (in progress)
5. Monte Carlo photon transport — (roadmap)
"""

from ._version import __version__
from .constants import LIGHT_SPEED
from .systems import OpticalSystem

# Diffraction methods
from .diffraction import (
    # Main diffraction functions
    scalar_diffraction_focused_field,
    scalar_circular_aperture_diffraction,
    richards_wolf_vector_diffraction, 
    # Profile creation functions
    create_axial_profile,
    create_transverse_profile,
    create_coordinate_grids,
    calculate_radial_longitudinal_cross_section,
    # Helper functions
    convert_to_optical_coordinates,
    calculate_integral_I0,
    calculate_integral_I1,
    calculate_integral_I2,
    calculate_field_Ex,
    calculate_field_Ey,
    calculate_field_Ez
)

# Pulsed beam diffraction
from .pulsed_diffraction import (
    # Core pulsed beam class
    PulsedBeam,
    # Main pulsed diffraction functions
    pulsed_richards_wolf_diffraction,
    create_pulsed_axial_profile,
    create_pulsed_transverse_profile
)

# OOP Richards-Wolf simulator (supports uniform/gaussian inputs, annular via Babinet)
from .richards_wolf import RichardsWolfSimulator

# Gaussian beam analytical theory (Tanaka et al. 1985, Horvath & Bor 2003)
from .gaussian_beam_theory import FocusedGaussianBeamTheory

# Annular aperture (Babinet's principle)
from .annular import annular_field, annular_intensity

# Beam quality and comparison metrics (Linfoot's criteria)
from . import metrics

__all__ = [
    # Package info
    "__version__",
    "LIGHT_SPEED", 
    
    # Core classes
    "OpticalSystem",
    "PulsedBeam",
    
    # Main diffraction functions
    "scalar_diffraction_focused_field",
    "scalar_circular_aperture_diffraction",
    "richards_wolf_vector_diffraction",
    
    # Profile creation functions
    "create_axial_profile",
    "create_transverse_profile", 
    "create_coordinate_grids",
    "calculate_radial_longitudinal_cross_section",
    
    # Helper functions
    "convert_to_optical_coordinates",
    "calculate_integral_I0",
    "calculate_integral_I1", 
    "calculate_integral_I2",
    "calculate_field_Ex",
    "calculate_field_Ey",
    "calculate_field_Ez",
    
    # Pulsed beam functions
    "pulsed_richards_wolf_diffraction",
    "create_pulsed_axial_profile",
    "create_pulsed_transverse_profile",

    # OOP RW simulator (uniform/gaussian inputs)
    "RichardsWolfSimulator",

    # Gaussian beam theory (Tanaka et al.)
    "FocusedGaussianBeamTheory",

    # Annular aperture (Babinet's principle)
    "annular_field",
    "annular_intensity",

    # Metrics (Linfoot's criteria, RMSE)
    "metrics",
]