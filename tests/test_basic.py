"""Basic tests to verify package installation and imports."""

import pytest


def test_import():
    """Test that the package can be imported."""
    import optical_diffraction
    assert optical_diffraction is not None


def test_version():
    """Test that version is accessible."""
    import optical_diffraction
    assert hasattr(optical_diffraction, '__version__')
    assert isinstance(optical_diffraction.__version__, str)
    assert 'dev' in optical_diffraction.__version__ or '.' in optical_diffraction.__version__


def test_basic_functionality():
    """Placeholder for basic functionality test."""
    # For now, just test that numpy works
    import numpy as np
    arr = np.array([1, 2, 3])
    assert len(arr) == 3
