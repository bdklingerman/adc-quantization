'''
test_analyze.py is a very basic testbench to test the CI workflow.

Author: Brad Klingerman (bdklingerman)
Version: September 23, 2026
'''

import numpy
import pytest

from adc_quantization.analyze import fourier_fs

SIZE = 1024
BITS = 12
BIN = 37  # whole number of cycles per record avoids leakage

def coherent_sine(amplitude: float) -> numpy.ndarray:
    n = numpy.arange(SIZE)
    return amplitude * numpy.sin(2 * numpy.pi * BIN * n / SIZE)

def test_fourier_fs_length():
    spectrum = fourier_fs(coherent_sine(1.0), SIZE, BITS)
    assert spectrum.size == SIZE // 2

def test_fourier_fs_full_scale_is_0_dbfs():
    spectrum = fourier_fs(coherent_sine(2**BITS / 2), SIZE, BITS)
    assert numpy.argmax(spectrum) == BIN
    assert spectrum[BIN] == pytest.approx(0, abs=1e-9)

def test_fourier_fs_half_scale_is_minus_6_db():
    spectrum = fourier_fs(coherent_sine(2**BITS / 4), SIZE, BITS)
    assert spectrum[BIN] == pytest.approx(20 * numpy.log10(0.5))
