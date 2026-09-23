'''
test_synthesize.py is a very basic testbench to test the CI workflow.

Author: Brad Klingerman (bdklingerman)
Version: September 23, 2026
'''

import numpy
import pytest

from adc_quantization.synthesize import synthesize_sine


def test_synthesize_sine_peaks():
    t = numpy.arange(4) / 4
    v = synthesize_sine(t, amplitude=1.0, freq=1.0, offset=1.0)
    assert v[1] == pytest.approx(2)
    assert v[3] == pytest.approx(0)