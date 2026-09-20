'''
Synthesize.py simulates the analog front-end of an oscilloscope by
generating new sampled signals, quantizing those signals, and saving
them to storage.

Author: Brad Klingerman (bdklingerman)
Version: September 18, 2026
'''

import argparse
import json
from scipy import signal
import numpy
import pandas

from . import visualize

# Default parameters, matching MT003
freq = 2.11206 * 10**6
fs = 82 * 10**6
adc_bits = 12
adc_span = 5
fft_size = 8192

def synthesize_sine(
        samples: numpy.ndarray,
        amplitude: float,
        freq: float,
        offset: float,
    ) -> numpy.ndarray:
    '''Generates v array based on t and input parameters.'''

    # Handle exceptions
    if amplitude <= 0:
        raise ValueError(f"amplitude {amplitude} must be positive")
    if freq <= 0:
        raise ValueError(f"frequency {freq} must be positive")

    # Generate sine wave axes
    return amplitude * numpy.sin(2 * numpy.pi * freq * samples) + offset

def synthesize_square(
        samples: numpy.ndarray,
        amplitude: float,
        freq: float,
        offset: float,
        duty: float
    ) -> numpy.ndarray:
    '''Generates v array based on t and input parameters.'''

    # Handle exceptions
    if amplitude <= 0:
        raise ValueError(f"amplitude {amplitude} must be positive")
    if freq <= 0:
        raise ValueError(f"frequency {freq} must be positive")
    if duty < 0:
        raise ValueError(f"duty cycle {duty} must be 0 or greater")
    if duty > 1:
        raise ValueError(f"duty cycle {duty} must be 1 or less")

    # Generate square wave axis
    return amplitude * signal.square(2 * numpy.pi * freq * samples, duty=duty) + offset

def quantize_signal(signal: numpy.ndarray, bits: int, vref: float) -> numpy.ndarray:
    '''Simulate ADC quantization on a signal.'''

    # Define the ADC bins
    lsb = vref / (2**bits)
    adc = numpy.arange(1, 2**bits) * lsb

    # Return the binned values
    return numpy.digitize(signal, adc)

def main():
    '''Defines signal from JSON and saves it as csv and plot.'''

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parameters", type=str, default="demo_parameters")
    args = parser.parse_args()

    # Load parameters from JSON
    with open(f"data/{args.parameters}.json") as file:
        params = json.load(file)

    # Define timebase
    t = numpy.arange(params["length"]) / params["fs"]

    # Generate the waveform
    if (params["waveform"] == "sine"):
        v = synthesize_sine(samples=t, amplitude=params["amplitude"], freq=params["frequency"], offset=params["offset"])
    elif (params["waveform"] == "square"):
        v = synthesize_sine(samples=t, amplitude=params["amplitude"], freq=params["frequency"], offset=params["offset"], duty=0.5)
    else:
        raise ValueError(f"invalid waveform {params["waveform"]}")

    # Plot the raw data
    visualize.save_plot(y=v,x=t,filename=params["name"])

    # Quantize the signal
    quantized_codes = quantize_signal(signal=v, bits=params["bits"], vref=params["vref"])

    # Plot the quantized data
    visualize.save_plot(y=quantized_codes,x=t,filename="adc_waveform")

    # Save the data
    frame = pandas.DataFrame({"Time (s)": t, "Voltage (v)": v, "Codes": quantized_codes})
    frame.to_csv(f"data/{params["name"]}.csv", index=False)

if __name__ == "__main__":
    main()