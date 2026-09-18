'''
Synthesize.py simulates the analog front-end of an oscilloscope by
generating new sampled signals, quantizing those signals, and saving
them to storage.

Author: Brad Klingerman (bdklingerman)
Version: September 18, 2026
'''

import argparse
import numpy
import pandas

from . import visualize

# Default parameters, matching MT003
freq = 2.11206 * 10**6
fs = 82 * 10**6
adc_bits = 12
adc_span = 5
fft_size = 8192

def synthesize_signal(
        amplitude: float,
        freq: float,
        offset: float,
        fs: float,
        length: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
    '''Generates t and v arrays based on input parameters.'''

    # Handle exceptions
    if amplitude <= 0:
        raise ValueError(f"amplitude {amplitude} must be positive")
    if freq <= 0:
        raise ValueError(f"frequency {freq} must be positive")
    if fs <= 0:
        raise ValueError(f"sampling frequency {fs} must be positive")
    if fs < 2 * freq:
        raise ValueError(f"sampling frequency {fs} must be above Nyquist {2 * freq}")
    if length <= 0:
        raise ValueError(f"record length {length} must be positive")

    # Generate sine wave axes
    t = numpy.arange(length) / fs
    v = amplitude * numpy.sin(2 * numpy.pi * freq * t) + offset

    # Return as {t, v}
    return t, v

def quantize_signal(signal: numpy.ndarray, bits: int, vref: float) -> numpy.ndarray:
    '''Simulate ADC quantization on a signal.'''

    # Define the ADC bins
    lsb = vref / (2**bits)
    adc = numpy.arange(1, 2**bits) * lsb

    # Return the binned values
    return numpy.digitize(signal, adc)

def main():
    '''Parses input from CLI and saves it as csv and plot.'''

    # Parse arguments for waveform
    parser = argparse.ArgumentParser()
    # TODO Add square wave
    parser.add_argument("-n", "--name", type=str, default="raw_waveform")
    parser.add_argument("-f", "--frequency", type=float, default=freq)
    parser.add_argument("-a", "--amplitude", type =float, default=adc_span/2)
    parser.add_argument("-o", "--offset", type=float, default=adc_span/2)
    parser.add_argument("-s", "--sampling", type=float, default=fs)
    parser.add_argument("-b", "--bits", type=int, default=adc_bits)
    parser.add_argument("-m", "--length", type=int, default=fft_size)
    parser.add_argument("-r", "--reference", type=float, default=adc_span)
    args = parser.parse_args()

    # Generate the waveform
    t, v = synthesize_signal(amplitude=args.amplitude, freq=args.frequency, offset=args.offset, fs=args.sampling, length=args.length)

    # Plot the raw data
    visualize.save_plot(y=v,x=t,filename="raw_waveform")

    # Quantize the signal
    quantized_codes = quantize_signal(signal=v, bits=args.bits, vref=args.reference)

    # Plot the quantized data
    visualize.save_plot(y=quantized_codes,x=t,filename="adc_waveform")

    # Save the data
    frame = pandas.DataFrame({"Time (s)": t, "Voltage (v)": v, "Codes": quantized_codes})
    frame.to_csv(f"data/{args.name}.csv", index=False)

if __name__ == "__main__":
    main()