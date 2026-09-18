'''
Analyze.py serves as the math analyzer of an oscilloscope that loads an
existing signal from storage and performs analysis functions like FFT
on the signal.

Author: Brad Klingerman (bdklingerman)
Version: September 18, 2026
'''

import argparse
import numpy
import pandas

from . import visualize

# Default parameters
adc_bits = 12

def fourier_fs(signal:numpy.ndarray, size: int, bits: int) -> numpy.ndarray:
    '''Calculate the discrete fourier transform and scale to dBFS'''

    # Get a raw FFT up to M/2
    spectrum_raw = numpy.fft.fft(signal, n=size)[:int(size/2)]

    # dBFS sets 0 dB at the maximum swing, use that to scale
    dbfs_ref = (2**bits / 2) * (size / 2)

    # Return the spectrum in dBFS
    return 20 * numpy.log10(numpy.abs(spectrum_raw) / dbfs_ref)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="raw_waveform")
    args = parser.parse_args()

    # Load two columns from CSV
    frame = pandas.read_csv(f"data/{args.name}.csv")
    x = frame.iloc[:, 0].to_numpy()
    y = frame.iloc[:, -1].to_numpy()

    # Extract parameters
    fft_size = x.size
    fs = 1 / (x[1] - x[0])

    # Generate the spectrum in dBFS
    f = numpy.arange(fft_size/2) * (fs/fft_size) # fs/fft_size = resolution bandwidth
    spectrum_dbfs = fourier_fs(y, fft_size, adc_bits)
    
    # Plot the spectrum
    fft_fig, fft_ax = visualize.save_plot(spectrum_dbfs,f/(10**6),
              xlabel="Frequency (MHz)",ylabel="Amplitude (dBFS)",filename="fft")
    
    # Calculate SNR and noise floor
    snr_db = 6.02 * adc_bits + 1.76
    process_gain_db = 10 * numpy.log10(fft_size / 2)
    noise_floor_db = snr_db + process_gain_db
    
    # Annotate spectrum
    fft_ax.axhline(-snr_db,color="grey",label=f"SNR = {snr_db:.0f} dB")
    fft_ax.axhline(-noise_floor_db,color="black",label=f"Noise Floor = {noise_floor_db:.0f} dB")
    fft_ax.legend()
    fft_fig.savefig("data/fft.png")

if __name__ == "__main__":
    main()