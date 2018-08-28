"""
05.07.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex1_windowed_FT.py

Using numpy, matplotlib

This is a script which calculates the spectrogram of a signal by calling
the function calc_spectrogram.

Try to replace the missing information (there are 6 incomplete lines as
indicated by the ??? and generate the spectrogram plot.
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_spectrogram(signal, width, sigma):
    """
    Calculate the spectrogram of a 1D-signal via the 1D Windowed Fourier
    Transform. Smoothen the rectangular window of width "width" with a
    Gaussian of width "sigma"

    Parameters are: - the signal
                    - the width of the rect window
                    - the width of the apodizing gaussian

    Returns: spectrogram
    """

    # append zeros to both sides of the signal, so that there is no
    # wrap-around when windowing
    signal = np.hstack([np.zeros(width//2), signal, np.zeros(width//2)])

    # create spectrogram data container
    spectrogram = np.zeros((len(signal), len(signal)))

    for position in np.arange(width//2, len(signal) - width//2):
        # extract rectangular region of width "width" from signal about
        # position
        windowed_signal = signal[position - width//2:position + width//2]

        # apodize edges of rect-window by gaussian
        gaussian = np.exp(
            -((np.linspace(0, width, width) - width//2)**2) / (2. * sigma**2))
        windowed_signal = windowed_signal * gaussian

        # zero-pad resulting windowed signal
        padded_window = np.zeros((len(signal)))

        padded_window[0:width] = windowed_signal

        # fourier transform the padded windowed signal
        # hint you need an fft shift here
        local_spectrum = np.fft.fftshift(np.fft.fft(padded_window))

        # calculate the spectrogram from the WFT
        spectrogram[:, position] = np.abs(local_spectrum)**2

    return spectrogram[:, width//2:len(signal) - width//2]

# Generate a vector x of 1000 points between 0 and 1
x = np.linspace(0, 1., 1000)
x2 = np.linspace(0, 4., 4000)
p1 = 1. / 80
p2 = 1. / 160
signal1 = np.cos(2 * np.pi / p1 * x)
signal2 = np.cos(2 * np.pi / p2 * x)
signal3 = (signal1 + signal2) / 2
signal4 = np.cos(2 * np.pi / p1 * x**2)

signal = np.hstack([signal1, signal2, signal3, signal4])

# Call the spectrogram function choosing an appropriate width:
width = 50
sigma = width / 5.
spec = calc_spectrogram(signal, width, sigma)

# Truncate the spectrogram to just take the positive frequencies:
spec = spec[0:spec.shape[0]//2, :]

# Plot the original signal
plt.figure(1, figsize=(14, 4))
plt.plot(x2, signal)

# Plot the spectrogram
plt.figure(2, figsize=(14, 4))
plt.imshow(spec, aspect='auto', extent=(0, 1, 0, 500), cmap='gray')
plt.title('spectrogram')
plt.ylabel('frequency')
plt.xlabel('time/space')
plt.show()
