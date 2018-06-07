"""
07.06.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex1_paganin_phase_retrieval.py

Script recover the projected thickness of a teflon plate quantitatively from
its intensity measurement in the near-field.

As usual replace the ???s with the appropriate command(s).
"""

import numpy as np
import matplotlib.pyplot as plt

# Load projection data from file. It is a 250 micro thick Teflon plate. Note
# that this data is already flatfield-corrected!

proj = np.load('proj.npy')

# Look at the data. You can see the edge-enhanced borders at the transition
# from Teflon to air. In addition, the absorbing properties of the Teflon plate
# are visible. Note that the background-values are around 1.

plt.figure()
plt.title('intensity')
plt.imshow(proj, cmap='gray', interpolation='none')
plt.colorbar()

# Also plot a line profile through the middle row.

plt.figure()
plt.plot(proj[:][proj.shape[0]//2])

# The parameters of the setup that influence the image formation process are
# specified below.

pixel_size = .964e-6
distance = 8.57e-3

# As Paganin assumes a single material which has to be known beforehand, we look
# up the absorption index and the decrement of the real part of the complex
# refractive index in some database for the given energy. I do that for you.

mu = 691.
delta = 2.6e-6

# I help you with creating the frequencies that correspond to the different
# parts of the Fourier image according to our convention.

v = 2. * np.pi * np.fft.fftfreq(proj.shape[0], d=pixel_size)
u = 2. * np.pi * np.fft.fftfreq(proj.shape[1], d=pixel_size)
ky, kx = np.meshgrid(v, u, indexing='ij')

# Build the Paganin kernel. Its representation was discussed in the lecture.

Paganin = 1 / ((distance*(delta/mu))*(ky**2 + kx**2) + 1)

# Recover the thickness from the projection by applying the Paganin kernel onto
# the intensity measurement.

trace = (-1/mu)*(np.log(np.fft.ifft2(Paganin * np.fft.fft2(proj))))

# Plot the recovered thickness of the sample in microns. Also plot a line
# through the center row of the trace. Check if the retrieved thickness matches
# the stated thickness in the beginning of our exercise.

plt.figure()
plt.title('trace')
plt.imshow(trace.real*1e6, cmap='gray', interpolation='none')
plt.colorbar()

plt.figure()
plt.plot(trace[:][trace.shape[0]//2]*1e6)
