"""
24.05.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex1_fresnel_propagation.py

Using numpy, matplotlib, scipy

Script to perform Fresnel (near-field) wavefront propagation.
Check your figures against the lecture notes.
You need to replace the ??? in the code with the required commands.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters

psize = 1e-5  # Detector pixelsize
wlen = 6e-7  # Wavelength (600nm = visible light)
prop_dist = 3e-3  # Propagation distance

# Read in test wavefield from image

img = plt.imread('tum.png')

# Sum up all channels

img = np.sum(img, axis=-1, dtype=float)

# Scale such that max value is 1

img = img / img.max()

# Generate a pure phase wavefield spanning from zero to np.pi
# from img

w = np.exp(1j * np.pi * img) #it's a pure phase wavefield so amplitude is 1

plt.figure(1)
plt.imshow(np.angle(w), cmap='jet', interpolation='none')
plt.title('Wavefront phase')
plt.colorbar()

# Generate the grids

u = 2. * np.pi * np.fft.fftfreq(img.shape[1], psize)
v = 2. * np.pi * np.fft.fftfreq(img.shape[0], psize)

uu, vv = np.meshgrid(u, v, indexing='xy')

# Generate wave number

k = 2 * np.pi/wlen

# Generate the kernel

kernel = np.exp(-.5j * (prop_dist/k) * (uu**2 + vv**2))

# Generate the propagated wave array

out = np.fft.ifft2(np.fft.fft2(w) * kernel)

# Plot the phase of the kernel, maybe the function np.angle helps ;)

plt.figure(2)
plt.imshow(np.fft.fftshift(np.angle(kernel)), cmap = 'jet', interpolation='none')
plt.title('Fresnel kernel')
plt.colorbar()

# Calculate the intensity from the propagated wave array

I = np.abs(out)**2

# Plot the propagated intensity (with zoom to centre of image)

plt.figure(3)
plt.imshow(I[int(img.shape[0]/2-256):int(img.shape[0]/2+256),
             int(img.shape[1]/2-256):int(img.shape[1]/2+256)],
           cmap='gray', interpolation='none')
plt.title('Intensity')
plt.colorbar()
