"""
26.04.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex2_laplace_filtering.py

Laplace filter in frequency domain

Your task in this exercise is to create your own implementation of a
Laplace filter in Fourier space and apply it to an image.
The formula for the Laplacian in the Fourier domain is:
    H(u,v) = -4*pi^2*(u^2+v^2)  # source: (Gonzalez, chapter 4, p286)

You need to replace the ??? in the code with the required commands
"""

import numpy as np
import matplotlib.pyplot as plt
venice = plt.imread('venice.jpg')/255
# Load venice.jpg using imread, normalize it to (0, 1)
# and take the red channel again
img = venice[:,:,0]

# Plot the image before applying the filter
plt.figure(1)
plt.imshow(img, cmap='gray')
plt.colorbar()

# Generate a coordinate systems with the discrete Fourier transform sample
# frequencies v and u. You can use the numpy function linspace to do it
# manually or fftfreq. Look up the documentation to get familiar with the
# parameters of these functions.
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])

# the function np.meshgrid creates coordinate arrays for the v and the u
# coordinates and writes them into vv and uu
# you can display them with plt.figure(); plt.imshow(uu); colorbar() if you
# want to have a look at them
vv, uu = np.meshgrid(v, u, indexing='ij')

# Caluclate the filter function H(v, u)
# If you want to do this in one line use vv and uu, as they are both of the
# image shape. The formula is given in the very top documentation of this
# script. Check if H has the same shape as the image.
H = -4*(np.pi)**2*(uu**2 + vv**2)

# Calculate the Fourier transform of the image
# You can use the numpy function fft2 included in np.fft
img_ft = np.fft.fft2(img)

# Multiply the Fourier transform of the image by the filter function
# Take care (if neccessary) to center the potential function H around the top
# left corner of the image, because a Fourier transform in python always has
# the central frequencies in the top left corner. Therefore, play with the
# function fftshift and ifftshift to see what it does. Check out the looks of
# the shifted and unshifted potential function H.

# Take the inverse Fourier transform of the product to get the filtered image
# and select the real part of it, as we do not want to have the imaginary part
# of real images.
img_filtered = np.real(np.fft.ifft2(img_ft*H))

plt.figure(2)
plt.imshow(img_filtered, cmap='gray')
plt.colorbar()
