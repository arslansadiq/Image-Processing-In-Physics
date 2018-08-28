"""
05.07.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex2_wavelet_transformation.py

Using numpy, matplotlib and scipy

This is a script which calculates the wavelet transform of an image,
thresholds the wavelet coefficients, performs the inverse wavelet transform,
in order to compress the image. The script plots the original image, the
wavelet tiling, the thresholded tiling, and the compressed image.

See IPPtools for details on the functions that have been added for this
exercise:
dwt_multiscale
idwt_multiscale
tile_dwt
rescale

Try to replace the missing information (there are 6 incomplete lines as
indicated by the ??? and hence generate a successfully compressed image.
"""
import numpy as np
import matplotlib.pyplot as plt
import IPPtools as IPPT

# Load in the image tree.jpg and choose the red channel
img = plt.imread('tree.jpg')[:, :, 0] / 255.

# Choose compression level for hard threshold
compression_level = 0.1  # Between 0 and 1

# Wavelet decomposition parameters
# Choose your favourite wavelet type. See the pywt wavelet object homepage for
# a list http://www.pybytes.com/pywavelets/regression/wavelet.html
# Choose the number of levels of decomposition

nLevel = 3        # Number of decompositions
wavelet = 'haar'  # mother wavelet
mode = 'per'      # zero padding mode

# Decomposition with IPPT.dwt_multiscale
coeffs, (A, H, V, D) = IPPT.dwt_multiscale(
    img, nLevel=nLevel, mode=mode, wavelet=wavelet)

# Extract the approximation image of last decomposition level
A0 = coeffs[-1][0]

# Group all coefficients to search for the right threshold
allcoeffs = np.hstack([A0.ravel(), H, V, D])**2

# Number of coefficients that have to be set to zeros
Nzeros = int((1 - compression_level) * len(allcoeffs))

# Sort coefficients by size, give back a sorted list of indices
iarg = allcoeffs.argsort()

# Find lowest allowed power
lowest_power = allcoeffs[iarg[Nzeros]]

# Threshold the coefficients
newcoeffs = [
    [iCoeffs*(iCoeffs**2 >= lowest_power) for iCoeffs in iLevels]
    for iLevels in coeffs
]

# reconstruct new coefficients
rec = IPPT.idwt_multiscale(newcoeffs, mode=mode, wavelet=wavelet)

# Total power before thresholding
power0 = allcoeffs.sum()

# Total power after thresholding
power1 = allcoeffs[iarg[Nzeros:]].sum()

print(
    'compression by %3.1f%% leads to %3.1f%% relative error' %
    (100-100*compression_level, 100*(1-power1/power0))
)

plt.figure(1, figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(2, 2, 2)
plt.imshow(rec, cmap='gray')
plt.title('Compression by %3.1f%%' % (100 - 100 * compression_level))
plt.subplot(2, 2, 3)
plt.imshow(IPPT.tile_dwt(coeffs, img.shape)**(1 / 4.), cmap='gray')
plt.title('Wavelet decomposition (gamma 0.25)')
plt.subplot(2, 2, 4)
plt.imshow(IPPT.tile_dwt(newcoeffs, img.shape)**(1 / 4.), cmap='gray')
plt.title('Wavelet thresholded (gamma 0.25)')
plt.show()
