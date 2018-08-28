"""
28.06.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex1_interferometry.py

Grating interferometry
In this exercise we will process and analyze a dataset from grating
interferometry.

As per usual, replace the ???s with the appropriate command(s).
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def wrap_phase(inarray):
    """
    This function just makes sure that differential phase values
    stay in the range -pi,pi.
    """
    outarray = np.angle(np.exp(1j * inarray))
    return outarray

# path to the raw data, os.sep is '/' on unix and '\\' on windows

PATH = 'data' + os.sep

# the number of stepping images

NUMIMGS = 11

# Format string for filenames, %s stands for string, %04d stands for
# a 4-digit integer with leading zeros.
# (you can use for example FILEPATTERN % ('data', 5) to get
# the string 'data_stepping_0005.npy')

FILEPATTERN = '%s_stepping_%04d.npy'

# Read in raw data
# there are two series of 11 .npy files:
# the stepping images with sample in the beam
# (data_stepping_0000.npy to data_stepping_0010.npy)
# and without sample, i.e. the flatfields
# (flat_stepping_0000.npy to flat_stepping_0010.npy)
# You should read them in one after the other and combine them into
# two numpy arrays with shape (11, 195, 487), the first dimension
# represents the stepping images, the other two the actual dimensions
# of each image.
# Use np.load to read the data.

imglist = []
flatlist = []
for i in range(NUMIMGS):
    # load the image
    img = np.load(os.path.join(os.getcwd(), PATH, FILEPATTERN % ('data', i)))
    imglist.append(img)
    # load the flatfield
    flat = np.load(os.path.join(os.getcwd(), PATH, FILEPATTERN % ('flat', i)))
    flatlist.append(flat)

imgarr = np.array(imglist)
flatarr = np.array(flatlist)

# Plot stepping curve of pixel (50, 200) of the data array
# you should see a cosine curve

stepping_curve = imgarr[:, 50, 200]
ref_curve = flatarr[:, 50, 200]

plt.figure(1)
plt.plot(stepping_curve, '*', label='stepping curve')
plt.plot(ref_curve, 'bo', label='reference curve')

point_ft = np.fft.fft(stepping_curve)

cons = np.abs(point_ft[0]) / NUMIMGS
ang = np.angle(point_ft[1])
mod = np.abs(point_ft[1]) / NUMIMGS

x = np.linspace(0, NUMIMGS, 1000)
fit_stepping_curve = cons + 2. * mod * np.cos(x / NUMIMGS * 2 * np.pi + ang)

plt.plot(x, fit_stepping_curve, label='fit')
plt.legend()

# Cropping
# Have a look at one of the stepping images. You will see a rounded
# white border around the actual image. This is an area in the field
# of view, where there is no grating and thus no interference.
# You should crop all of the images, so that this outside area is not
# in the image anymore.

data_cropped = imgarr[:, :, 72:430]
flatfield_cropped = flatarr[:, :, 72:430]

# Fourier processing
# With the images cropped to their actual content, you will now do a
# Fourier processing to extract absorption, differential phase and
# darkfield signals.
# You will have to do a one-dimensional Fourier transform of both arrays
# along the stepping dimension (remember that there is a stepping curve
# for each pixel in the images) and normalize by the number of stepping
# images.
# Then extract the signals for the data and the flatfields:
# absorption: the absolute value of the zeroth (DC) term (equivalent to
# the mean value of the stepping curve)
# differential phase: the phase of the first order term (equivalent to
# the phase of the sine curve)
# darkfield: the absolute value of the first order term divided by
# absorpion (equivalent to half of the amplitude of the stepping curve)

data_fft = np.fft.fft(data_cropped, axis=0) / NUMIMGS
flat_fft = np.fft.fft(flatfield_cropped, axis=0) / NUMIMGS

data_absorption = np.abs(data_fft[0])
data_differential_phase = np.angle(data_fft[1])
data_darkfield = np.abs(data_fft[1]) / data_absorption

flatfield_absorption = np.abs(flat_fft[0])
flatfield_differential_phase = np.angle(flat_fft[1])
flatfield_darkfield = np.abs(flat_fft[1]) / flatfield_absorption

# now that you have the three signals for both stepping scans, you can
# do a flatfield correction
# for absorption, this is data divided by flatfield
# for differential phase: wrap_phase(data - flatfield)
# for darkfield you have to divide for the data by the
# values of the flatfield.
# for absorption and darkfield you also should then use the negative
# logarithm of the flatfield corrected images.

absorption = -np.log(data_absorption / flatfield_absorption)
differential_phase = wrap_phase(
    data_differential_phase - flatfield_differential_phase)
darkfield = -np.log(data_darkfield / flatfield_darkfield )

plt.figure(2)
plt.subplot(3, 1, 1)
plt.title('absorption')
plt.imshow(absorption, cmap='gray')
plt.subplot(3, 1, 2)
plt.title('differential phase')
plt.imshow(differential_phase, cmap='gray')
plt.subplot(3, 1, 3)
plt.title('darkfield')
plt.imshow(darkfield, cmap='gray')
