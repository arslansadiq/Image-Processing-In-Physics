"""
21.06.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex1_tomography.py

This exercise will be about a very simplified implementation of tomographic
reconstruction, using filtered backprojection.

The exercise consists of three parts:
First, you will simulate the data aquisistion in computed tomography, by
calculating the sinogram from a given input sample slice.
Second, you will have to apply a ramp filter to this sinogram.
Third, you will implement a simple backprojection algorithm.

If you do not manage to do one part of the exercise you can still go one by
loading the provided .npy arrays 'backup_sinogram.npy' and
'backup_filtered_sinogram.npy'.

You need to replace the ??? in the code with the required commands.
"""
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import sys


def roundmask(shape, radius=1):
    """
    This function creates a ellipsoid mask given a certain shape and radius.
    Give shape as a tuple indicating each image axis. Radius=1 means the mask
    will exactly touch each image axis, with radius=0.5 the mask will fill half
    of the image.
    """
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    xx, yy = np.meshgrid(x, y)
    return xx**2 + yy**2 < radius**2


def forwardproject(sample, angles):
    """
    Simulate data aquisition in tomography from line projections.
    Forwardproject a given input sample slice to obtain a simulated sinogram.

    Hints
    -----
    Use scipy.ndimage.rotate(..., reshape=False) to simulate the sample
    rotation.
    Use numpy.sum() along one axis to simulate the line projection integral.
    """
    sh = np.shape(sample)                # calculate shape of sample
    Nproj = len(angles)                  # calculate number of projections

    # define empty sinogram container, angles along y-axis
    sinogram = np.zeros((Nproj, sh[1]))

    for proj in np.arange(Nproj):  # loop over all projections
        sys.stdout.write("\r Simulating:     %03i/%i" % (proj+1, Nproj))
        sys.stdout.flush()
        im_rot = nd.rotate(sample, angles[proj], reshape=False)
        sinogram[proj,:] = np.sum(im_rot, axis=0)
    return sinogram


def filter_sino(sinogram):
    """
    Filter a given sinogram using a ramp filter

    Hints:
    First define a ramp filter in Fourier domain (you can use np.fft.fftfreq).
    Filter the sinogram in Fourier space unsing the convolution theorem.
    """

    Nproj, Npix = np.shape(sinogram)

    # Generate basic ramp filter (hint: there is the function np.fft.fftfreq.
    # Try it and see what it does. Watch out for a possible fftshift)
    ramp_filter = np.abs(np.fft.fftfreq(Npix))

    # filter the sinogram in Fourier space in detector pixel direction
    # Use the np.fft.fft along the axis=1
    sino_ft = np.fft.fft(sinogram, axis=1)

    # Multiply the ramp filter onto the 1D-FT of the sinogram and transform it
    # back into spatial domain
    sino_filtered = np.real(np.fft.ifft(sino_ft * ramp_filter, axis=1))

    return sino_filtered


def backproject(sinogram, angles):
    """
    Backproject a given sinogram.
    Hints:
    Perform the backprojection inversely to the way we did the
    forwardprojection, by smearing each projection in the sinogram back along
    the axis that you summed before in forwardproject() (you can use for
    example numpy.tile() for this), then rotating the resulting backprojection
    to get the right backprojection angle.
    Use scipy.ndimage.rotate(...,...,reshape=False)
    Using roundmask helps to improve the result.
    """
    # calculate number of projections, and pixels
    Nproj, Npix = np.shape(sinogram)
    # define empty container for reconstruction of sample
    reconstruction = np.zeros((Npix, Npix))

    for proj in np.arange(Nproj):  # loop over all projections
        sys.stdout.write("\r Reconstructing: %03i/%i" % (proj+1, Nproj))
        sys.stdout.flush()

        backprojection = np.tile(sinogram[proj, :], (Npix, 1))
        backprojection /= Npix  # Just normalization
        rotated_backprojection = nd.rotate(backprojection, -angles[proj], reshape=False)

        # Add the rotated backprojection multiplied with a roundmask
        reconstruction += rotated_backprojection * roundmask((Npix, Npix))

    return reconstruction


# read in sample data (in reality, this data is unknown and what you are
# looking for)
sample = plt.imread('Head_CT_scan.jpg')

# define vector containing the projection angles
Nangles = 301
angles = np.linspace(0, 360, Nangles, False) 

# simulate the process of tomographic data acquisition by line projections
sino = forwardproject(sample, angles)

# use this line if you do not manage the last step
# sino = np.load('backup_sinogram.npy')

# filter the sinogram with the ramp filter (or some other filter)
filtered_sino = filter_sino(sino)

# use this line if you do not manage the last step
# filtered_sino = np.load('backup_filtered_sinogram.npy')

# reconstruct the image from its filtered sinogram
reco = backproject(filtered_sino, angles)

plt.figure(1, figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0., cmap='gray', interpolation='none')


# Image Artifacts
# ---------------

# Artifact 1 - Hot / Dead Pixel
# -----------------------------
Nangles = 301
angles = np.linspace(0, 360, Nangles, False)

sino = forwardproject(sample, angles)

# simulate a dead pixel in the detector line
sino[:, 120] = 0

# filter the sinogram with the ramp filter and reconstruct it
filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(2, figsize=(12, 12))
plt.suptitle('dead pixel')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=sample.min(), vmax=sample.max(),
           cmap='gray', interpolation='none')


# Artifact 2 - Simulate a center shift
# ------------------------------------
# Intrinsically, tomography assumes that the rotation axis is in the center of
# each projection. If this is not the case, each projection is shifted left or
# right with respect to the optical axis. These are called center shift.

Nangles = 301
angles = np.linspace(0, 360, Nangles, False)

sino = forwardproject(sample, angles)

# shift the sinogram by a few pixels (~2) or pad the detector either to the
# left or right side.
np.append(sino, np.ones((Nangles, 10)), axis=1)

# filter the sinogram with the ramp filter and reconstruct it
filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(3, figsize=(12, 12))
plt.suptitle('center shift')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0, cmap='gray', interpolation='none')


# Artifact 3 - few angles / undersampling
# ---------------------------------------
Nangles = 91
angles = np.linspace(0, 360, Nangles, False)

sino = forwardproject(sample, angles)

# filter the sinogram with the ramp filter and reconstruct it
filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(4, figsize=(12, 12))
plt.suptitle('undersampling')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0., cmap='gray', interpolation='none')


# Artifact 4 - missing projections to tomosynthese
# ------------------------------------------------
Nangles = 301
angles = np.linspace(0, 180, Nangles, False)

sino = forwardproject(sample, angles)

# simulate one or more missing projections (e.g. replace with zeros) up to a
# missing projection wedge
sino[:100] = 0

# filter the sinogram with the ramp filter and reconstruct it
filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(5, figsize=(12, 12))
plt.suptitle('missing projections')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0., cmap='gray', interpolation='none')


# Artifact 5 - Noise
# ------------------
Nangles = 301
angles = np.linspace(0, 360, Nangles, False)

sino = forwardproject(sample, angles)

# simulate noise
sino += 5000 * np.random.standard_normal(sino.shape)

# filter the sinogram with the ramp filter and reconstruct it
filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(6, figsize=(12, 12))
plt.suptitle('noise')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0, cmap='gray', interpolation='none')

