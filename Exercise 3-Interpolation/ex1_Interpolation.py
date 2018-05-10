"""
03.05.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex1_interpolation.py

Using numpy, matplotlib and scipy

The goal of this exercise is for you to try a simple image interpolation
using interpolation kernels. For this you should first downsample an image
and then try to increase its sampling again by using an appropriate
interpolation method.

You need to replace the ??? in the code with the required commands
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# read an image using matplotlib tools and normalize it to [0, 1]

img = plt.imread('tree.jpg')[:, :, 0] / 255.
sh = np.shape(img)

# Subsample the original image by a certain factor by keeping only every k-th
# Row and column. Hint: remember array indexing, a[i:j:k]
img_sub = np.zeros(sh)
factor = 5
img_sub = img[::factor, ::factor]

# OPTIONAL TASK: Rebin the old image via averaging.
# Therefore, patches of 4 times 4 pixels are averaged and written into the
# according position of the subarray. You can do this explicitely or
# by reshaping the 2d-array into a 4d array, where the two new axes contain
# the values that are to be averaged

img_sub = np.mean(np.reshape(img, (sh[0]//factor, factor, sh[1]//factor,
                                   factor)), axis=(1,3))

# prepare the subsampled image for interpolation by inserting zeros between all
# pixel values in img_sub. img_up should be the same size as the original (img)
# To fill the upscaled image with a sparse matrix, remember stepping in slicing

img_up = np.zeros(sh)
img_up[factor//2::factor, factor//2::factor] = img_sub 
# Define the interpolation kernel for nearest neighbour interpolation
# Hint: the pixels are separated by 5 distance units,
#       So how wide must the kernel be?

kernel_nearest = np.ones((factor, factor))

# Perform nearest-neighbor interpolate using either convolution (easier) or fft
# You can use nd.convolve for the convolution with mode='wrap'

img_nearest = nd.convolve(img_up, kernel_nearest, mode = 'wrap')

# define the interpolation kernel for linear interpolation
# Hint: the linear kernel can be obtained by a convolution
#       of two rectangular kernels centered in a larger kernel
#       Make sure, that the kernel is wide enough

kernel_rect = np.zeros((2*factor - factor % 2, 2*factor - factor % 2))
kernel_rect[factor//2:3*factor//2, factor//2:3*factor//2] = 1
kernel_linear = nd.convolve(kernel_rect, kernel_rect)
kernel_linear /= factor**2  # normalization

# Perform linear interpolation using either convolution (easier) or fft
# Check if the images are normalized correctly and have a look if the filtered
# and unfiltered images are correctly aligned

img_linear = nd.convolve(img_up, kernel_linear, mode='wrap')

# Perform sinc interpolation using the convolution theorem and fft
# Hint: the sinc kernel is easier to define in Fourier domain:
#       In Fourier domain, the sinc is given by a rectangular function.
#       Its width is given by the width of the subsampled image, sh/factor/2.

kernel_sinc = np.zeros(sh)
w = sh[0]//2//factor
kernel_sinc[sh[0]//2-w:sh[0]//2+w, sh[1]//2-w:sh[1]//2+w] = 1
kernel_sinc = np.fft.ifftshift(kernel_sinc)
img_sinc = np.real(np.fft.ifft2(np.fft.fft2(img_up) * kernel_sinc))

# Plot results

plt.figure(1)
plt.subplot(2, 3, 1)
plt.title('original')
plt.imshow(img, cmap='gray', interpolation='none')
plt.subplot(2, 3, 2)
plt.imshow(img_sub, cmap='gray', interpolation='none')
plt.title('downsampled')
plt.subplot(2, 3, 3)
plt.imshow(img_up, cmap='gray', interpolation='none')
plt.title('upsampled again')
plt.subplot(2, 3, 4)
plt.imshow(img_nearest, cmap='gray', interpolation='none')
plt.title('nearest interpolated')
plt.subplot(2, 3, 5)
plt.imshow(img_linear, cmap='gray', interpolation='none')
plt.title('linear interpolated')
plt.subplot(2, 3, 6)
plt.imshow(img_sinc, cmap='gray', interpolation='none')
plt.title('sinc interpolated')
