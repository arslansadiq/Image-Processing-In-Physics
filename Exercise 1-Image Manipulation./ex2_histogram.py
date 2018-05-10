"""
19.04.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex2_histogram.py

Using numpy, matplotlib and scipy
The goal of this exercise is for you to become familiar with the important
packages numpy, matplotlib and scipy.
Here you will load an image, add noise and look at the histograms of the
different color channels

You need to replace the ??? in the code with the required commands
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# use plt.imread to read in 'bears.jpg' and save the color channels
# into separate numpy arrays.
# Check the image dimensions before and after splitting the colors
# with with the shape attribute of img, red, green, and blue
# img should be a 3-dimensional array and the colors a 2d array, respectively.

img = plt.imread('bears.jpg') / 255.  # Division to norm to an interval [0, 1]
sh = img.shape

# Select red, green, and blue channel

red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]
print(sh, red.shape, green.shape, blue.shape)

# Display the original and the three color channels in an array of subplots.
# Therefore, open a figure with plt.figure() and use plt.subplot(...) to plot
# them in a 2x2 array. To use the function correctly, look up the help by
# typing plt.subplot? in the ipython console or help(plt.subplot) in python

plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('color')
plt.subplot(2, 2, 2)
plt.imshow(red, cmap='gray')
plt.title('red channel')
plt.subplot(2, 2, 3)
plt.imshow(green, cmap='gray')
plt.title('green channel')
plt.subplot(2, 2, 4)
plt.imshow(blue, cmap='gray')
plt.title('blue channel')

# Create the histograms of the three color channels separately
# using the np.histogram function. Use 50 bis and a range of (0, 1)
# Afterwards plot them into one histogram line plot. Keep in mind that
# np.histograms returns left and right bin margins. Therefore, you will need
# to create the central bin positions by yourself

red_hist = np.histogram(red, bins=50, range=(0,1))
green_hist = np.histogram(green, bins=50, range=(0,1))
blue_hist = np.histogram(blue, bins=50, range=(0,1))

# In case you do not know how to do the last part look at the lower parts of
# the script. The lines before will appear again in a similar fashion.

red_bins = red_hist[1]
central_bins = (red_bins[1:] + red_bins[:-1]) / 2.
'''green_bins = green_hist[1]
central_bins_g = (green_bins[1:] + green_bins[:-1]) / 2.
blue_bins = blue_hist[1]
central_bins_b = (blue_bins[1:] + blue_bins[:-1]) / 2.'''

plt.figure(2)
plt.title('histograms of 3 color channels')
plt.plot(central_bins, blue_hist[0], label='blue')
plt.plot(central_bins, green_hist[0], label='green')
plt.plot(central_bins, red_hist[0], label='red')
plt.grid()
plt.legend()

# Now, add Gaussian noise to the image with the function
# np.random.standard_normal with a standard deviation of 0.1

img_noisy = img + 0.1*np.random.standard_normal(sh)

# Note, that values below 0. and above 1. wrap around on the color scale
# Therefore, they have to be set back to 0. or 1. respectively
# Hint: The coordinates to index the array can also be a boolean array of the
# same shape. So, if you want to select all pixels with a value smaller
# than 0, you can use img_noisy < 0.

img_noisy[img_noisy < 0.] = 0.
img_noisy[img_noisy > 1.] = 1.

plt.figure(3)
plt.title('noisy image')
plt.imshow(img_noisy, cmap='gray', vmin=0, vmax=1.)

red_hist_noisy = np.histogram(img_noisy[..., 0], bins=50, range=(0, 1))
green_hist_noisy = np.histogram(img_noisy[..., 1], bins=50, range=(0, 1))
blue_hist_noisy = np.histogram(img_noisy[..., 2], bins=50, range=(0, 1))

plt.figure(4)
plt.title('histograms of 3 noisy color channels')
plt.plot(central_bins, blue_hist_noisy[0], label='blue')
plt.plot(central_bins, green_hist_noisy[0], label='green')
plt.plot(central_bins, red_hist_noisy[0], label='red')
plt.grid()
plt.legend()

# After adding noise, we want to remove it again by Gaussian filtering.
# Therefore, the function gaussian_filter of the nd.filter module can be used.
# Apply the filter with a filter kernel size of sigma=1.
# You can either filter each image band separately or give a list of sigmas
# (one for each dimension) and make sure that you do not filter across color
# channels with a zero at the right place.

# The 0 in the last axis mean that we do not filter
# across color channels of the image

sigma = (1, 1, 0)
img_filtered = nd.filters.gaussian_filter(img_noisy, sigma=sigma)

plt.figure(5)
plt.title('filtered image')
plt.imshow(img_filtered, cmap='gray', vmin=0, vmax=1.)

red_hist_filtered = np.histogram(img_filtered[..., 0], bins=50, range=(0, 1))
green_hist_filtered = np.histogram(img_filtered[..., 1], bins=50, range=(0, 1))
blue_hist_filtered = np.histogram(img_filtered[..., 2], bins=50, range=(0, 1))

plt.figure(6)
plt.title('histograms of 3 filtered color channels')
plt.plot(central_bins, blue_hist_filtered[0], label='blue')
plt.plot(central_bins, green_hist_filtered[0], label='green')
plt.plot(central_bins, red_hist_filtered[0], label='red')
plt.grid()
plt.legend()
