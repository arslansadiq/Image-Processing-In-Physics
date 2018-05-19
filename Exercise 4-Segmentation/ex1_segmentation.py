"""
17.05.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex1_segmentation.py

This exercise is all about counting stars.
The goal is to know how many stars are in the image and what sizes they are.

As per usual, replace the ???s with the appropriate command(s).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# Load the respective image

img = plt.imread('stars.jpg')

# Sum up all color channels to get a grayscale image.
# use numpy function sum and sum along axis 2, be careful with the datatypes
# rescale the finale image to [0.0, 1.0]

img = img.sum(axis = 2)/255.0

# Now look at your image using imshow. Use vmin and vmax parameters in imshow

plt.figure(1)
plt.title('img')
plt.imshow(img, cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
plt.colorbar()

# You can set thresholds to cut the background noise
# Once you are sure you have all stars included use a binary threshold.
# (Tip: a threshold of 0.1 seemed to be good, but pick your own)

threshold = 0.15
img_bin = img > threshold

plt.figure(2)
plt.title('img_bin')
plt.imshow(img_bin, cmap='gray', interpolation='none')

# Now with the binary image use the opening and closing to bring the star
# to compacter format. Take care that no star connects to another

s1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
img_bin1 = nd.binary_closing(img_bin, structure=s1)

plt.figure(3)
plt.title('img_bin1')
plt.imshow(img_bin1, cmap='gray', interpolation='none')

# Remove isolated pixels around the moon with closing by a 2 pixel structure

s2 = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
img_bin2 = nd.binary_opening(img_bin1, structure=s2)

plt.figure(4)
plt.title('img_bin2')
plt.imshow(img_bin2, cmap='gray', interpolation='none')

# play with all the morphological options in ndimage package to increase the
# quality if still needed

#s3 = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
#img_bin3 = nd.binary_dilation(img_bin2, structure=s3)  # optional
img_bin3 = img_bin2
plt.figure(5)
plt.title('img_bin3')
plt.imshow(img_bin3, cmap='gray', interpolation='none')

# plotting the sum of all your binary images can help identify if you loose
# stars. In principal every star is present in every binary image, so real
# stars have always at least one pixel maximum

plt.figure(6)
plt.imshow(img_bin.astype(int) + img_bin1.astype(int) + img_bin2.astype(int) +
           img_bin3.astype(int), cmap='jet', interpolation='none')
plt.colorbar()

# Once you're done, label your image with nd.label

img_lbld, num_stars = nd.label(img_bin3)

plt.figure(7)
plt.imshow(img_lbld, cmap='jet', interpolation='none')
plt.colorbar()

# Use nd.find_objects to return a list of slices through the image for each
# star

slice_list = nd.find_objects(img_lbld)

# You can have a look now at the individual stars. Just apply the slice to your
# labelled array

starnum = 150

plt.figure(8)
plt.title("star %i" % starnum)
plt.imshow(img_lbld[slice_list[starnum-1]], cmap='gray', interpolation='none')

# Remaining task: Sum up each individual star to get a list of star sizes and
# make a detailed histogram (>100 bins). Take care to exclude the moon! This
# can be done by sorting the star sizes list and removing the last element

# Remember: img_lbld[slice_list[<number>]] selects one star. Create a list of
# boolian star images (star_list). Afterwards, sum their extent up (take
# care about the datatypes) to get their sizes and sort the list

star_list = [img_lbld[slc] > 0 for slc in slice_list]
mass_list = [np.sum(star) for star in star_list]
mass_list_sorted = np.sort(mass_list)
mass_list_sorted = mass_list_sorted[0:len (mass_list_sorted) - 2]

plt.figure(9)
plt.title("sizes of stars")
plt.hist(mass_list_sorted, bins = 200, range = (0, 200), align = 'left')
