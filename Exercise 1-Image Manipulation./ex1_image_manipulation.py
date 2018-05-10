"""
19.04.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex1_image_manipulation.py

Using numpy, matplotlib and scipy
The goal of this exercise is for you to become familiar with the important
packages numpy (for n-dimensional array manipulations) and matplotlib (for
matlab-like plotting and visualization).
Your task is to load a provided image into a numpy array and do some basic
manipulations.

You need to replace the ??? in the code with the required commands
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# use plt.imread to read in 'tree.jpg' as numpy array, select
# only the red channel. Check the image dimensions with img.shape.
# img_red should contain a 2-dimensional array
# Please note that this is actually a gray scale image and all 3
# image channels are the same.

img = plt.imread('tree.jpg')

# Choose the red color channel. For exploring the shape of your image type
# img.shape into your interpreter to know which axis of the array to take.
# Your img has the shape (640, 640, 3) for 640x640 pixels and 3 color channels.
# To leave the for example the first axis untouched type img[:, something]
# You can always check the result by img_red.shape -> should be (640, 640)

img_red = img[:, :, 0]

# show img_red with plt.imshow

plt.figure(1)
plt.imshow(img_red, cmap='gray')

# Using imread, the image values returns unsigned integers between 0 and 255
# Add a colorbar to verify the range of values.

plt.colorbar()

# Create a new numpy array that is the subarray containing only the tree in
# the image. Then invert the intensity values of the small subimage and call
# the resulting array img_crop_inv

# Use slicing to select the tree (get the coordinates by looking at the image)

img_crop = img_red[360:490, 360:490].copy()

# Invert the image by subtracting from its maximum. You can find the maximum by
# calling <picture>.max()

img_crop_inv = img_crop.max() - img_crop

plt.figure(2)
plt.imshow(img_crop_inv, cmap='gray')
plt.colorbar()

# apply a threshold to img_red to make a binary image separating the
# tree from its background

# Define a threshold. You can check by looking at the image colorbar, if your
# threshold is appropriate

threshold = 60

# Thresholding is possible by a simple "<" or ">" sign and the threshold value.
# You do not have to explicitly loop over the image

img_binary = img_red < threshold

plt.figure(3)
plt.imshow(img_binary, cmap='gray')

# Plot a vertical profile line through the tree of img_red
# Select a column of the image via slicing. Your result line_tree should be a
#  1D array of the shape 640.

line_tree = img_red[:, 425]

plt.figure(4)
plt.plot(line_tree)

# Generate a matrix that consists only of zeros of dimension 400 x 400 pixels

img_seg = np.zeros((400, 400))
cs = img_crop.shape
ss = img_seg.shape

# Place the subarray containing just the tree (img_crop) in the center of
# img_seg

img_seg[(ss[0]//2 - cs[0]//2):(ss[0]//2 + cs[0]//2),
        (ss[1]//2 - cs[1]//2):(ss[1]//2 + cs[1]//2)] = img_crop

# Have a look at img_seg with imshow

plt.figure(5)
plt.imshow(img_seg, cmap='gray')

# Use the function nd.rotate to rotate img_seg by 45 degrees
# Use nd.rotate? to see the function definition in ipython interpreter or
# help(nd.rotate) in python interpreter or look it up in spyder (recommended)

img_rot = nd.rotate(img_seg, 45, reshape = True)

# check if the shape of img_rot is the same as for img_seg, if not look at
# the additional paramters of the nd.rotate functions

plt.figure(6)
plt.imshow(img_rot, cmap='gray')
