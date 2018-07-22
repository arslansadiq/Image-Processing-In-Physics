"""
03.05.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex2_spline_order.py

Using numpy, matplotlib and scipy

The goal of this exercise is to get an idea of the frequency performance of
different interpolation algorithms. For simplicity, we will use the
scipy.ndimage.rotate function to rotate an image multiple times. The
function uses a special case of affine transform, where the re-gridding and
interpolation step is performed automatically.
The image should be rotated multiple times over 360 degrees. At each
rotation step an interpolation is necessary, and the cumulative effect of these
interpolations deteriorates the final result.

You need to replace the ??? in the code with the required commands
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# read an image using our matplotlib tools

img = plt.imread('tree.jpg') / 255.
img = np.mean(img, axis=2)
sh = np.shape(img)

# Define a full rotation, split into N seperate rotations

step = 1
Nsteps = 20
angle = 360 / Nsteps

# Crop the image to speed up the program
# eg 300x300 pixels with the tree in the centre

img_cropped = img[300:600, 300:600]

# In all cases start with the cropped image

img_order0 = img_cropped
img_order1 = img_cropped
img_order2 = img_cropped
img_order3 = img_cropped
img_order5 = img_cropped

# Creates a figure instance that will be updated in the loop. Change therefore
# the Graphics backend of the Ipython console in the preferences of Spyder from
# inline to automatic. For those who use ipython directly in the console, use
# the matplotlib command plt.ion() before plotting the figure or start ipython
# with an additional flag "python --pylab" for interactive plotting.

plt.figure(1)

while step <= Nsteps:

    # If you are using python 3, add parentheses around the arguments

    print('rotation No ' + str(step) + ' angle ' + str(step * angle))

    # Use ndi.rotate to rotate the image. Interpolation is done using splines
    # of certain order which can be passed as a variable. Please use order
    # 0 (nearest neighbor), 1 (bilinear), 2 (biquadratic), 3 (bicubic), and 5
    # also use the option reshape=False

    img_order0 = nd.interpolation.rotate(img_order0, angle=angle, order=0, reshape = False)
    img_order1 = nd.interpolation.rotate(img_order1, angle=angle, order=1, reshape = False)
    img_order2 = nd.interpolation.rotate(img_order2, angle=angle, order=2, reshape = False)
    img_order3 = nd.interpolation.rotate(img_order3, angle=angle, order=3, reshape = False)
    img_order5 = nd.interpolation.rotate(img_order5, angle=angle, order=5, reshape = False)

    # Plot the resulting images at the current step

    plt.subplot(231)
    plt.imshow(img_order0, cmap='gray', interpolation='none')
    plt.title('nearest neighbour')
    plt.subplot(232)
    plt.imshow(img_order1, cmap='gray', interpolation='none')
    plt.title('bilinear')
    plt.subplot(233)
    plt.imshow(img_order2, cmap='gray', interpolation='none')
    plt.title('biquadratic')
    plt.subplot(234)
    plt.imshow(img_order3, cmap='gray', interpolation='none')
    plt.title('bicubic')
    plt.subplot(235)
    plt.imshow(img_order5, cmap='gray', interpolation='none')
    plt.title('5th order')

    # You can use the function plt.pause to update your figure
    # During the calculation

    plt.pause(.1)

    # Increment the counter

    step = step+1

# Plot final results

plt.subplot(236)
plt.imshow(img_cropped, cmap='gray', interpolation='none')
plt.title('original')
