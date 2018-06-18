"""
14.06.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex1_deconvolution.py

Using numpy, matplotlib, scipy

The goal of this exercise is to try out some deconvolution tasks in the
presence of noise. First the "naive" deconvolution, then using the Wiener
filter.
You need to replace the ??? in the code with the required commands.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# load tree image

img = plt.imread('tree.jpg') / 255.
img = img.mean(axis=2)
sh = img.shape

# Let's create a convolution kernel (PSF) and produce the convolved image. We
# want the convolved image to suffer from motion blur in the direction of the
# diagonal. The function np.diag creates an appropriate convolution kernel.
# The kernel should be 51x51 pixels with 1 on the diagonal, 0 otherwise, and
# then normalized so that the sum of the diagonal is 1.

M = 51
psf = np.diag(np.ones(M))/M
img_conv = nd.convolve(img, psf, mode='wrap')

# Add zero-mean Gaussian noise with a standard deviation sigma to the image
# Hint: look at np.random.randn

sigma = .01
noise = sigma*np.random.randn(sh[0], sh[0])
img_noisy = img_conv + noise

# In order to use Fourier space deconvolution we need to zeropad our
# convolution kernel to the same size as the original image

psf_pad = np.zeros_like(img)
psf_pad[sh[0]//2-M//2:sh[0]//2+M//2+1, sh[1]//2-M//2:sh[1]//2+M//2+1] = psf

# Now we'll try out "naive" deconvolution by dividing the noisy, blurred image
# by the filter function in the Fourier domain

img_deconv = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(img_noisy)/np.fft.fft2(psf_pad))))

#img_deconv = np.fft.ifft2((np.fft.fft2(img_noisy)/np.fft.fft2(psf_pad))
# As soon as you add a little noise to the image, the naive deconvolution will
# go wrong, since for white noise, the noise power will exceed the signal
# power for high frequencies. Since the inverse filtering also enhances the
# power frequncies the result will be nonsense

# Let's first define the Wiener deconvolution in a seperate function.


def wiener_deconv(img, psf, nps):
    """
    This function performs an image deconvolution using a Wiener filter.

    Parameters
    ----------
    img : ndarray
      convolved image
    psf : ndarray
      the convolution kernel
    nps : float or ndarray
      noise power spectrum of the image, you will have to choose an
      appropriate value

    Returns
    -------
    deconvolved_image : ndarray
      The deconvolved image or volume depending on the input image.

    Notes
    -----
    If a float is given as nps, it assumes intrinsically white noise.
    A nps of 0 corresponds to no noise, and therefore naive image
    deconvolution.
    """
    # Apart from the noise power spectrum (nps), which is passed as a
    # parameter, you'll also need the frequency representation of your psf,
    # the power spectrum of the filter and the signal power spectrum (sps).
    # Calculate them.

    f_psf = np.fft.fft2(psf)
    sps_psf = np.abs(f_psf)**2
    sps = np.abs(np.fft.fft2(img))**2

    # create the Wiener filter

    wiener_filter = (1/f_psf)*(sps_psf/(sps_psf + (nps/sps)))

    # Do a Fourier space convolution of the image with the wiener filter

    deconv_img = np.fft.fftshift(np.real(np.fft.ifft2(
        np.fft.fft2(img) * wiener_filter)))

    return deconv_img

# Try out Wiener deconvolution.
# Assume white noise, i.e. a noise power spectrum that has a constant value
# for all frequencies. Try out a few values to get a good result.

nps = np.fft.fft2(sigma * np.random.randn(sh[0], sh[1]))

img_deconv_W = wiener_deconv(img_noisy, psf_pad, nps)

# The Wiener filter is essentially the same as the naive filter, only with an
# additional weighting factor that depends on the SNR in the image in
# frequency domain. Frequencies where the noise power exceeds the signal power
# will be damped.

plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', interpolation='none')
plt.title('original image')
plt.subplot(2, 2, 2)
plt.imshow(img_noisy, cmap='gray', interpolation='none')
plt.title('acquired noisy image')
plt.subplot(2, 2, 3)
plt.imshow(img_deconv, cmap='gray', interpolation='none')
plt.title('naive deconvolution')
plt.subplot(2, 2, 4)
plt.imshow(img_deconv_W, cmap='gray', interpolation='none')
plt.title('Wiener deconvolution')
