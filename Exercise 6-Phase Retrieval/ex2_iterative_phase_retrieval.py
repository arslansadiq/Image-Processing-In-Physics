"""
07.06.2018
Image Processing Physics TU Muenchen
Julia Herzen, Klaus Achterhold, (Maximilian Teuffenbach, Juanjuan Huang)

ex2_iterative_phase_retrieval.py

Using numpy, matplotlib, scipy

Script to perform iterative phase retrieval to recover the phase at the
aperture plane from the intensity at the focal plane and the known support
magnitude in the aperture plane.

The chosen phase screen is simply a sum of tip and tilt modes, to ensure fast
convergence and no problems with phase wrapping, twin images or piston offset.

As per usual, replace the ???s with the appropriate command(s).
"""
import numpy as np
import matplotlib.pylab as plt

N = 512  # Square dimension of phase screen

# Calculate the support constraint (magnitude at the aperture plane)
# Generate a round mask and make sure that the radius is 128 pixels.
uu, vv = np.meshgrid(np.linspace(-N//2, N//2, N), np.linspace(-N//2, N//2, N))
aperture = (uu**2 + vv**2) < 128**2

# Plot your aperture function
plt.figure(1)
plt.imshow(aperture, cmap='gray')
plt.title('Support constraint')

# Generate the tip & tilt zernikes
x = range(N) - N/2*np.ones(N) + 0.5
y = range(N) - N/2*np.ones(N) + 0.5
xx, yy = np.meshgrid(x,y)
tip = xx / np.max(xx)
tip = tip * aperture
tilt = yy / np.max(yy)
tilt = tilt * aperture

# set the phase screen as a combination of tip and tilt
screen = tip*4. + tilt*3.

plt.figure(2)
plt.imshow(screen * aperture, cmap='jet')
plt.colorbar()
plt.title('Aperture phase')

# Propagate the phase screen from the aperture to the focal plane using
# Fraunhofer propagation.
# Hints - aperture is the magnitude, and screen is the phase
# You may need to use a fftshift here
# Intensity is the absolute value of field at the focal plane squared
speckle = np.abs(np.fft.fftshift(np.fft.fft2(aperture * np.exp(screen * -1j))))**2

# Plot the speckle image (zoomed in to show the centre)
plt.figure(3)
plt.imshow(speckle[N//2-32:N//2+32,N//2-32:N//2+32], aspect='auto',
    extent=(N//2-32,N//2+32,N//2-32,N//2+32), interpolation='none', cmap='gray')
plt.colorbar()
plt.title('Intensity')

nloops = 50  # Number of loops (iterations) to run the phase retrieval
# If your code doesn't converge in <50, there is something wrong! 

# Calculate the magnitude at the focal plane as a function of the intensity
focal_magnitude = np.sqrt(speckle)

# Initial guess for the focal plane
focal_plane = focal_magnitude * np.exp(1j*np.zeros((N, N)))

# Create empty arrays to store the values for the errors and the strehl
errors_aperture = np.zeros(nloops)
errors_focal = np.zeros(nloops)

for loop in np.arange(nloops):

    print(loop)

    # calculate the field at the aperture from the focal plane
    # using Fraunhofer (ifft2). May need an ifftshift here!
    aperture_plane = np.fft.ifft2(np.fft.ifftshift(focal_plane))

    # Enforce the support constraint in the aperture plane
    # ie zero all the points outside the known extent of the aperture    
    aperture_plane = aperture_plane*aperture

    # calculate the error in the apeture plane as the
    # difference between the amplitudes within the aperture
    errors_aperture[loop] = np.sum((np.abs(aperture_plane)-aperture)**2)

    # calculate the field at the focal plane from the aperture plane
    # using Fraunhofer (fft2). May need an fftshift here!
    focal_plane = np.fft.fftshift(np.fft.fft2(aperture_plane))

    # calculate the error in the focal plane as the
    # difference between the estimated magnitude and known magnitude
    errors_focal[loop] = np.sum((np.abs(focal_plane)-focal_magnitude)**2)
    
    # enforce the magnitude constraint at the focal plane
    focal_plane = focal_magnitude * np.exp(1j*np.angle(focal_plane))


# Plot the figures - compare with the lecture slides
plt.figure(4)
plt.imshow(np.angle(aperture_plane) * aperture, cmap='jet')
plt.title('Phase aperture plane')
plt.colorbar()

plt.figure(5)
plt.imshow(np.abs(aperture_plane) * aperture, cmap='gray')
plt.title('Magnitude aperture plane')
plt.colorbar()

plt.figure(6)
plt.imshow(np.angle(focal_plane), cmap='jet')
plt.title('Phase focal plane')
plt.colorbar()

plt.figure(7)
plt.imshow(np.abs(focal_plane)[N//2-32:N//2+32,N//2-32:N//2+32], aspect='auto',
    extent=(N//2-32,N//2+32,N//2-32,N//2+32), interpolation='none', cmap='gray')
plt.title('Magnitude focal plane')
plt.colorbar()

plt.figure(8)
plt.plot(np.log(errors_aperture))
plt.xlabel('Iteration')
plt.ylabel('Log Error')
plt.title('Error reduction - Aperture plane')

plt.figure(9)
plt.plot(np.log(errors_focal))
plt.xlabel('Iteration')
plt.ylabel('Log Error')
plt.title('Error reduction - Focal plane')
