"""
This code plots a interactive figure
Juanjuan Huang
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
def calc_spectrogram(signal, width, sigma, apply_gaussian = True):
    """
    Calculate the spectrogram of a 1D-signal via the 1D Windowed Fourier
    Transform. Smoothen the rectangular window of width "width" with a
    Gaussian of width "sigma"

    Parameters are: - the signal
                    - the width of the rect window
                    - the width of the apodizing gaussian

    Returns: spectrogram
    """

    signal = np.hstack([np.zeros(width//2), signal, np.zeros(width//2)])
    print(signal.shape)
    spectrogram = np.zeros((len(signal), len(signal)))
    check = np.arange(width, len(signal), width)
    counter = 0
    
    for position in np.arange(width//2, len(signal) - width//2):
        c = float(counter)/len(check)
        windowed_signal = signal[position-width//2:position + width//2]
        if apply_gaussian == True:
            gaussian = np.exp(-((np.linspace(0, width, width) - width//2)**2) / (2. * sigma**2))
        else:            
            gaussian = 1
        
        if position in check:
            plt.figure(5)
            plt.clf()
            plt.subplot(2,1,1)
            plt.plot(windowed_signal, label = 'windowed_signal', color = (1-c, c, c**2))
            #plt.plot(gaussian, color = 'orange',linewidth = 3)
            plt.legend(loc = 'center')
            counter += 1
            
        windowed_signal = windowed_signal * gaussian
        
        if position in check:
            plt.figure(5)
            plt.subplot(2,1,2)
            plt.plot(windowed_signal, label = 'windowed_signal * gaussian',color = (1-c, c, c**2))
            plt.plot(gaussian, color = 'orange', linestyle = '--', linewidth = 3)
            plt.legend(loc = 'center')
            plt.pause(.1)

        padded_window = np.zeros((len(signal)))

        padded_window[position-width//2:position + width//2] = windowed_signal
        local_spectrum = np.fft.fftshift(np.fft.fft(padded_window))
        
        if position in check:
            plt.figure(7)
            plt.clf()
            plt.subplot(3,1,2)
            plt.plot(padded_window, linewidth = 0.5, color= (1-c, c, c**2), label = 'padded')
            
            plt.subplot(3,1,1)
            plt.plot(signal, color = 'k', linewidth = 0.5, alpha = 0.5)
            plt.axvspan(position - width//2, 
                        position + width //2, 
                        #facecolor = plt.cm.Accent_r,                        
                        facecolor= (1-c, c, c**2),
                        alpha=0.9)
            plt.xticks([])
            plt.title('signal')
            
            plt.subplot(3,1,3)
            plt.plot(np.abs(local_spectrum) ** 2, color = (1-c, c, c**2))
            plt.xticks([])
            plt.pause(0.01)
            
        spectrogram[:, position] = np.abs(local_spectrum) ** 2
        
        if position in check:
            plt.figure(8)
            plt.clf()
            plt.imshow(spectrogram, 
                       vmin = 0, vmax = 300,
                       aspect='auto', cmap='gray')
            plt.title('spectrogram')
            plt.ylabel('frequency')
            plt.xlabel('time/space')
            plt.pause(0.01)
    return spectrogram[:, width//2:len(signal) - width//2]

#%%
# our signal
x = np.linspace(0, 1., 1000)
x2 = np.linspace(0, 4., 4000)
p1 = 1. / 80
p2 = 1. / 160
signal1 = np.cos(2 * np.pi / p1 * x)
signal2 = np.cos(2 * np.pi / p2 * x)
signal3 = (signal1 + signal2) / 2
signal4 = np.cos(2 * np.pi / p1 * x**2)

signal = np.hstack([signal1, signal2, signal3, signal4])

# In[1]:


# adjust width & sigma to see the diffrence of the results
# and the uncertainty principle --  a trade of frequency resolution & time resolution

# Adjust width values
#width = 50
width = 200
#width = 1000
#width = 250
#width = 400
sigma = width / 5.
spec = calc_spectrogram(signal, width, sigma)

#%%
# Adjust sigma values
# sigma too small
sigma = width / 20.
# sigma too big
sigma = width 

spec = calc_spectrogram(signal, width, sigma)

# In[2]:
# without applying Gaussian
width = 100

spec = calc_spectrogram(signal, width, sigma, apply_gaussian= False)
