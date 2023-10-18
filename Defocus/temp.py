# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:30:01 2022

@author: andrea
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.fft import fft2,fftshift,ifft2,ifftshift

Npixels = 256 # number of pixels
R = 1 # extent of the xy space
x = y = np.linspace(-R, +R, Npixels)
X, Y = np.meshgrid(x,y)
kmin = 2

deltak = 2

Z = np.cos(2*np.pi*(kmin+deltak*np.abs(R-Y)) *X ) **2

fftZ = fftshift(fft2(ifftshift(Z)))

ps = fftZ**2

im = ifftshift(ifft2(ps))




for z in (Z,np.abs(fftZ),np.abs(ps),np.abs(im)):
    plt.figure(figsize=(9, 9))
    plt.imshow(z, 
               interpolation='none',
               cmap=cm.gray,
               origin='lower',
               extent = [-R,R,-R,R]
               )
    
    plt.colorbar()
