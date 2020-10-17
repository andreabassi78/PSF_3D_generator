# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:00:29 2020

Creates a 3D PSF starting from an abberrated pupil (with Zernike Polynomials)

@author: Andrea Bassi
"""

import numpy as np
from zernike_polynomials import nm_polynomial
from numpy.fft import ifft2, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt

um = 1.0
mm = 1000*um

Npixels = 256 # Pixels in x,y
assert Npixels%2 == 0 # Npixels must be even 

n = 1 # refractive index

wavelength = 0.520*um 

NA = 0.14

dr = 0.25 * um # spatial sampling in xy
dz = 2 * um

N = 4 # Zernike radial order
M = 0 # Zernike azimutal frequency
weight = 1.0 # weight of the Zernike abberration weight=1 means a wavefront error of lambda

SaveData = 'True'

# %% Start calculation

k = n/wavelength # wavenumber

k_cut_off = NA/wavelength # cut off frequency in the coherent case

DeltaXY = wavelength/2/NA # Diffraction limited transverse resolution
DeltaZ = wavelength/n/(1-np.sqrt(1-NA**2/n**2)) # Diffraction limited axial resolution
# DeltaZ = 2*n*wavelength/NA**2 # Fresnel approximation

# z position to be used: the range is 4 times the axial resolution
Nz = int(4*DeltaZ//dz)
zs = np.linspace(-2*DeltaZ, 2*DeltaZ, Nz)

# generate the k-space
kx_lin = fftshift(fftfreq(Npixels, dr))
ky_lin = fftshift(fftfreq(Npixels, dr))
# kx_lin = dk * (np.arange(Npixels) - Npixels // 2)
# ky_lin = dk * (np.arange(Npixels) - Npixels // 2)
dk = kx_lin[1]-kx_lin[0]
kx, ky = np.meshgrid(kx_lin,ky_lin) 

# #generate the real space (unused) 
x = y = dr * (np.arange(Npixels) - Npixels // 2)
#x = y = fftshift(fftfreq(Npixels, dk))

# k-space in radial coordinates
k_rho = np.sqrt(kx**2 + ky**2)
k_theta = np.arctan2(ky,kx)

# create a Zernike Polynomial to insert abberrations
phase = np.pi* nm_polynomial(N, M, k_rho/k_cut_off, k_theta, normalized = False) 

ATF0 = np.exp (1.j * weight * phase) # Amplitude Transfer Function
kz = np.sqrt(k**2-k_rho**2)

evanescent_idx = (k_rho >= k) # indexes of the evanescent waves (kz is NaN for these indexes)
# evanescent_idx = np.isnan(kz)
                    
PSF3D = np.zeros(((Nz,Npixels-1,Npixels-1)))
    
for idx,z in enumerate(zs):
   
    angular_spectrum_propagator = np.exp(1.j*2*np.pi*kz*z)
    
    angular_spectrum_propagator[evanescent_idx] = 0 # exclude evanescent k
    
    ATF = ATF0 * angular_spectrum_propagator

    mask_idx = (k_rho > k_cut_off)
    ATF[mask_idx] = 0 # Creates a circular mask
    
    ASF = ifftshift(ifft2(ATF)) #* k**2/f**2 # Amplitude Spread Function
    ASF = ASF[1:,1:]
    
    PSF = np.abs(ASF)**2 # Point Spread Function
    
    PSF3D[idx,:,:] = PSF
    
    if idx%50 == 0:
        fig, ax = plt.subplots(1, 2, figsize=(9, 5), tight_layout=False)
        fig.suptitle(f'Zernike coefficient ({N},{M}):{weight}, z={z:.2f}$\mu$m')
        
        im0=ax[0].imshow(np.angle(ATF), 
                         cmap='gray',
                         extent = [np.amin(kx),np.amax(kx),np.amin(ky),np.amax(ky)],
                         origin = 'lower'
                         )
        ax[0].set_xlabel('kx (1/$\mu$m)')
        ax[0].set_ylabel('ky (1/$\mu$m)')
        ax[0].set_title('Pupil (phase)')
        fig.colorbar(im0,ax = ax[0])
        
        im1=ax[1].imshow(PSF,
                         cmap='gray',
                         extent = [np.amin(x)+dr,np.amax(x),np.amin(y)+dr,np.amax(y)],
                         origin = 'lower'
                         )
        ax[1].set_xlabel('x ($\mu$m)')
        ax[1].set_ylabel('y ($\mu$m)')
        ax[1].set_title('PSF')
      
print('The numerical aperture of the system is:', NA) 
print('The transverse resolution is:', DeltaXY ,'um') 
print('The axial resolution is:', DeltaZ ,'um') 
print('The axial resolution is:', 2*n*wavelength/NA**2 ,'um, with Fresnel approximation') 
print('The pixel size is:', dr ,'um') 
print('The voxel depth is:', dz ,'um') 

if SaveData:
    
    if N !=0 or M != 0:
        note = f'aberratted_n{N}-m{M}_w{weight}'
    else: note = None
    
    basename = 'psf'
    filename = '_'.join(filter(None,[basename,f'NA{NA}',f'n{n}',note]))
    
    from skimage.external import tifffile as tif
    psf16 = ( PSF3D * (2**16-1) / np.amax(PSF3D) ).astype('uint16') #normalize and convert to 16 bit
    psf16.shape = 1, Nz, 1, Npixels-1, Npixels-1, 1 # dimensions in TZCYXS order
    
    tif.imsave(filename+'.tif', psf16, imagej=True, resolution = (1.0/dr, 1.0/dr),
                metadata={'spacing': dz, 'unit': 'um'})