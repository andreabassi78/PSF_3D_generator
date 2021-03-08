# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:02:23 2021

Creates a 3D PSF aberrated by the presence of an inclined slab 
between the object and the lens. 

@author: Andrea Bassi
"""

import numpy as np
from numpy.fft import ifft2, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt

um = 1.0
mm = 1000 * um
deg = np.pi/180

Npixels = 256 # Pixels in x,y
assert Npixels % 2 == 0 # Npixels must be even 

n0 = 1.33 # refractive index of the medium
n1 = 1.51 # refractive index of the slab

thickness = 170 * um # slab thickness

wavelength = 0.518 * um 

alpha = 45 * deg # angle of the slab relative to the y axis

NA = 0.28

SaveData = False

# %% generate the spatial frequencies to propagate

DeltaXY = wavelength/2/NA # Diffraction limited transverse resolution

DeltaZ = wavelength/n0/(1-np.sqrt(1-NA**2/n0**2)) # Diffraction limited axial resolution

dr = DeltaXY/4 # spatial sampling in xy, chosen to be a fraction of the resolution
aspect_ratio = 2
dz = aspect_ratio * dr # spatial sampling in z

k = n0/wavelength # wavenumber

k_cut_off = NA/wavelength # cut off frequency in the coherent case

# z positions to be used: the range is z_extent_ratio times the depth of field
Nz= Npixels-1 # Nz can be chosen to have any value
zs = dz * (np.arange(Nz) - Nz // 2)

# generate the k-space
kx_lin = fftshift(fftfreq(Npixels, dr))
ky_lin = fftshift(fftfreq(Npixels, dr))
# kx_lin = dk * (np.arange(Npixels) - Npixels // 2)
# ky_lin = dk * (np.arange(Npixels) - Npixels // 2)
dk = kx_lin[1]-kx_lin[0]
kx, ky = np.meshgrid(kx_lin,ky_lin) 

# #generate the real space (used only for visulaization) 
x = y = dr * (np.arange(Npixels) - Npixels // 2)
#x = y = fftshift(fftfreq(Npixels, dk))

# k-space in radial coordinates
with np.errstate(invalid='ignore'):    
    k_rho = np.sqrt(kx**2 + ky**2)
    k_theta = np.arctan2(ky,kx)  
    kz = np.sqrt(k**2-k_rho**2)

# %% calculate the effect of the slab 

k1 = n1/n0 * k
# note that in the slab k_rho remains the same, in agreement with Snell's law

theta0 = np.arcsin(ky/k)                                                                                                                                                                                                                                                                                                                     

theta1 = np.arcsin(n0/n1 * np.sin(theta0 + alpha)) - alpha

ky1 = k1 * np.sin (theta1)

with np.errstate(invalid='ignore'):
    k_rho1 = np.sqrt( kx**2 + ky1**2 )
    kz1 = np.sqrt( k1**2 - k_rho1**2 ) 

# additional phase due to propagation in the slab
phase = 2*np.pi * (kz1-kz) * thickness / np.cos(alpha) 

# Fresnel law of refraction 
Ts01 = 2 * n0 * np.cos(theta0) / (n0* np.cos(theta0) + n1 * np.cos(theta1))
Tp01 = 2 * n0 * np.cos(theta0) / (n0* np.cos(theta1) + n1 * np.cos(theta0))
Ts10 = 2 * n1 * np.cos(theta1) / (n1* np.cos(theta1) + n0 * np.cos(theta0))
Tp10 = 2 * n1 * np.cos(theta1) / (n1* np.cos(theta0) + n0 * np.cos(theta1))

T = (Ts01*Ts10 + Tp01*Tp10 ) / 2 # assuming equal s and p polarization components

# %% calculate the displacement of the focus calculated by ray-tracing 

maxtheta0 = np.arcsin(NA/n0)
maxtheta1 = np.arcsin(NA/n1)

# diplacement along z (it is calculated at alpha==0)
displacementZ = thickness * (1-np.tan(maxtheta1)/np.tan(maxtheta0))

# calculate the displacement of a paraxial ray from the optical axis 
# (this is correctly calculated as a fucntion of alpha)
alpha1 = np.arcsin(n0/n1*np.sin(alpha))
displacementY = - thickness/ np.cos(alpha1)*np.sin(alpha-alpha1)

# correct for defocus, to recenter the PSF in z==0 and y==0 (the new ray-tracing focus point)
phase +=  2*np.pi * kz * displacementZ
phase +=  2*np.pi * ky * displacementY

# %% generate the pupil and the transfer functions """

ATF0 = T * np.exp( 1.j*phase) 
#ATF0 = np.exp( 1.j*phase) # Amplitude Transfer Function (pupil)
cut_idx = (k_rho >= k_cut_off) # indexes of the evanescent waves (kz is NaN for these indexes)

# evanescent_idx = np.isnan(kz)
ATF0[cut_idx] = 0 # exclude evanescent k
                    
PSF3D = np.zeros(((Nz,Npixels-1,Npixels-1)))

for idx,z in enumerate(zs):
   
    angular_spectrum_propagator = np.exp(1.j*2*np.pi*kz*z)
     
    ATF = ATF0 * angular_spectrum_propagator

    evanescent_idx = (k_rho > k_cut_off)
    ATF[evanescent_idx] = 0 
    
    ASF = ifftshift(ifft2(ATF)) #* k**2/f**2 # Amplitude Spread Function
    ASF = ASF[1:,1:]
    
    PSF = np.abs(ASF)**2 # Point Spread Function
    
    PSF3D[idx,:,:] = PSF
    
      
print('The numerical aperture of the system is:', NA) 
print('The transverse resolution is:', DeltaXY ,'um') 
print('The axial resolution is:', DeltaZ ,'um') 
print('The pixel size is:', dr ,'um') 
print('The voxel depth is:', dz ,'um') 
print(f'The displacement z from the focus is: {displacementZ} um')
print(f'The displacement y from the optical axis is: {displacementY} um')

# %% figure 1
fig1, ax = plt.subplots(1, 2, figsize=(9, 5), tight_layout=False)
fig1.suptitle(f'NA = {NA}, slab thickness = {thickness} $\mu$m, n0 = {n0}, n1 = {n1}')

im0=ax[0].imshow(np.angle(ATF0), 
                 cmap='gray',
                 extent = [np.amin(kx),np.amax(kx),np.amin(ky),np.amax(ky)],
                 origin = 'lower'
                 )
ax[0].set_xlabel('kx (1/$\mu$m)')
ax[0].set_ylabel('ky (1/$\mu$m)')
ax[0].set_title('Pupil (phase)')
fig1.colorbar(im0,ax = ax[0])
im1=ax[1].imshow(PSF,
                 cmap='gray',
                 extent = [np.amin(x)+dr,np.amax(x),np.amin(y)+dr,np.amax(y)],
                 origin = 'lower'
                 )
ax[1].set_xlabel('x ($\mu$m)')
ax[1].set_ylabel('y ($\mu$m)')
ax[1].set_title(f'PSF at z={z:.2f}$\mu$m')

# %% figure 2

plane_x = plane_y = (Npixels//2)
plane_z = (Nz//2)

fig2, axs = plt.subplots()
fig2.suptitle('Max intensity projection')

MIPz = np.amax(PSF3D,axis=0)
axs.set_title('|PSF(x,y)|')  
axs.set(xlabel = 'x ($\mu$m)')
axs.set(ylabel = 'y ($\mu$m)')
axs.imshow(MIPz, 
           extent = [np.amin(x)+dr,np.amax(x),np.amin(y)+dr,np.amax(y)],
           cmap='twilight'
           )


fig3, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=False)
fig3.suptitle('Max intensity projections')


MIPy = np.amax(PSF3D,axis=1)
axs[0].set_title('|PSF(x,z)|')  
axs[0].set(xlabel = 'x ($\mu$m)')
axs[0].set(ylabel = 'z ($\mu$m)')
axs[0].imshow(MIPy, 
              extent = [np.amin(x)+dr,np.amax(x),np.amin(zs),np.amax(zs)],
              cmap='twilight'
              )
axs[0].set_aspect(1/aspect_ratio)

MIPx = np.amax(PSF3D,axis=2)
axs[1].set_title('|PSF(y,z)|')  
axs[1].set(xlabel = 'y ($\mu$m)')
#axs[1].set(ylabel = 'z ($\mu$m)')
axs[1].imshow(MIPx,
              extent = [np.amin(x)+dr,np.amax(x),np.amin(zs),np.amax(zs)],
              cmap='twilight'
              )
axs[1].set_aspect(1/aspect_ratio)

if SaveData:
    
    basename = 'psf'
    filename = '_'.join(filter(None,[basename,f'NA_{NA}',
                                     f'size_{thickness}', f'alpha_{alpha:.2f}',
                                     f'n0_{n0}',f'n1_{n1}',f'lambda_{wavelength}']))
    
    from skimage.external import tifffile as tif
    psf16 = ( PSF3D * (2**16-1) / np.amax(PSF3D) ).astype('uint16') #normalize and convert to 16 bit
    psf16.shape = 1, Nz, 1, Npixels-1, Npixels-1, 1 # dimensions in TZCYXS order
    
    tif.imsave(filename+'.tif', psf16, imagej=True, resolution = (1.0/dr, 1.0/dr),
                metadata={'spacing': dz, 'unit': 'um'})