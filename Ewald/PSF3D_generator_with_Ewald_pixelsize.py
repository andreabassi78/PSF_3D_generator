# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:13:15 2020

Creates a 3D PSF starting from the Edward sphere

@author: Andrea Bassi
"""

import numpy as np 
from numpy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq
import matplotlib.pyplot as plt
from AmplitudeTransferFunction_3D import amplitude_transfer_function

um = 1.0   # base unit is um
mm = 1000 * um

N = 128  # the number of voxel in N**3
assert N%2 == 0 # N must be even

n = 1.33     # refractive index
NA = 1.1 # numerical aperture
wavelength = 0.520 * um

dr = 0.1 * um # spatial sampling in xyz

SaveData = 'False'

Detection_Mode = 'standard'
# choose between 'standard' and '4pi'

Microscope_Type = 'widefield'
# choose between: 'widefield', 'gaussian', 'bessel', 'SIM', 'STED', 'aberrated' 

waist = 1.5 * um

if Microscope_Type == 'gaussian':
    effectiveNA = wavelength/np.pi/waist
else: effectiveNA = NA


# %% Start calculation

K = n/wavelength # wavenumber

k_cut_off = NA/wavelength # cut off frequency in the coherent case

DeltaXY = wavelength/2/NA # Diffraction limited transverse resolution
DeltaZ = wavelength/n/(1-np.sqrt(1-NA**2/n**2)) # Diffraction limited axial resolution
# DeltaZ = 2*n*wavelength/NA**2 # Fresnel approximation

# generate the k-space
kx_lin = kylin = kzlin = fftshift(fftfreq(N, dr))
Kmin=min(kx_lin)
Kmax=max(kx_lin)

if K > Kmax:
    raise ValueError('k-frequencies not allowed, try reducing the voxel size dr')

dk = kx_lin[1]-kx_lin[0]


#%% generate the Amplitude Transfer Function (also called Coherent Transfer Function)
H = amplitude_transfer_function(N, Kmin, Kmax, n)
H.create_ewald_sphere(K)
H.set_numerical_aperture(NA, Detection_Mode)
H.set_microscope_type(NA, Microscope_Type, effectiveNA/NA)


# calculate the Spread and Transfer Function
ATF = H.values # 3D Amplitude Transfer Function

ASF = ifftshift(ifftn(fftshift(ATF))) # 3D Amplitude Spread Function 
ASF = ASF[1:,1:,1:]

PSF = np.abs(ASF)**2 # 3D Point Spread Function
PSF = PSF / np.sum(PSF) # Normalize the PSF on its eneregy

OTF = fftshift(fftn(ifftshift(PSF))) # 3D Optical Transfer Function

# show figures
plane=round(N/2)
epsilon = 1e-9 # to avoid calculating log 0 later
ATF_show = np.rot90( ( np.abs(ATF[plane,:,:]) ) )
ASF_show = np.rot90( ( np.abs(ASF[plane,:,:]) ) )
PSF_show = np.rot90( ( np.abs(PSF[plane,:,:]) ) )
OTF_show = np.rot90( ( np.abs(OTF[plane,:,:]) ) ) 

# set font size
plt.rcParams['font.size'] = 12

#create figure 1 and subfigures
fig1, axs = plt.subplots(2,2,figsize=(9,9))
fig1.suptitle(Detection_Mode + ' ' + Microscope_Type + ' microscope')

# recover the extent of the axes x,y,z
rmin = H.rmin+H.dr
rmax = H.rmax

# create subplot:
axs[0,0].set_title("|ASF(x,0,z)|")  
axs[0,0].set(ylabel = 'z ($\mu$m)')
axs[0,0].imshow(ASF_show, extent=[rmin,rmax,rmin,rmax])

# create subplot:
axs[0,1].set_title("|ATF($k_x$,0,$k_z$)|") 
axs[0,1].set(ylabel = '$k_z$ (1/$\mu$m)')
axs[0,1].imshow(ATF_show, extent=[Kmin,Kmax,Kmin,Kmax])

# create subplot:
axs[1,0].set_title('|PSF(x,0,z)|')  
axs[1,0].set(xlabel = 'x ($\mu$m)')
axs[1,0].set(ylabel = 'z ($\mu$m)')
axs[1,0].imshow(PSF_show, extent=[rmin,rmax,rmin,rmax])

# create subplot:
axs[1,1].set_title('log|OTF($k_x$,0,$k_z$)|')  
axs[1,1].set(xlabel = '$k_x$ (1/$\mu$m)')
axs[1,1].set(ylabel = '$k_z$ (1/$\mu$m)')
axs[1,1].imshow(OTF_show, extent=[Kmin,Kmax,Kmin,Kmax])

# finally, render the figures
plt.show()    

print('The numerical aperture of the system is:', effectiveNA) 
print('The transverse resolution is:', wavelength/2/effectiveNA ,'um') 

if Detection_Mode == 'standard':
    print('The axial resolution is:', wavelength/n/(1-np.sqrt(1-effectiveNA**2/n**2)) ,'um') 
    print('The axial resolution is:', 2*n*wavelength/effectiveNA**2 ,'um, with Fresnel approximation') 

if SaveData:
    voxel_size = H.dr
    print('The voxel size is:',voxel_size,'um')
    from skimage.external import tifffile as tif
    psf16 = np.transpose(PSF,(2,0,1))
    psf16 = ( psf16 * (2**16-1) / np.amax(psf16) ).astype('uint16') #normalize and convert to 16 bit
    psf16.shape = 1, N-1, 1, N-1, N-1, 1 # dimensions in TZCYXS order
    sampling = voxel_size
    tif.imsave(f'Ewald_NA{effectiveNA}_n{n}.tif', psf16, imagej=True, resolution = (1.0/sampling, 1.0/sampling),
                metadata={'spacing': sampling, 'unit': 'um'})