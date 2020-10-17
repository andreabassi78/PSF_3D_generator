'''
Created on 28 jul 2019
@author: Andrea Bassi, Politecnico di Milano
Lecture on 3D Optical Transfer Functions and Ewald Sphere. 
Optical Microscopy Course (Biophotonics)
'''

import numpy as np 
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import matplotlib.pyplot as plt
import time
from AmplitudeTransferFunction_3D import amplitude_transfer_function
from xlrd.formula import num2strg

N = 64      # sampling number
um = 1.      # base unit is um

n = 1       # refractive index

NA = 0.14  # numerical aperture

wavelength = 0.520 * um

print('The numerical aperture of the system is:', NA) 
print('The transverse resolution is:', wavelength/2/NA ,'um') 
print('The axial resolution is:', wavelength/n/(1-np.sqrt(1-NA**2/n**2)) ,'um') 
print('The axial resolution is:', 2*n*wavelength/NA**2 ,'um, with Fresnel approximation') 

K = n / wavelength

Kextent = 1.01*K # extent of the k-space

Detection_Mode = 'standard'
#choose between 'standard' and '4pi'

Microscope_Type = 'widefield'
# choose between: 'widefield', 'gaussian', 'bessel', 'SIM', 'STED', 'aberrated' 

SaveData = True

# Generate the Amplitude Transfer Function (or Coherent Transfer Function)
t0=time.time() # this is to calculate the execution time

H = amplitude_transfer_function(N, Kextent, n)

voxel_size = H.dr
extent = H.xyz_extent
print('The voxel size is', voxel_size,'um') 

H.create_ewald_sphere(K)
H.set_numerical_aperture(NA, Detection_Mode)
pupil, psf_xy0 = H.set_microscope_type(NA, Microscope_Type)

ATF = H.values # 3D Amplitude Transfer Function

ASF = ifftshift(ifftn(fftshift(ATF))) * N**3 #
# 3D Amplitude Spread Function (normalized for the total volume)

PSF = np.abs(ASF)**2 # 3D Point Spread Function

OTF = fftshift(fftn(ifftshift(PSF))) # 3D Optical Transfer Function

print('Elapsed time for calculation: ' + num2strg( time.time()-t0) + 's' )



############################
#####    Show figures

plane=round(N/2)
epsilon = 1e-9 # to avoid calculating log 0 later
ATF_show = np.rot90( ( np.abs(ATF[plane,:,:]) ) )
ASF_show = np.rot90( ( np.abs(ASF[plane,:,:]) ) )
PSF_show = np.rot90( ( np.abs(PSF[plane,:,:]) ) )
OTF_show = np.rot90( 10*np.log10 ( np.abs(OTF[plane,:,:]) + epsilon ) ) 


# set font size
plt.rcParams['font.size'] = 12

#create figure 1 and subfigures
fig1, axs = plt.subplots( 2, 2, figsize= (9,9) )
fig1.suptitle(Detection_Mode + ' ' + Microscope_Type + ' microscope')

# Recover extent of axes x,y,z (it is the inverse of the k sampling xyz_extent = 1/(2*dK))
Rmax=H.xyz_extent


zoom_factor = 1 #Kextent / K

# create subplot:
axs[0,0].set_title("|ASF(x,0,z)|")  
axs[0,0].set(ylabel = 'z ($\mu$m)')
axs[0,0].imshow(ASF_show, vmin = ASF_show.min(), vmax = ASF_show.max(), 
                 extent=[-Rmax,Rmax,-Rmax,Rmax])
axs[0,0].xaxis.zoom(zoom_factor) 
axs[0,0].yaxis.zoom(zoom_factor)

# create subplot:
axs[0,1].set_title("|ATF($k_x$,0,$k_z$)|") 
axs[0,1].set(ylabel = '$k_z$ (1/$\mu$m)')
axs[0,1].imshow(ATF_show ,  extent=[-Kextent,Kextent,-Kextent,Kextent])

# create subplot:
axs[1,0].set_title('|PSF(x,0,z)|')  
axs[1,0].set(xlabel = 'x ($\mu$m)')
axs[1,0].set(ylabel = 'z ($\mu$m)')
axs[1,0].imshow(PSF_show, vmin = PSF_show.min(), vmax = PSF_show.max(), 
                 extent=[-Rmax,Rmax,-Rmax,Rmax])
axs[1,0].xaxis.zoom(zoom_factor) 
axs[1,0].yaxis.zoom(zoom_factor)

# create subplot:
axs[1,1].set_title('log|OTF($k_x$,0,$k_z$)|')  
axs[1,1].set(xlabel = '$k_x$ (1/$\mu$m)')
axs[1,1].set(ylabel = '$k_z$ (1/$\mu$m)')
axs[1,1].imshow(OTF_show, extent=[-Kextent,Kextent,-Kextent,Kextent])

# finally, render the figures
plt.show()

############################
##### Save Psf to .tif file

if SaveData:
    
    
    from skimage.external import tifffile as tif
    psf16 = np.transpose(PSF,(2,0,1))
    psf16 = ( psf16 * (2**16-1) / np.amax(psf16) ).astype('uint16') #normalize and convert to 16 bit
    psf16.shape = 1, N, 1, N, N, 1 # dimensions in TZCYXS order
    sampling = voxel_size
    tif.imsave(f'psf_{NA}.tif', psf16, imagej=True, resolution = (1.0/sampling, 1.0/sampling),
                metadata={'spacing': sampling, 'unit': 'um'})