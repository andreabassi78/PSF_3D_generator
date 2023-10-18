# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:02:23 2021

@author: Andrea Bassi
"""

import numpy as np
from numpy.fft import ifft2, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
from zernike_polynomials import nm_polynomial
from SIM_pupil import multiple_gaussians
from psf_generator import PSF_generator


if __name__ == '__main__':

    um = 1.0
    mm = 1000 * um
    deg = np.pi/180
        
    NA = 1.4
    
    wavelength = 0.532 * um 
    n0 = 1.51 # refractive index of the medium
    
    n1 = 1.51 # refractive index of the slab
    thickness = 0 * um # slab thickness
    alpha = 0 * deg # angle of the slab relative to the y axis
    
    SaveData = False
    
    Nxy = 256
    Nz = 64
    
    aspect_ratio = 2 # ratio between z and xy sampling
    
    gen = PSF_generator(NA, n0, wavelength, Nxy, Nz, 
                        over_sampling = 4, aspect_ratio=aspect_ratio)
    
    M = 60 # magnification
    f= 200/M # lens focal lenght assuming a tube lens of 200
    
    r = np.arcsin(NA/n0)*f # radius of the pupil
    # r = NA/n0*f # radius of the pupil
    
    reff = 0.87 # measured radius at the pupil 
    
    kr = reff/r
    gen.add_Ndimensional_SIM_pupil(kr = kr,
                     waist = 0.02,
                     source_num = 3)
    
    
    period = wavelength/2/(NA*kr*np.sqrt(3)/2)
    print(f'Distance between two peaks: {period:.02f} um')
    print(f'Radius at the pupil: {r:.02f} mm')
    print(f'Effective radius at the pupil divided by the full pupil radius: {kr:.02f}')
        
    
    gen.add_slab_scalar(n1, thickness, alpha)
    
    gen.generate_pupil()
    gen.generate_3D_PSF()
    
    # Show results    
    # gen.print_values()
    
    gen.show_pupil()
    # gen.plot_phase()
    gen.show_PSF_projections(aspect_ratio=1, mode='plane') 
            
    if SaveData:
        gen.save_data()            
