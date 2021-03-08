# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:02:23 2021

Generate 3D PSF with different pupils and various abberrations
between the object and the lens. 

@author: Andrea Bassi
"""

import numpy as np
from numpy.fft import ifft2, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
from zernike_polynomials import nm_polynomial
from SIM_pupil import multiple_gaussians
from psf_generator import PSF_generator
from vectors import Vector
    

class vectorial_generator(PSF_generator):

    def add_slab_vectorial(self, n1, thickness, alpha):
        """ 
        Calculates the effect of the slab, using vectorial theory, considering s and p polarizations.
        n1: refractive index of the slab
        thickness: thickness
        alpha: angle from the xy plane, conventionally from the y axis 
        """
        
        n0 = self.n
        k= self.k
        k1 = n1/n0 * k
        
        kx=self.kx
        ky=self.ky
        kz=self.kz
        
        # k vector
        vk = Vector(kx,ky,kz)  
        uk = vk.norm
        
        ux = Vector(1,0,0).to_size_like(vk)
        uy = Vector(0,1,0).to_size_like(vk)
        uz = Vector(0,0,1).to_size_like(vk)
        
        
        # normal to the surface
        u = Vector(0, np.sin(alpha), np.cos(alpha)).to_size_like(vk)
        us = (uk.cross(u)).norm
        up = us.cross(uk)

        E0 = ux # assume initial polarization along x
        
        E = uk.cross(-uz.cross(E0))
        E= E/E.mag
 
        Es = E.dot(us)
        Ep = E.dot(up)
        
        print(E)

        # incidence angle, relative to the slab surface
        #phi0 = np.arccos(uk.cross(u).mag())
        phi0 = np.arcsin(np.abs(uk.dot(u)))
        
        #Snell's law:
        phi1 = np.arcsin(n0/n1 * np.sin(phi0))
        
        beta = phi0-phi1
        
        #uk1 = uk * np.cos(beta) - up * np.sin(beta)
        
        with np.errstate(invalid='ignore'):
            theta0 = np.arcsin(ky/k)
            theta1 = np.arcsin(n0/n1 * np.sin(theta0 + alpha)) - alpha
            ky1 = k1 * np.sin (theta1)
            k_rho1 = np.sqrt( kx**2 + ky1**2 )
            kz1 = np.sqrt( k1**2 - k_rho1**2 ) 
        
        # additional phase due to propagation in the slab
        phase =  (kz1-kz) * 2*np.pi * thickness / np.cos(alpha) 
        
        # Fresnel law of refraction
        Ts01 = 2 * n0 * np.cos(theta0) / (n0* np.cos(theta0) + n1 * np.cos(theta1))
        Tp01 = 2 * n0 * np.cos(theta0) / (n0* np.cos(theta1) + n1 * np.cos(theta0))
        Ts10 = 2 * n1 * np.cos(theta1) / (n1* np.cos(theta1) + n0 * np.cos(theta0))
        Tp10 = 2 * n1 * np.cos(theta1) / (n1* np.cos(theta0) + n0 * np.cos(theta1))
        transmittance = (Es*Ts01*Ts10 + Ep*Tp01*Tp10 ) # assuming equal s and p polarization components
        
        #bprint(transmittance)
        self.amplitude *= transmittance
        
        self.phase += phase
        self.thickness = thickness
        self.alpha = alpha
        self.n1 = n1
        self.correct_slab_defocus()
        

if __name__ == '__main__':

    um = 1.0
    mm = 1000 * um
    deg = np.pi/180
    
    NA = 0.1
    
    wavelength = 0.518 * um 
    n0 = 1.33 # refractive index of the medium
    
    n1 = 1.4 # refractive index of the slab
    thickness = 170 * um # slab thickness
    alpha = 0 * deg # angle of the slab relative to the y axis
    
    SaveData = False
    
    Nxy = 8
    Nz = 1
    
    aspect_ratio = 2 # ratio between z and xy sampling
    
    gen = vectorial_generator(NA, n0, wavelength, Nxy, Nz,
                              over_sampling=1, aspect_ratio=aspect_ratio)
    
    # gen.add_Ndimensional_SIM_pupil()
    # gen.add_lattice_pupil(cutin = 0.84, cutout = 1, 
    #                       waistx = 0.015,waist_ratio = 20,source_num = 4)
    # gen.add_lightsheet_pupil()
    gen.add_slab_vectorial(n1, thickness, alpha)
    # gen.add_slab_vectorial(n1, thickness, alpha)
    # gen.add_Zernike_aberration(3, 3, weight=1)
    
    
    
    gen.generate_pupil()
    gen.generate_3D_PSF()
    
    # Show results    
    #gen.print_values()
    gen.show_pupil()
    gen.show_PSF_projections(aspect_ratio=1, mode='MIP') 
            
    if SaveData:
        gen.save_data()            
