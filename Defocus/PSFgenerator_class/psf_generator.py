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

class PSF_generator():
    '''
    Class to generate 3D Point Spread Functions
    with different pupils and various abberrations between the object and the lens. 
    
    '''
    
    def __init__(self, NA, n, wavelength, Nxy, Nz, over_sampling=4, aspect_ratio=1):
        '''
        NA: numerical aperture
        n: refractive index
        wavelength
        Nxy: number of pixels in kx-ky (pixels in x-y is Nxy-1)
        Nz: number of pixels in kz (and in z) 
        over_sampling: ratio between the Abbe resolution and spatial sampling
        aspect_ratio: ratio between z and xy sampling
        
        '''
        assert Nxy % 2 == 0, "XY number of pixels must be even" 
        
        self.NA = NA # Numerical aperture
        self.n = n # refraction index at the object
        self.wavelength = wavelength
        
        self.Nxy = Nxy
        self.Nz = Nz
        
        DeltaXY = wavelength/2/NA # Diffraction limited transverse resolution
        
        self.dr = DeltaXY/over_sampling 
        # spatial sampling in xy, chosen to be a fraction (over_sampling) of the resolution
        self.aspect_ratio = aspect_ratio
        self.dz = aspect_ratio * self.dr # spatial sampling in z
        
        # generate the real  space (xy is used only for visulaization) 
        # x = y = fftshift(fftfreq(Npixels, dk))
        self.x = self.y = self.dr * (np.arange(Nxy) - Nxy // 2)
        self.z = self.dz * (np.arange(self.Nz) - self.Nz // 2)
                
        self.k = n/wavelength # wavenumber
        self.k_cut_off = NA/wavelength # cut off frequency in the coherent case
        
        self.generate_kspace()
        
    def generate_kspace(self):
        """ Generates the k-space used to define the transfer functions
        """
        kx_lin = fftshift(fftfreq(self.Nxy, self.dr))
        ky_lin = fftshift(fftfreq(self.Nxy, self.dr))
        # kx_lin = dk * (np.arange(Nxy) - Nxy // 2)
        # ky_lin = dk * (np.arange(Nxy) - Nxy // 2)
        self.dk = kx_lin[1]-kx_lin[0]
        kx, ky = np.meshgrid(kx_lin,ky_lin) 

        # k-space in radial coordinates
        with np.errstate(invalid='ignore'):    
            self.k_rho = np.sqrt(kx**2 + ky**2)
            self.k_theta = np.arctan2(ky,kx)  
            self.kz = np.sqrt(self.k**2-self.k_rho**2)
        
        self.kx = kx
        self.ky = ky
        
        self.phase = np.zeros_like(self.kz)
        self.amplitude = np.ones_like(self.kz)
        
        self.ATF0 = np.zeros_like(self.kz) 
        self.PSF3D = np.zeros(((self.Nz,self.Nxy-1,self.Nxy-1)))

    def add_slab_scalar(self, n1, thickness, alpha):
        """ 
        Calculates the effect of the slab, using scalar theory, without considering s and p polarizations.
        n1: refractive index of the slab
        thickness: thickness
        alpha: angle from the xy plane, conventionally from the y axis 
        """
        
        n0 = self.n
        k= self.k
        k1 = n1/n0 * k
        
        ky=self.ky
        kx=self.kx
        kz=self.kz
            
        # note that, at alpha=0, k_rho remains the same in the slab, in agreement with Snell's law
        
        with np.errstate(invalid='ignore'):
            theta0 = np.arcsin(ky/k)
            theta1 = np.arcsin(n0/n1 * np.sin(theta0 + alpha)) - alpha
            ky1 = k1 * np.sin (theta1)
            k_rho1 = np.sqrt( kx**2 + ky1**2 )
            kz1 = np.sqrt( k1**2 - k_rho1**2 ) 
        
        # additional phase due to propagation in the slab
        phase = 2*np.pi * (kz1-kz) * thickness / np.cos(alpha) 
        
        # Fresnel law of refraction (not used and not important al low NA)
        # Ts01 = 2 * n0 * np.cos(theta0) / (n0* np.cos(theta0) + n1 * np.cos(theta1))
        # Tp01 = 2 * n0 * np.cos(theta0) / (n0* np.cos(theta1) + n1 * np.cos(theta0))
        # Ts10 = 2 * n1 * np.cos(theta1) / (n1* np.cos(theta1) + n0 * np.cos(theta0))
        # Tp10 = 2 * n1 * np.cos(theta1) / (n1* np.cos(theta0) + n0 * np.cos(theta1))
        # transmittance = (Ts01*Ts10 + Tp01*Tp10 ) / 2 # assuming equal s and p polarization components
        # self.amplitude *= transmittance
        
        self.phase += phase
        self.thickness = thickness
        self.alpha = alpha
        self.n1 = n1
        self.correct_slab_defocus()
        
    def correct_slab_defocus(self):    
        """ Calculates the displacement of the focus along z and y 
        as it was calculated by ray-tracing and changes the phase accordingly.
        Correcting for defocus does not change the PSF shape but recenters it around the origin
        """
        n0 = self.n
        NA= self.NA
        
        if hasattr(self, 'thickness'):
            thickness = self.thickness
            n1 = self.n1
            alpha = self.alpha
        else: 
            raise Exception('Slab parameters not specified')
            
        maxtheta0 = np.arcsin(NA/n0)
        maxtheta1 = np.arcsin(NA/n1)
        
        # diplacement along z (it is calculated at alpha==0)
        self.displacementZ = thickness * (1-np.tan(maxtheta1)/np.tan(maxtheta0))
        
        # calculate the displacement of a paraxial ray from the optical axis 
        # (this is correctly calculated as a fucntion of alpha)
        alpha1 = np.arcsin(n0/n1*np.sin(alpha))
        self.displacementY = - thickness/ np.cos(alpha1)*np.sin(alpha-alpha1)
        
        # correct for defocus, to recenter the PSF in z==0 and y==0 (the new ray-tracing focus point)
        self.phase +=  2*np.pi * self.kz * self.displacementZ
        self.phase +=  2*np.pi * self.ky * self.displacementY
        
        # remove piston
        phase = self.phase
        phase = phase[np.isfinite(phase)]
        self.phase= (self.phase - np.min(phase))
        
        
    def add_Zernike_aberration(self, N, M, weight):
        self.phase += weight*nm_polynomial(N, M, 
                                           self.k_rho/self.k_cut_off, 
                                           self.k_theta, normalized = True
                                           ) 
        
    def generate_pupil(self):
        """
        generate the pupil and the transfer functions 
        """
        ATF0 = self.amplitude * np.exp( 1.j*self.phase) 
        #ATF0 = np.exp( 1.j*phase) # Amplitude Transfer Function (pupil)
        cut_idx = (self.k_rho >= self.k_cut_off) # indexes of the evanescent waves (kz is NaN for these indexes)
        ATF0[cut_idx] = 0 # exclude k above the cut off frequency
        self.ATF0 = ATF0
        
    def add_Ndimensional_SIM_pupil(self,     
                     kr = 0.7,
                     waist = 0.01,
                     source_num = 3
                     ):
        ''' Generates the pupil for mutlidimensional SIM microscopy
            (typically 2,3 or 4 sources are used)
            kr: spatial frequency (radial component) of the sources, 
                relative to the  cutoff frequency self.k_cut_off
            waist: of the gaussian source,
                relative to the  cutoff frequency self.k_cut_off
            
            '''
        NumSources = 3    
        source_theta = 2*np.pi/source_num * np.arange(source_num)
        source_kr = [kr] * NumSources
        self.amplitude *= multiple_gaussians(self.kx/self.k_cut_off,
                                  self.ky/self.k_cut_off,
                                  waist, waist,
                                  source_kr, source_theta)  
    
    def add_lightsheet_pupil(self, 
                     waistx = 0.015,
                     waist_ratio = 20,
                     ):
        ''' Generates the pupil for light sheet illumination
        waistx: of the gaussian source along. 
        waist_ratio: ratio between the waist along y and the one along x
        If waist_x<<1 waist_ratio>>1, a light sheet is formed in the plane xz 
        
        '''
        waisty = waistx * waist_ratio
        beam= multiple_gaussians(self.kx/self.k_cut_off,
                                  self.ky/self.k_cut_off,
                                  waistx, waisty,
                                  [0.0], [0.0])  
        
        self.amplitude *=beam
        
    def add_lattice_pupil(self, 
                     cutin = 0.84,
                     cutout = 1,
                     waistx = 0.015,
                     waist_ratio = 20,
                     source_num = 3
                     ):
        ''' Generates the pupil for lattice light sheet microscopy
            All parameters are relative to the cutoff frequency self.k_cut_off
        cutin: minimum radial spatial frequency of the annular ring
        cutout: maximum radial spatial frequency of the annular ring
        waistx: of the gaussian source along x
        waist_ratio: ratio between the waist along y and the one along x
        source_num: order of the lattice
        
        '''
        
        source_rho = [(cutout+cutin)/2] * source_num # repeat list source_num times
        source_theta = 2*np.pi/source_num * np.arange(source_num)
        
        waisty = waistx * waist_ratio
        
        beams= multiple_gaussians(self.kx/self.k_cut_off,
                                  self.ky/self.k_cut_off,
                                  waistx, waisty,
                                  source_rho, source_theta)  
        
        cut_idx = (self.k_rho <= self.k_cut_off*cutin) | (self.k_rho >= self.k_cut_off*cutout)  
        mask = np.ones_like(self.k_rho)
        mask[cut_idx] = 0 # exclude k above the cut off frequency
        
        self.amplitude *=beams*mask
                
    def generate_3D_PSF(self): 
        
        ATF0 = self.ATF0
        
        for idx,zi in enumerate(self.z):
           
            angular_spectrum_propagator = np.exp(1.j*2*np.pi*self.kz*zi)
             
            ATF = ATF0 * angular_spectrum_propagator
        
            evanescent_idx = (self.k_rho > self.k)
            ATF[evanescent_idx] = 0 
            
            ASF = ifftshift(ifft2(ATF)) #* k**2/f**2 # Amplitude Spread Function
            ASF = ASF[1:,1:]
            
            PSF = np.abs(ASF)**2 # Point Spread Function
            
            self.PSF3D[idx,:,:] = PSF
       
    def _calculateRMS(self):
        '''calculates the RMS wavefron error
        For Zernike abberrations it is the weight of the 
        rms == 1 indicates 1-wavelength wavefront error 
        '''
        cut_idx = self.k_rho >= self.k_cut_off  
        area = self.ATF0.size - np.sum(cut_idx)
        phase = self.phase # /(2*np.pi)
        #m = np.zeros_like(phase)
        phase[cut_idx] = 0
        phase = phase[np.isfinite(phase)]
        rms = np.sqrt(np.sum(phase**2)/area) 
        return(rms)
       
    def print_values(self):
    
        DeltaXY = self.wavelength/2/self.NA # Diffraction limited transverse resolution
        DeltaZ = self.wavelength/self.n/(1-np.sqrt(1-self.NA**2/self.n**2)) # Diffraction limited axial resolution
    
        print(f'The numerical aperture of the system is: {self.NA}') 
        print(f'The transverse resolution is: {DeltaXY:.03f} um') 
        print(f'The axial resolution is: {DeltaZ:.03f} um') 
        print(f'The pixel size is: {self.dr:.03f} um') 
        print(f'The voxel depth is: {self.dz:.03f} um') 
        if hasattr(self, 'displacementZ'):
            print(f'The displacement z from the focus is: {self.displacementZ} um')
        if hasattr(self, 'displacementY'):
            print(f'The displacement y from the optical axis is: {self.displacementY} um')
        if hasattr(self, 'calculateRMS'):
            print(f'The RMS wavefront error is {self.calculateRMS()}')    
            
    def show_pupil(self):
        """ 
        Shows the Amplitude Transfer Function (pupil) in amplitude and phase
        """
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=False)
        sup_title =  f'NA = {self.NA}, n = {self.n}'
        if hasattr(self,'thickness'):
             sup_title += f', slab thickness = {self.thickness} $\mu$m, n1 = {self.n1}, alpha = {self.alpha:.02f}'
        fig.suptitle(sup_title)     
        
        for idx in range(2):
            if idx==0:
                data = np.abs(self.ATF0)
                title = 'Pupil (amplitude)'
            elif idx ==1:
                data = np.angle(self.ATF0)
                title = 'Pupil (phase)'
                    
        
            im0=ax[idx].imshow(data, 
                             cmap='pink',
                             extent = [np.amin(self.kx),np.amax(self.kx),np.amin(self.ky),np.amax(self.ky)],
                             origin = 'lower'
                             )
            ax[idx].set_xlabel('kx (1/$\mu$m)')
            ax[idx].set_ylabel('ky (1/$\mu$m)')
            ax[idx].set_title(title)
            fig.colorbar(im0,ax = ax[idx])
    
    def show_PSF_projections(self, aspect_ratio, mode ='MIP'):
        """ Shows the 3D PSF in 3 orthogonal views
        Aspect ratio: between the z axis and the other axes scale
        mode: 'MIP' or 'plane': MAximum intensity projection or plane incercepting the origin
        """
       
        fig, axs = plt.subplots(1, 3, figsize=(12,6), tight_layout=False)
        sup_title =  f'NA = {self.NA}, n = {self.n}'
        if hasattr(self,'thickness'):
             sup_title += f', slab thickness = {self.thickness} $\mu$m, n1 = {self.n1}, alpha = {self.alpha:.02f}'
        fig.suptitle(sup_title)
        
        label_list = ( ('x','y'),('x','z'),('y','z') )
        
        for idx, labels in enumerate(label_list):

            if mode =='MIP':
                # create maximum intensity projection
                MIP = np.amax(self.PSF3D,axis=idx)
                im_to_show = MIP
            elif mode =='plane':
                PSF = self.PSF3D
                Nz,Ny,Nx = PSF.shape
                Nlist = [Nz,Ny,Nx]
                im_to_show = PSF.take(indices=Nlist[idx]//2 , axis=idx)
            else:
                raise(ValueError, 'Please specify PSF showing mode' )
 
            values0 = getattr(self, labels[0])
            values1 = getattr(self, labels[1])
            delta0 = self.dr
            delta1 = self.dr if idx==0 else 0
            extent = [np.amin(values0)+delta0, np.amax(values0),
                      np.amin(values1)+delta1, np.amax(values1)]
            
            if idx == 0:
                vmin = np.amin(im_to_show)
                vmax = np.amax(im_to_show)
                
            
            axs[idx].imshow(im_to_show,
                             cmap='twilight',
                             extent = extent,
                             origin = 'lower',
                             vmin=vmin, vmax=vmax
                             )
            
            axs[idx].set_xlabel(f'{labels[0]} ($\mu$m)')
            axs[idx].set_ylabel(f'{labels[1]} ($\mu$m)')
            axs[idx].set_title(f'|PSF({labels[0]},{labels[1]})|')  
            
            if labels[1] == 'z': 
                axs[idx].set_aspect(1/aspect_ratio)
               
    def save_data(self):
        
        basename = 'psf'
        filename = '_'.join([basename,
                             f'NA_{self.NA}',
                             f'n_{self.n}'])
        
        if hasattr(self, 'thickness'):
            filename = '_'.join([filename,
                                 f'size_{self.thickness}',
                                 f'alpha_{self.alpha:.2f}',
                                 f'n1_{self.n1}'])
        
        
        from skimage.external import tifffile as tif
        psf16 = ( self.PSF3D * (2**16-1) / np.amax(self.PSF3D) ).astype('uint16') #normalize and convert to 16 bit
        psf16.shape = 1, self.Nz, 1, self.Nxy-1, self.Nxy-1, 1 # dimensions in TZCYXS order
        
        tif.imsave(filename+'.tif', psf16, imagej=True, resolution = (1.0/self.dr, 1.0/self.dr),
                    metadata={'spacing': self.dz, 'unit': 'um'})

if __name__ == '__main__':

    um = 1.0
    mm = 1000 * um
    deg = np.pi/180
    
    NA = 0.3
    
    wavelength = 0.518 * um 
    n0 = 1.33 # refractive index of the medium
    
    n1 = 1.5 # refractive index of the slab
    thickness = 170 * um # slab thickness
    alpha = 45 * deg # angle of the slab relative to the y axis
    
    SaveData = False
    
    Nxy = 256 
    Nz = 256
    
    aspect_ratio = 2 # ratio between z and xy sampling
    
    gen = PSF_generator(NA, n0, wavelength, Nxy, Nz, 
                        over_sampling = 4, aspect_ratio=aspect_ratio)
    
    # gen.add_Ndimensional_SIM_pupil()
    # gen.add_lattice_pupil()
    # gen.add_lightsheet_pupil()
    gen.add_slab_scalar(n1, thickness, alpha)
    # gen.add_slab_vectorial(n1, thickness, alpha)
    # gen.add_Zernike_aberration(5, 1, weight=0.5)

    
    gen.generate_pupil()
    gen.generate_3D_PSF()
    
    # Show results    
    gen.print_values()
    
    gen.show_pupil()
    gen.show_PSF_projections(aspect_ratio=1, mode='MIP') 
            
    if SaveData:
        gen.save_data()            
