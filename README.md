Scripts to generate a 3D Point Spread Function.

Ewald uses the method of the Ewald Sphere to create a 3D Ampliture Transfer Function. 
It works for standard and 4pi microscopes.
It accepts also a gaussian pupil with an effective NA smaller that the microscope NA. 
Recommended for high NA objective lenses

Defocus takes a 2D pupil and generate the Amplitute Transfer Function at different depths z, multiplying it with an angular spectrum propagator
(2D Fourier Transfor of the Rayleigh-Sommerfielf free space propagator).

It accepts abberrated pupils, with nm Zernike Polynomials weighted by a weight (1 corresponds to 1 lambda wavefrnt error).