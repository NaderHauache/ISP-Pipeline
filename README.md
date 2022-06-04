# ISP-Pipeline
An Open Source ISP Pipeline which converts RAW images into sRGB Images wrote in Python. 

This work was done with the purpose of provide a simple, open source and efficient ISP Pipeline into the educational context of Digital Image Process. This first release of software was conceived from my undergraduate thesys at Federal University of Amazonas to acquire the Computer Engineer Bachelors Degree. 

Basically, this script works aims to provide, in the first version, a basic ISP Pipeline with the main modules to provide an sRGB image into .bmp format, it means, there is a lot of futher improvements to reach a high performance software with final high end quality. 

On this version 1.0-alpha, i did the implementation 8 moduludes of DIP:

1. Loader Module(Read and Load Image to a numpy array)
3. BLC module (to perform the correction of Black Levels of the image with a linearization process)
4. White Balance Adjstument Module (to correct the White Levels of the image)
5. Channel Discrimination Module (to provide an .bmp representation of each channel that composes the image into Bayer Filter Format, aka CFA image)
6. Demosaicing Module (using a Malvar2004 method implemented by colour demosaic library)
7. XYZ to sRGB Module(to provide a Color Space Conversion)
8. Gamma Correction Module(to correct the colors of the image)

As futher works to be done, an own implementation of a Demosaicing process is intersting, such as Lens Shade Correction Process, Noise Reduction Treatment, 4-channel Bayer Filters support(Usually using diffent tons of green into a 2x2 array), Sharpening Process, Contrast and Brightness Adjustment, Auto White Balance Algorithm(instead use of values provided by camera trough metadata), support to anothers CFA arrays and monochromatic sensors could be instersting too...

The Python language was choosen due the easily write philosophy, but unfornately, this language has serious problems in the matter of performance. In the future, as a continued project, Rust Language is a nice candidate to refactoring this script. 

On this version, it was used the following libs:
1. RawPy 
2. NumPy
3. ImageIOv2
4. colour_demosaicing

Even with the support of RawPy to diverse extensions of image, this work was tested only with .ARW files provided by Sony Cameras. So, the work with another extensions was not guaranteed. Despite that, you will need to adapt this code to your necessities. 

Feel free to use it, this project is licenced under GPL v3.

Best Regards
