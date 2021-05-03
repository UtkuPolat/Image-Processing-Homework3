import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

noiseImage1 = cv2.imread('/Users/utkupolat/Desktop/ImageProcessing_Homework3/images_for_HW3/noisy1.jpg',0)
noiseImage2 = cv2.imread('/Users/utkupolat/Desktop/ImageProcessing_Homework3/images_for_HW3/noisy2.jpg',0)
noiseImage3 = cv2.imread('/Users/utkupolat/Desktop/ImageProcessing_Homework3/images_for_HW3/noisy3.jpg',0)
def spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.imshow(magnitude_spectrum, cmap = 'gray')
    grid = plt.GridSpec(4, 4, wspace=0.1, hspace=0.1)
    plt.figure(figsize=(20,20))
    plt.subplot(grid[0, 0]),plt.imshow(img, cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(grid[0, 1]),plt.hist(img.ravel(),256,[0,256])
    plt.title('Histogram of Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(grid[0, 2]),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum of Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(grid[0, 3]),plt.hist(magnitude_spectrum.ravel(),256,[0,256])
    plt.title('Histogram of Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
spectrum(noiseImage1)
spectrum(noiseImage2)
spectrum(noiseImage3)

def function1(im, newsize=None):
    dft = np.fft.fft2(np.float32(im),newsize)
    return np.fft.fftshift(dft)
def function2(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)
def lowPass(img_in):
   # Write low pass filter here
   imf = function1(img_in, (int(img_in.shape[0]),int(img_in.shape[1]))) 
   mask = np.zeros((int(img_in.shape[0]),int(img_in.shape[1])),np.uint8)                                              
   mask[int(img_in.shape[0])/2-20:int(img_in.shape[0]/2+20), int(img_in.shape[1]/2-20):int(img_in.shape[1]/2+20)] = 1
   img_out = function2(imf*mask).astype('uint8')
   cv2.imwrite('Output.png', img_out)
   return True, img_out

lowPass(noiseImage1)
lowPass(noiseImage2)

def bandReject(i):
    # Read in the file
    file = FILEPATH('/Users/utkupolat/Desktop/ImageProcessing_Homework3/images_for_HW3/noisy1.jpg', SUBDIR=['noisy1','noisy2','noisy3'])
    imageOriginal = READ_PNG(file)
    # Generate some sinusoidal noise
    xCoords = LINDGEN(300,300) % 300
    yCoords = TRANSPOSE(xCoords)
    noise = -SIN(xCoords*1.5)-SIN(yCoords*1.5)
    imageNoise = imageOriginal + 50*noise
    # Filter the noise with a band reject filter
    imageFiltered = BANDREJECT_FILTER(imageNoise, 0.28, 0.38)
    plt.imshow(imageFiltered,cmap = 'gray')
    # Display the original, noise-added, and filtered images
    i=IMAGE(imageOriginal, LAYOUT=[3,1,1], TITLE='Original Image', DIMENSIONS=[700,300])
    i=IMAGE(imageFiltered, LAYOUT=[3,1,3], CURRENT, TITLE='Band Reject Filtered')

bandReject(noiseImage3)