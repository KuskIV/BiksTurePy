import logging
from PIL import Image 
import numpy as np
import cv2
class homo_noise():
    a = 0.5
    b = 1.5
    cut_off = 30
    fil_order = 2
    mode = 'butterworth'

    def __init__(self,config):
        self.Keys=['a','b','cutoff','filorder','mode']
        for key in self.Keys:
            if key in config:
                setattr(self, key, config.get(key))

    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered   

    def butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    def homomorfilter(self,img):

        I_log = np.log1p(np.array(img, dtype="float"))
        I_fft = np.fft.fft2(I_log)
        I_shape =  I_fft.shape

        if self.mode == 'butterbworth':
            H = self.butterworth_filter(I_shape,(self.cut_off,self.fil_order))

        if self.mode == 'gaussian':
            H = self.gaussian_filter(I_shape,(self.cut_off,self.fil_order))

        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1

        return np.uint8(I)

    def get_img_color_bands(self,img):
        temp = []
        arr = np.array(im, dtype=float)
        for c in range(3):
            temp.append(arr[:,:,c])
        return temp

    def homofy(self,img):
        temp = []
        color_bands = self.get_img_color_bands(img)
        for bands in color_bands:
            temp.append(self.homomorfilter(bands))
        for i in range(len(temp)):
            im = Image.fromarray(np.uint8(temp[i]))
            temp[i] = im.convert('L')

        return Image.merge("RGB", (temp[0], temp[1], temp[2]))

    def __add__(self, img):
        return self.homofy(img)

if __name__ == "__main__":

    filtred_bands = []
    temp = []
    im = Image.open(r"C:/Users/jeppe/Desktop/Homomorphic/2.png") 
    config = {'a':1.0,'b':1.5,'cutoff':30,'filorder':2,'mode':'gaussian'}
    homo = homo_noise(config)
    homo.homofy(im).save(r"C:/Users/jeppe/Desktop/Homomorphic/another_day_another_yeet.png")


